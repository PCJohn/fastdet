// fastdet_score.cpp -- C++17 SIMD runtime for a fastdet single-file model.
//
// Reads the FDT1 container produced by fastdet (a JSON header followed by an
// IMSY symmetric-tree blob), bins a native-resolution float fixture, scores
// every cell, and checks the two retained scoring paths against each other
// (and, when given, against a recorded expected output).
//
// SIMD is Google Highway (static dispatch, as in imfeat): the target is chosen by
// the compiler flags, so one source runs SSE4/AVX2/AVX-512 on x86 and NEON/SVE on
// Arm.  Build with CMake (fetches Highway 1.2.0, same pin as imfeat):
//   cmake -S cpp -B build && cmake --build build --config Release
//
// Usage:
//   fastdet_score model.fdt fixture.f32 [expected.f32] [iters=50] [stages]
//
// `stages` switches on the optional early exit and times it next to the full evaluation:
// "trees:theta,trees:theta,..." drops a pack of tiles once `trees` trees have run and every raw
// sum in it is below `theta`.  Thresholds must be calibrated on data.
//
// fixture.f32 holds each kept feature at its native resolution, in model
// order: feature f is constant on blocks of a side x side grid (side = 64 >>
// level_shift[f] / 2; 1 for image-wide features) and stores side * side
// row-major float32 values.  This is Detector.native_matrix(image); no dense
// 4096 x n_features table is ever built on this path.
//
// There are exactly two scoring paths:
//   * score_cells_scalar -- straightforward reference, one cell at a time, on the
//                           fixture expanded to all 64 x 64 cells.
//   * score_cells_simd   -- the shipped path.  Cells are scored in 4 x 4 tiles; the
//                           splits that are constant on a tile (most of them: the
//                           features form a pyramid) are evaluated once per tile and
//                           only choose which group of leaves the tile can reach,
//                           and the leaf values are fetched with byte shuffles
//                           rather than gathers.  See "Tiled scorer" below.
// and one binner (bin_tiles) that bins each native value of each USED feature once.
// The vector path must be bit-identical to the scalar path.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"

namespace hn = hwy::HWY_NAMESPACE;

namespace {

constexpr size_t kGrid = 64;                // output grid side
constexpr size_t kCells = kGrid * kGrid;    // 4096 cells per image
constexpr uint8_t kMaxBin = 15;             // bins are table indices: at most 16 per feature
constexpr uint8_t kMaxShift = 12;           // level_shift of an image-wide (1x1) feature

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

double median(std::vector<double>& v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

std::string read_all(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return std::string();
  std::fseek(f, 0, SEEK_END);
  const long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string s((size_t)size, '\0');
  if (size > 0 && std::fread(&s[0], 1, (size_t)size, f) != (size_t)size) s.clear();
  std::fclose(f);
  return s;
}

inline uint16_t rd_u16(const uint8_t* p) {
  uint16_t v;
  std::memcpy(&v, p, sizeof v);  // not a cast: the blob gives no alignment guarantee
  return v;
}

inline uint32_t rd_u32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

// Extract the IMSY blob from either an FDT1 container or a bare IMSY blob.
bool find_blob(const std::string& raw, const uint8_t** blob, size_t* blob_size) {
  if (raw.size() < 12) return false;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(raw.data());
  if (std::memcmp(p, "FDT1", 4) == 0) {
    uint32_t header_len;
    std::memcpy(&header_len, p + 8, 4);
    if (raw.size() < 12 + header_len) return false;
    *blob = p + 12 + header_len;
    *blob_size = raw.size() - 12 - header_len;
    return true;
  }
  if (std::memcmp(p, "IMSY", 4) == 0) {  // accept a bare blob too
    *blob = p;
    *blob_size = raw.size();
    return true;
  }
  return false;
}

struct Stage {
  uint32_t trees;
  float theta;
};

struct ImysModel {
  uint32_t version = 0, n_trees = 0, n_features = 0, n_leafs_total = 0, depth = 0;
  uint32_t leaf_bits = 8, leaf_chunk = 16, coarse_trees = 0, n_chunks = 0;
  int32_t e_min = 0;
  std::vector<uint32_t> tree_offsets, tree_base;
  std::vector<uint16_t> split_feat;
  std::vector<uint8_t> split_bin;
  std::vector<uint8_t> codes;      // leaf codes, < 2**leaf_bits
  std::vector<float> offsets;      // per tree
  std::vector<uint8_t> shifts;     // per chunk: step = 2**(e_min + shift)
  std::vector<Stage> stages;       // early-exit stages calibrated at fit time
  std::vector<uint32_t> n_borders, border_offset;
  std::vector<float> borders;
  std::vector<uint8_t> level_shift;
  std::vector<uint32_t> chunk_of;  // per tree
  double offset_sum = 0.0;

  uint32_t chunk_start(uint32_t c) const { return chunk_starts[c]; }
  uint32_t chunk_end(uint32_t c) const { return c + 1 < n_chunks ? chunk_starts[c + 1] : n_trees; }
  std::vector<uint32_t> chunk_starts;
};

bool load_imys(const uint8_t* p, size_t size, ImysModel* m) {
  auto rd32 = [&](size_t off) { uint32_t v; std::memcpy(&v, p + off, 4); return v; };
  if (size < 48 || std::memcmp(p, "IMSY", 4) != 0) return false;
  m->version = rd32(4);
  if (m->version != 4) { std::fprintf(stderr, "IMSY version %u; this scorer reads version 4\n", m->version); return false; }
  m->n_trees = rd32(8); m->n_features = rd32(12); m->n_leafs_total = rd32(16); m->depth = rd32(20);
  m->leaf_bits = rd32(24); m->leaf_chunk = rd32(28); m->coarse_trees = rd32(32);
  std::memcpy(&m->e_min, p + 36, 4);
  m->n_chunks = rd32(40);
  const uint32_t n_stages = rd32(44);
  size_t off = 48;
  auto take = [&](size_t bytes) { const uint8_t* q = p + off; off += bytes; return q; };
  if (m->depth == 0 || m->depth > 8 || (m->leaf_bits != 4 && m->leaf_bits != 8) || m->leaf_chunk == 0 || m->leaf_chunk > 17) return false;
  m->tree_offsets.resize(m->n_trees + 1); std::memcpy(m->tree_offsets.data(), take(4 * (m->n_trees + 1)), 4 * (m->n_trees + 1));
  m->tree_base.resize(m->n_trees + 1); std::memcpy(m->tree_base.data(), take(4 * (m->n_trees + 1)), 4 * (m->n_trees + 1));
  const size_t n_splits = static_cast<size_t>(m->n_trees) * m->depth;
  m->split_feat.resize(n_splits); m->split_bin.resize(n_splits);
  for (size_t i = 0; i < n_splits; ++i) { const uint8_t* q = take(4); std::memcpy(&m->split_feat[i], q, 2); m->split_bin[i] = q[2]; }
  m->codes.resize(m->n_leafs_total); std::memcpy(m->codes.data(), take(m->n_leafs_total), m->n_leafs_total);
  m->offsets.resize(m->n_trees); std::memcpy(m->offsets.data(), take(4 * m->n_trees), 4 * m->n_trees);
  m->shifts.resize(m->n_chunks); std::memcpy(m->shifts.data(), take(m->n_chunks), m->n_chunks);
  for (uint32_t i = 0; i < n_stages; ++i) { const uint8_t* q = take(8); Stage s; std::memcpy(&s.trees, q, 4); std::memcpy(&s.theta, q + 4, 4); m->stages.push_back(s); }
  m->n_borders.resize(m->n_features); std::memcpy(m->n_borders.data(), take(4 * m->n_features), 4 * m->n_features);
  m->border_offset.resize(m->n_features + 1);
  size_t total = 0;
  for (uint32_t f = 0; f < m->n_features; ++f) { m->border_offset[f] = static_cast<uint32_t>(total); total += m->n_borders[f]; if (m->n_borders[f] > kMaxBin) return false; }
  m->border_offset[m->n_features] = static_cast<uint32_t>(total);
  m->borders.resize(total); std::memcpy(m->borders.data(), take(4 * total), 4 * total);
  m->level_shift.resize(m->n_features); std::memcpy(m->level_shift.data(), take(m->n_features), m->n_features);
  if (off != size) { std::fprintf(stderr, "IMSY blob: %zu trailing bytes\n", size - off); return false; }
  // chunks: leaf_chunk trees each, restarting at coarse_trees
  m->chunk_of.resize(m->n_trees);
  m->chunk_starts.clear();
  const uint32_t starts[3] = {0, m->coarse_trees, m->n_trees};
  for (int s = 0; s < 2; ++s) {
    const uint32_t lo = starts[s], hi = starts[s + 1];
    for (uint32_t t = lo; t < hi; t += m->leaf_chunk) {
      for (uint32_t u = t; u < std::min(t + m->leaf_chunk, hi); ++u) m->chunk_of[u] = static_cast<uint32_t>(m->chunk_starts.size());
      m->chunk_starts.push_back(t);
    }
  }
  if (m->chunk_starts.size() != m->n_chunks) { std::fprintf(stderr, "IMSY blob: %zu chunks, header says %u\n", m->chunk_starts.size(), m->n_chunks); return false; }
  for (uint32_t t = 0; t < m->n_trees; ++t) m->offset_sum += m->offsets[t];
  return true;
}

inline uint8_t bin_one(const ImysModel& m, uint32_t f, float x) {
  const float* a = m.borders.data() + m.border_offset[f];
  const uint32_t n = m.n_borders[f];
  uint32_t lo = 0, hi = n;
  while (lo < hi) {
    const uint32_t mid = (lo + hi) >> 1;
    if (a[mid] < x) lo = mid + 1; else hi = mid;
  }
  return (uint8_t)lo;
}

void bin_full(const ImysModel& m, const float* X, size_t n_cells, uint8_t* B) {
  for (size_t c = 0; c < n_cells; ++c) {
    const float* row = X + c * m.n_features;
    for (uint32_t f = 0; f < m.n_features; ++f)
      B[(size_t)f * n_cells + c] = bin_one(m, f, row[f]);
  }
}

// Side of the square grid feature f is stored at: 64 >> (level_shift / 2), so a
// level-L feature has L x L values and an image-wide feature (shift 12) has one.
uint32_t native_side(const ImysModel& m, uint32_t f) {
  return m.level_shift.empty() ? kGrid : static_cast<uint32_t>(kGrid >> (m.level_shift[f] >> 1));
}

// Float offset of each feature in a native fixture (n_features + 1 entries).
std::vector<size_t> native_offsets(const ImysModel& m) {
  std::vector<size_t> offset(m.n_features + 1, 0);
  for (uint32_t f = 0; f < m.n_features; ++f) {
    const size_t side = native_side(m, f);
    offset[f + 1] = offset[f] + side * side;
  }
  return offset;
}

// bins[i] = number of cuts below x[i] (searchsorted-left), one float lane per value.
// A true comparison mask is all-ones, i.e. -1 as int32, so subtracting it counts it.  The
// cut count K is a template parameter: features have few cuts, so a loop over them costs
// as much as the comparisons, and unrolled the cuts stay broadcast in registers.
template <uint32_t K>
void bin_run_k(const float* x, uint32_t n, const float* cuts, uint8_t* bins) {
  const hn::ScalableTag<float> df;
  const hn::RebindToSigned<decltype(df)> di;
  const hn::Rebind<uint8_t, decltype(di)> dq;
  const uint32_t lanes = static_cast<uint32_t>(hn::Lanes(df));
  hn::VFromD<decltype(df)> cut[K];
  for (uint32_t j = 0; j < K; ++j) cut[j] = hn::Set(df, cuts[j]);
  uint32_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    const auto v = hn::LoadU(df, x + i);
    auto count = hn::Zero(di);
    for (uint32_t j = 0; j < K; ++j)
      count = hn::Sub(count, hn::BitCast(di, hn::VecFromMask(df, hn::Gt(v, cut[j]))));
    hn::StoreU(hn::DemoteTo(dq, count), dq, bins + i);
  }
  for (; i < n; ++i) {
    uint32_t count = 0;
    for (uint32_t j = 0; j < K; ++j) count += (x[i] > cuts[j]) ? 1u : 0u;
    bins[i] = static_cast<uint8_t>(count);
  }
}

void bin_run(const float* x, uint32_t n, const float* cuts, uint32_t k, uint8_t* bins) {
  switch (k) {
    case 0:
      std::memset(bins, 0, n);
      return;
    case 1:
      return bin_run_k<1>(x, n, cuts, bins);
    case 2:
      return bin_run_k<2>(x, n, cuts, bins);
    case 3:
      return bin_run_k<3>(x, n, cuts, bins);
    case 4:
      return bin_run_k<4>(x, n, cuts, bins);
    case 5:
      return bin_run_k<5>(x, n, cuts, bins);
    case 6:
      return bin_run_k<6>(x, n, cuts, bins);
    case 7:
      return bin_run_k<7>(x, n, cuts, bins);
    case 8:
      return bin_run_k<8>(x, n, cuts, bins);
    case 9:
      return bin_run_k<9>(x, n, cuts, bins);
    case 10:
      return bin_run_k<10>(x, n, cuts, bins);
    case 11:
      return bin_run_k<11>(x, n, cuts, bins);
    case 12:
      return bin_run_k<12>(x, n, cuts, bins);
    case 13:
      return bin_run_k<13>(x, n, cuts, bins);
    case 14:
      return bin_run_k<14>(x, n, cuts, bins);
    default:
      return bin_run_k<kMaxBin>(x, n, cuts, bins);
  }
}

// Reference-only: expand a native fixture to the dense cells x features matrix.
std::vector<float> expand_native(const ImysModel& m, const float* xn,
                                 const std::vector<size_t>& offset) {
  const uint32_t nf = m.n_features;
  std::vector<float> x(kCells * nf);
  for (uint32_t f = 0; f < nf; ++f) {
    const uint32_t side = native_side(m, f);
    const uint32_t factor = static_cast<uint32_t>(kGrid) / side;
    for (uint32_t r = 0; r < kGrid; ++r)
      for (uint32_t c = 0; c < kGrid; ++c)
        x[(static_cast<size_t>(r) * kGrid + c) * nf + f] =
            xn[offset[f] + static_cast<size_t>(r / factor) * side + c / factor];
  }
  return x;
}

void sigmoid(const std::vector<float>& raw, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp(-raw[i]));
}

// Raw score of a cell from its integer total: the runtimes agree to the bit because the total
// is an integer and this is the one float expression that turns it into a score.
inline float raw_of(const ImysModel& m, int64_t total) {
  return static_cast<float>(m.offset_sum + std::ldexp(static_cast<double>(total), m.e_min));
}

// Reference: every tree on every cell in order, integer codes, no SIMD.
void score_cells_scalar(const ImysModel& m, const uint8_t* B, size_t n_cells, float* out) {
  std::vector<int64_t> total(n_cells, 0);
  for (uint32_t t = 0; t < m.n_trees; ++t) {
    const size_t s0 = static_cast<size_t>(t) * m.depth;
    const uint8_t* code = m.codes.data() + m.tree_offsets[t];
    const int shift = m.shifts[m.chunk_of[t]];
    for (size_t c = 0; c < n_cells; ++c) {
      uint32_t idx = 0;
      for (uint32_t d = 0; d < m.depth; ++d) idx |= static_cast<uint32_t>(B[static_cast<size_t>(m.split_feat[s0 + d]) * n_cells + c] > m.split_bin[s0 + d]) << d;
      total[c] += static_cast<int64_t>(code[idx]) << shift;
    }
  }
  std::vector<float> raw(n_cells);
  for (size_t c = 0; c < n_cells; ++c) raw[c] = raw_of(m, total[c]);
  sigmoid(raw, out, n_cells);
}

constexpr size_t kTile = 4;                       // tile side, in cells
constexpr size_t kTileCells = kTile * kTile;      // 16 cells: one 128-bit vector of byte lanes
constexpr size_t kTileGrid = kGrid / kTile;       // 16 x 16 tiles
constexpr size_t kTiles = kTileGrid * kTileGrid;  // 256
constexpr uint8_t kVaryShift = 2;                 // level_shift <= 2 (side 64, 32) varies in a tile
constexpr uint32_t kShuffleBits = 4;              // varying splits one byte shuffle can index
constexpr uint32_t kBlendBits = 5;                // ... and with two shuffles and a blend

struct TiledModel {
  std::vector<int32_t> fine_slot, coarse_slot;  // per feature: its plane, or -1 if unused
  uint32_t n_fine = 0, n_coarse = 0;
  std::vector<uint8_t> nvary;                       // per tree: v (varying splits inside a tile)
  std::vector<uint32_t> plane;                      // per split, varying first: its plane slot
  hwy::AlignedFreeUniquePtr<uint8_t[]> table;       // per split: 16 bytes, bin -> its index bit
  std::vector<uint8_t> code;                        // leaf codes in permuted order
  std::vector<int32_t> code32;                      // the same, widened, for per-tile gathers
  hwy::AlignedFreeUniquePtr<uint8_t[]> nib;         // per leaf group: NP nibble planes of the codes
};

bool varies_in_tile(const ImysModel& m, uint32_t f) {
  return m.level_shift.empty() || m.level_shift[f] <= kVaryShift;
}

// Reorders every tree's splits so the ones that vary inside a tile take the low index bits,
// permutes its leaf codes to match, and lays the codes of each leaf group out as nibble
// planes: plane k of group g holds ((code >> 4k) & 15) for the group's 2^v leaves, so a byte
// shuffle fetches a 4-bit slice of 16 (or 32) leaf codes at once.
TiledModel build_tiled(const ImysModel& m) {
  TiledModel tm;
  const uint32_t depth = m.depth, np = m.leaf_bits / 4;
  tm.fine_slot.assign(m.n_features, -1);
  tm.coarse_slot.assign(m.n_features, -1);
  tm.nvary.resize(m.n_trees);
  tm.plane.resize(static_cast<size_t>(m.n_trees) * depth);
  tm.table = hwy::AllocateAligned<uint8_t>(static_cast<size_t>(m.n_trees) * depth * 16);
  tm.code.resize(m.n_leafs_total);
  tm.code32.resize(m.n_leafs_total);
  tm.nib = hwy::AllocateAligned<uint8_t>(static_cast<size_t>(m.n_leafs_total) * np + 64);
  std::vector<uint32_t> order(depth);
  for (uint32_t t = 0; t < m.n_trees; ++t) {
    const size_t s0 = static_cast<size_t>(t) * depth;
    uint32_t v = 0, n = 0;  // varying splits take the low index bits, in their original order
    for (uint32_t d = 0; d < depth; ++d)
      if (varies_in_tile(m, m.split_feat[s0 + d])) order[n++] = d, ++v;
    for (uint32_t d = 0; d < depth; ++d)
      if (!varies_in_tile(m, m.split_feat[s0 + d])) order[n++] = d;
    tm.nvary[t] = static_cast<uint8_t>(v);
    for (uint32_t j = 0; j < depth; ++j) {
      const uint32_t f = m.split_feat[s0 + order[j]];
      const uint8_t bin = m.split_bin[s0 + order[j]];
      int32_t& slot = j < v ? tm.fine_slot[f] : tm.coarse_slot[f];
      if (slot < 0) slot = static_cast<int32_t>(j < v ? tm.n_fine++ : tm.n_coarse++);
      tm.plane[s0 + j] = static_cast<uint32_t>(slot);
      // varying splits set index bits 0..v-1; tile-constant ones set the group's bits 0..depth-v-1
      uint8_t* table = tm.table.get() + (s0 + j) * 16;
      const uint32_t bit = j < v ? j : j - v;
      for (uint32_t b = 0; b < 16; ++b) table[b] = b > bin ? static_cast<uint8_t>(1u << bit) : 0;
    }
    const uint32_t n_leaves = 1u << depth;
    const uint8_t* src = m.codes.data() + m.tree_offsets[t];
    uint8_t* dst = tm.code.data() + m.tree_offsets[t];
    for (uint32_t idx = 0; idx < n_leaves; ++idx) {  // new index -> old index
      uint32_t old_idx = 0;
      for (uint32_t j = 0; j < depth; ++j) old_idx |= ((idx >> j) & 1u) << order[j];
      dst[idx] = src[old_idx];
      tm.code32[m.tree_offsets[t] + idx] = src[old_idx];
    }
    uint8_t* planes = tm.nib.get() + static_cast<size_t>(m.tree_offsets[t]) * np;
    const uint32_t width = 1u << v, groups = n_leaves / width;
    for (uint32_t g = 0; g < groups; ++g)
      for (uint32_t k = 0; k < np; ++k)
        for (uint32_t i = 0; i < width; ++i) planes[(g * np + k) * width + i] = static_cast<uint8_t>((dst[g * width + i] >> (4 * k)) & 15);
  }
  return tm;
}

void bin_tiles(const ImysModel& m, const TiledModel& tm, const float* xn,
               const std::vector<size_t>& offset, uint8_t* fine, uint8_t* coarse, bool skip_side64 = false) {
  uint8_t rows[kTile][kGrid];
  uint8_t small[kTileGrid * kTileGrid];
  for (uint32_t f = 0; f < m.n_features; ++f) {
    if (tm.fine_slot[f] < 0 && tm.coarse_slot[f] < 0) continue;
    if (skip_side64 && tm.fine_slot[f] >= 0 && tm.coarse_slot[f] < 0 && native_side(m, f) == kGrid) continue;
    const uint32_t k = m.n_borders[f];
    const float* cuts = m.borders.data() + m.border_offset[f];
    const float* xf = xn + offset[f];
    const uint32_t side = native_side(m, f);
    if (tm.fine_slot[f] >= 0) {
      // Four 64-cell rows hold 16 tiles; tile j is column group j of each row, so a block of
      // four tiles is a 4 x 4 transpose of 32-bit lanes.  A side-32 feature is first doubled
      // along the row, and each of its rows serves two cell rows.
      uint8_t* dst = fine + static_cast<size_t>(tm.fine_slot[f]) * kCells;
      const hn::FixedTag<uint8_t, 16> db;
      const hn::FixedTag<uint32_t, 4> du;
      for (uint32_t tr = 0; tr < kTileGrid; ++tr) {
        const uint8_t* row[kTile];
        if (side == kGrid) {
          for (uint32_t i = 0; i < kTile; ++i) {
            bin_run(xf + static_cast<size_t>(tr * kTile + i) * side, side, cuts, k, rows[i]);
            row[i] = rows[i];
          }
        } else {
          uint8_t half[kGrid / 2];
          for (uint32_t i = 0; i < 2; ++i) {
            bin_run(xf + static_cast<size_t>(tr * 2 + i) * side, side, cuts, k, half);
            for (uint32_t c = 0; c < kGrid / 2; c += 16) {
              const auto v = hn::LoadU(db, half + c);
              hn::StoreU(hn::InterleaveLower(db, v, v), db, rows[i] + 2 * c);
              hn::StoreU(hn::InterleaveUpper(db, v, v), db, rows[i] + 2 * c + 16);
            }
            row[2 * i] = row[2 * i + 1] = rows[i];
          }
        }
        uint8_t* tiles = dst + static_cast<size_t>(tr) * kTileGrid * kTileCells;
        for (uint32_t c = 0; c < kGrid; c += 16, tiles += 4 * kTileCells) {
          const auto r0 = hn::BitCast(du, hn::LoadU(db, row[0] + c)),
                     r1 = hn::BitCast(du, hn::LoadU(db, row[1] + c));
          const auto r2 = hn::BitCast(du, hn::LoadU(db, row[2] + c)),
                     r3 = hn::BitCast(du, hn::LoadU(db, row[3] + c));
          const auto t0 = hn::InterleaveLower(du, r0, r1), t1 = hn::InterleaveUpper(du, r0, r1);
          const auto t2 = hn::InterleaveLower(du, r2, r3), t3 = hn::InterleaveUpper(du, r2, r3);
          hn::StoreU(hn::BitCast(db, hn::ConcatLowerLower(du, t2, t0)), db, tiles);
          hn::StoreU(hn::BitCast(db, hn::ConcatUpperUpper(du, t2, t0)), db, tiles + kTileCells);
          hn::StoreU(hn::BitCast(db, hn::ConcatLowerLower(du, t3, t1)), db, tiles + 2 * kTileCells);
          hn::StoreU(hn::BitCast(db, hn::ConcatUpperUpper(du, t3, t1)), db, tiles + 3 * kTileCells);
        }
      }
    }
    if (tm.coarse_slot[f] >= 0) {
      uint8_t* dst = coarse + static_cast<size_t>(tm.coarse_slot[f]) * kTiles;
      if (side == 1) {
        uint8_t one = 0;
        bin_run(xf, 1, cuts, k, &one);
        std::memset(dst, one, kTiles);
        continue;
      }
      uint32_t shift = 0;  // log2(tiles per native value, per side)
      while ((side << shift) < kTileGrid) ++shift;
      bin_run(xf, side * side, cuts, k, small);
      for (uint32_t r = 0; r < side; ++r) {
        uint8_t* row = dst + (static_cast<size_t>(r) << shift) * kTileGrid;
        for (uint32_t c = 0; c < kTileGrid; ++c) row[c] = small[r * side + (c >> shift)];
        for (uint32_t dr = 1; dr < (1u << shift); ++dr)
          std::memcpy(row + dr * kTileGrid, row, kTileGrid);
      }
    }
  }
}

// One tree's view of a tile pass: everything the inner loop touches, resolved to pointers.
struct TiledTree {
  uint32_t v = 0;
  const uint8_t* table = nullptr;   // 16 bytes per split, varying first
  const uint8_t* fine[8] = {};      // plane of each varying split (4096 bytes, tile-major)
  const uint8_t* nib = nullptr;     // this tree's nibble planes
  const uint8_t* group = nullptr;   // leaf group per tile
};

#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
constexpr size_t kPack = 2;  // tiles per index vector (32 byte lanes)
#else
constexpr size_t kPack = 1;
#endif
using PackB = hn::FixedTag<uint8_t, 16 * kPack>;
using PackW = hn::FixedTag<uint16_t, 8 * kPack>;

// One 16-byte table per tile of the pack, side by side.
HWY_INLINE hn::VFromD<PackB> pack_tables(const uint8_t* const* base, size_t offset) {
  const hn::FixedTag<uint8_t, 16> dq;
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
  return hn::Combine(PackB(), hn::LoadU(dq, base[1] + offset), hn::LoadU(dq, base[0] + offset));
#else
  return hn::LoadU(dq, base[0] + offset);
#endif
}

// One tree over the active packs: the index is built in byte lanes (one shuffle per varying
// split), and each nibble plane of the leaf codes is fetched with one shuffle per plane and
// added into byte-lane sums.  Trees of a chunk share the sums (at most 17 * 15 per lane), which
// are widened once per chunk.  For v >= 5 the 32-entry (or wider) group is two or more
// 16-entry sub-tables merged by a blend on the high index bits.
template <uint32_t V, uint32_t NP>
HWY_NOINLINE void tree_codes_over_packs(const TiledTree& tr, const uint16_t* active, size_t n_active, uint8_t* bsum) {
  const PackB db;
  hn::VFromD<PackB> table[V];
  const uint8_t* plane[V];
  for (uint32_t j = 0; j < V; ++j) { table[j] = hn::LoadDup128(db, tr.table + j * 16); plane[j] = tr.fine[j]; }
  constexpr size_t width = static_cast<size_t>(1) << V, subs = width > 16 ? width / 16 : 1;
  const uint8_t* const nib = tr.nib;
  const uint8_t* const group = tr.group;
  const auto low = hn::Set(db, 15);
  for (size_t a = 0; a < n_active; ++a) {
    const size_t tile = active[a], cell0 = tile * kTileCells;
    auto idx = hn::TableLookupBytes(table[0], hn::LoadU(db, plane[0] + cell0));
    for (uint32_t j = 1; j < V; ++j) idx = hn::Or(idx, hn::TableLookupBytes(table[j], hn::LoadU(db, plane[j] + cell0)));
    const uint8_t* base[kPack];
    for (size_t k = 0; k < kPack; ++k) base[k] = nib + static_cast<size_t>(group[tile + k]) * NP * width;
    hn::VFromD<PackB> sel[3];
    if constexpr (V > 4) {
      for (uint32_t b = 4; b < V; ++b) sel[b - 4] = hn::VecFromMask(db, hn::TestBit(idx, hn::Set(db, static_cast<uint8_t>(1u << b))));
      idx = hn::And(idx, low);
    }
    for (uint32_t k = 0; k < NP; ++k) {
      hn::VFromD<PackB> t[subs];
      for (size_t sub = 0; sub < subs; ++sub) t[sub] = hn::TableLookupBytes(pack_tables(base, k * width + sub * 16), idx);
      if constexpr (V > 4)
        for (uint32_t level = 0; level < V - 4; ++level)
          for (size_t i = 0; i < (subs >> (level + 1)); ++i) t[i] = hn::IfThenElse(hn::MaskFromVec(sel[level]), t[2 * i + 1], t[2 * i]);
      uint8_t* sum = bsum + k * kCells + cell0;
      hn::StoreU(hn::Add(hn::LoadU(db, sum), t[0]), db, sum);
    }
  }
}

template <uint32_t NP>
void run_tree(const TiledTree& tr, const uint16_t* active, size_t n_active, uint8_t* bsum) {
  switch (tr.v) {
    case 1: tree_codes_over_packs<1, NP>(tr, active, n_active, bsum); break;
    case 2: tree_codes_over_packs<2, NP>(tr, active, n_active, bsum); break;
    case 3: tree_codes_over_packs<3, NP>(tr, active, n_active, bsum); break;
    case 4: tree_codes_over_packs<4, NP>(tr, active, n_active, bsum); break;
    case 5: tree_codes_over_packs<5, NP>(tr, active, n_active, bsum); break;
    case 6: tree_codes_over_packs<6, NP>(tr, active, n_active, bsum); break;
    default: tree_codes_over_packs<7, NP>(tr, active, n_active, bsum); break;
  }
}

// Lazy binning: one side-64 feature for the packs still alive, straight from the native
// fixture (4 rows x 4*kPack floats per pack), written in the pack's tile-major order.
void bin_side64_packs(const float* xf, const float* cuts, uint32_t k, const uint16_t* active, size_t n_active, uint8_t* plane) {
  const size_t cols = kTile * kPack;
  alignas(64) float vals[kTile * kTile * kPack];
  alignas(64) uint8_t bins[kTile * kTile * kPack];
  for (size_t i = 0; i < n_active; ++i) {
    const size_t tile = active[i], tr = tile / kTileGrid, tc = tile % kTileGrid;
    for (size_t r = 0; r < kTile; ++r) std::memcpy(vals + cols * r, xf + (kTile * tr + r) * kGrid + kTile * tc, cols * sizeof(float));
    bin_run(vals, static_cast<uint32_t>(kTile * cols), cuts, k, bins);
    uint8_t* dst = plane + tile * kTileCells;
    for (size_t p = 0; p < kPack; ++p)
      for (size_t r = 0; r < kTile; ++r)
        for (size_t c = 0; c < kTile; ++c) dst[p * kTileCells + r * kTile + c] = bins[cols * r + kTile * p + c];
  }
}

void bin_side64_lazy(const ImysModel& m, const TiledModel& tm, const float* xn, const std::vector<size_t>& offset,
                     const uint16_t* active, size_t n_active, uint8_t* fine) {
  for (uint32_t f = 0; f < m.n_features; ++f) {
    if (tm.fine_slot[f] < 0 || native_side(m, f) != kGrid) continue;
    bin_side64_packs(xn + offset[f], m.borders.data() + m.border_offset[f], m.n_borders[f], active, n_active,
                     fine + static_cast<size_t>(tm.fine_slot[f]) * kCells);
  }
}

// Everything but the side-64 features (streaming): coarse planes, side-32 fine planes.
void bin_except_side64(const ImysModel& m, const TiledModel& tm, const float* xn, const std::vector<size_t>& offset, uint8_t* fine, uint8_t* coarse) {
  bin_tiles(m, tm, xn, offset, fine, coarse, /*skip_side64=*/true);
}

// The scorer: chunk by chunk (the quantisation chunks are the passes), coarse trees once per
// tile, fine trees on the packs still alive, integer totals, early exit at the blob's stages.
// With `lazy` the side-64 features are binned only for the packs alive after the coarse tier.
struct ScoreOptions {
  const std::vector<Stage>* stages = nullptr;  // nullptr: no early exit
  bool lazy = false;                           // bin side-64 features after the coarse tier
  size_t* packs_finished = nullptr;
};

template <uint32_t NP>
void score_cells_simd(const ImysModel& m, const TiledModel& tm, const float* xn, const std::vector<size_t>& offset,
                      uint8_t* fine, uint8_t* coarse, float* out, const ScoreOptions& opt) {
  const hn::ScalableTag<uint8_t> d8;
  const hn::ScalableTag<int64_t> d64;
  const hn::Rebind<int32_t, decltype(d64)> d32;
  const hn::Rebind<uint8_t, decltype(d64)> dq;
  const size_t l64 = hn::Lanes(d64);
  const uint32_t depth = m.depth;
  const bool lazy = opt.lazy && m.coarse_trees > 0;
  if (lazy) bin_except_side64(m, tm, xn, offset, fine, coarse);
  else bin_tiles(m, tm, xn, offset, fine, coarse);
  auto total = hwy::AllocateAligned<int64_t>(kCells);  // tile-major: tile * 16 + cell
  auto bsum = hwy::AllocateAligned<uint8_t>(NP * kCells);
  auto group = hwy::AllocateAligned<uint8_t>(kTiles + 64);
  std::vector<int32_t> flat(kTiles);
  std::fill(total.get(), total.get() + kCells, int64_t{0});
  std::vector<uint16_t> active(kTiles / kPack);
  for (size_t a = 0; a < active.size(); ++a) active[a] = static_cast<uint16_t>(a * kPack);
  size_t next_stage = 0;
  double offsets_done = 0.0;
  bool side64_binned = !lazy;
  for (uint32_t c = 0; c < m.n_chunks; ++c) {
    const uint32_t t0 = m.chunk_start(c), t1 = m.chunk_end(c);
    if (!side64_binned && t0 >= m.coarse_trees) {  // the fine tier starts: bin what it needs where it runs
      bin_side64_lazy(m, tm, xn, offset, active.data(), active.size(), fine);
      side64_binned = true;
    }
    std::memset(bsum.get(), 0, NP * kCells);
    std::fill(flat.begin(), flat.end(), 0);
    for (uint32_t t = t0; t < t1; ++t) {
      const size_t s0 = static_cast<size_t>(t) * depth;
      TiledTree tr;
      tr.v = tm.nvary[t];
      tr.table = tm.table.get() + s0 * 16;
      for (uint32_t j = 0; j < tr.v; ++j) tr.fine[j] = fine + static_cast<size_t>(tm.plane[s0 + j]) * kCells;
      tr.nib = tm.nib.get() + static_cast<size_t>(m.tree_offsets[t]) * NP;
      uint8_t* g = group.get();
      tr.group = g;
      if (tr.v == depth) for (size_t i = 0; i < kTiles; i += hn::Lanes(d8)) hn::Store(hn::Zero(d8), d8, g + i);
      for (uint32_t j = tr.v; j < depth; ++j) {  // tile-constant splits: the leaf group per tile
        const auto table = hn::LoadDup128(d8, tm.table.get() + (s0 + j) * 16);
        const uint8_t* bins = coarse + static_cast<size_t>(tm.plane[s0 + j]) * kTiles;
        if (j == tr.v) for (size_t i = 0; i < kTiles; i += hn::Lanes(d8)) hn::Store(hn::TableLookupBytes(table, hn::LoadU(d8, bins + i)), d8, g + i);
        else for (size_t i = 0; i < kTiles; i += hn::Lanes(d8)) hn::Store(hn::Or(hn::Load(d8, g + i), hn::TableLookupBytes(table, hn::LoadU(d8, bins + i))), d8, g + i);
      }
      if (tr.v == 0) {  // tile-constant tree: one gathered code per tile
        const int32_t* code = tm.code32.data() + m.tree_offsets[t];
        for (size_t i = 0; i < kTiles; i += hn::Lanes(d32))
          hn::StoreU(hn::Add(hn::LoadU(d32, flat.data() + i), hn::GatherIndex(d32, code, hn::PromoteTo(d32, hn::LoadU(dq, g + i)))), d32, flat.data() + i);
      } else {
        run_tree<NP>(tr, active.data(), active.size(), bsum.get());
      }
      offsets_done += m.offsets[t];
    }
    // widen this chunk's byte sums and tile totals into the running totals of the active packs
    const int shift = m.shifts[c];
    for (const uint16_t first : active) {
      for (size_t i = 0; i < kPack * kTileCells; i += l64) {
        const size_t cell = first * kTileCells + i;
        auto v = hn::PromoteTo(d64, hn::PromoteTo(d32, hn::LoadU(dq, bsum.get() + cell)));
        if constexpr (NP == 2) v = hn::Add(v, hn::ShiftLeft<4>(hn::PromoteTo(d64, hn::PromoteTo(d32, hn::LoadU(dq, bsum.get() + kCells + cell)))));
        v = hn::Add(v, hn::Set(d64, static_cast<int64_t>(flat[cell / kTileCells])));
        hn::StoreU(hn::Add(hn::LoadU(d64, total.get() + cell), hn::ShiftLeftSame(v, shift)), d64, total.get() + cell);
      }
    }
    if (opt.stages && next_stage < opt.stages->size() && (*opt.stages)[next_stage].trees == t1) {
      const double theta = (*opt.stages)[next_stage++].theta;
      // integer threshold: total * 2^e_min + offsets_done >= theta  <=>  total >= ceil((theta - offsets_done) / 2^e_min)
      const double lim = std::ceil(std::ldexp(theta - offsets_done, -m.e_min));
      const int64_t theta_int = lim <= -9.0e18 ? INT64_MIN : lim >= 9.0e18 ? INT64_MAX : static_cast<int64_t>(lim);
      const auto th = hn::Set(d64, theta_int);
      size_t kept = 0;
      for (const uint16_t first : active) {
        auto top = hn::Set(d64, INT64_MIN);
        for (size_t i = 0; i < kPack * kTileCells; i += l64) top = hn::Max(top, hn::LoadU(d64, total.get() + first * kTileCells + i));
        if (!hn::AllTrue(d64, hn::Lt(top, th))) active[kept++] = first;
      }
      active.resize(kept);
    }
  }
  if (opt.packs_finished) *opt.packs_finished = active.size();
  std::vector<float> raw(kCells);
  for (size_t r = 0; r < kGrid; ++r)
    for (size_t c = 0; c < kGrid; ++c)
      raw[r * kGrid + c] = raw_of(m, total[((r / kTile) * kTileGrid + c / kTile) * kTileCells + (r % kTile) * kTile + c % kTile]);
  sigmoid(raw, out, kCells);
}

void score_cells(const ImysModel& m, const TiledModel& tm, const float* xn, const std::vector<size_t>& offset,
                 uint8_t* fine, uint8_t* coarse, float* out, const ScoreOptions& opt) {
  if (m.leaf_bits == 4) score_cells_simd<1>(m, tm, xn, offset, fine, coarse, out, opt);
  else score_cells_simd<2>(m, tm, xn, offset, fine, coarse, out, opt);
}

// ---------------------------------------------------------------------------------------------
// Harness: gates against the scalar reference and the Python fixture, then timings.

template <class Fn>
double time_it(Fn fn, int iters) {
  std::vector<double> ms;
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

std::vector<Stage> parse_stages(const char* text, bool* ok) {
  std::vector<Stage> stages;
  *ok = true;
  for (const char* q = text; *q;) {
    char* end = nullptr;
    const uint32_t trees = static_cast<uint32_t>(std::strtoul(q, &end, 10));
    if (*end != ':') { *ok = false; break; }
    const float theta = std::strtof(end + 1, &end);
    stages.push_back({trees, theta});
    q = *end == ',' ? end + 1 : end;
  }
  return stages;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.fdt fixture.f32 [expected.f32] [iters] [stages trees:theta,...]\n", argv[0]);
    return 2;
  }
  const std::string container = read_all(argv[1]);
  if (container.empty()) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  const uint8_t* blob = nullptr;
  size_t blob_size = 0;
  if (!find_blob(container, &blob, &blob_size)) { std::fprintf(stderr, "no IMSY blob in %s\n", argv[1]); return 1; }
  ImysModel model;
  if (!load_imys(blob, blob_size, &model)) { std::fprintf(stderr, "bad IMSY blob\n"); return 1; }
  const int iters = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 20;
  std::vector<Stage> stages = model.stages;
  if (argc >= 6) {
    bool ok = false;
    stages = parse_stages(argv[5], &ok);
    if (!ok) { std::fprintf(stderr, "bad stage list: want trees:theta,trees:theta,...\n"); return 9; }
  }
  const TiledModel tiled = build_tiled(model);
  const std::vector<size_t> offset = native_offsets(model);
  const std::string fixture = read_all(argv[2]);
  const size_t native_values = offset.back();
  if (fixture.size() != native_values * sizeof(float)) {
    std::fprintf(stderr, "fixture has %zu floats, model wants %zu native values\n", fixture.size() / sizeof(float), native_values);
    return 3;
  }
  std::vector<float> xn(native_values);
  std::memcpy(xn.data(), fixture.data(), fixture.size());
  std::printf("simd target: %s, %zu tiles per index vector, %u-bit leaf codes\n", hwy::TargetName(HWY_TARGET), kPack, model.leaf_bits);
  std::printf("model: %u trees x depth %u (%u tile-constant, then fine), %u features, %u borders, %zu exit stages\n",
              model.n_trees, model.depth, model.coarse_trees, model.n_features, static_cast<uint32_t>(model.borders.size()), stages.size());
  {
    uint32_t hist[8] = {};
    for (uint32_t t = 0; t < model.n_trees; ++t) hist[tiled.nvary[t]]++;
    std::printf("used features: %u vary inside a 4x4 tile, %u are constant on it; trees by varying splits:", tiled.n_fine, tiled.n_coarse);
    for (uint32_t v = 0; v < 8; ++v) std::printf(" %u:%u", v, hist[v]);
    std::printf("\n");
  }
  // -- reference: dense bins per cell, scalar traversal -------------------------------------
  const std::vector<float> dense = expand_native(model, xn.data(), offset);
  std::vector<uint8_t> full_bins(kCells * model.n_features);
  bin_full(model, dense.data(), kCells, full_bins.data());
  std::vector<float> scalar_out(kCells), simd_out(kCells);
  score_cells_scalar(model, full_bins.data(), kCells, scalar_out.data());
  auto fine = hwy::AllocateAligned<uint8_t>(static_cast<size_t>(tiled.n_fine) * kCells + 64);
  auto coarse = hwy::AllocateAligned<uint8_t>(static_cast<size_t>(tiled.n_coarse) * kTiles + 64);
  ScoreOptions plain;
  score_cells(model, tiled, xn.data(), offset, fine.get(), coarse.get(), simd_out.data(), plain);
  // binner gate: every plane against the dense bins
  size_t bin_mismatch = 0;
  for (uint32_t f = 0; f < model.n_features; ++f) {
    if (tiled.fine_slot[f] < 0 && tiled.coarse_slot[f] < 0) continue;
    for (size_t r = 0; r < kGrid; ++r)
      for (size_t c = 0; c < kGrid; ++c) {
        const uint8_t expect = full_bins[static_cast<size_t>(f) * kCells + r * kGrid + c];
        const size_t tile = (r / kTile) * kTileGrid + c / kTile;
        if (tiled.fine_slot[f] >= 0)
          bin_mismatch += fine[static_cast<size_t>(tiled.fine_slot[f]) * kCells + tile * kTileCells + (r % kTile) * kTile + c % kTile] != expect;
        if (tiled.coarse_slot[f] >= 0) bin_mismatch += coarse[static_cast<size_t>(tiled.coarse_slot[f]) * kTiles + tile] != expect;
      }
  }
  std::printf("tile binner == reference binner: %zu mismatches %s\n", bin_mismatch, bin_mismatch ? "FAIL" : "PASS");
  float dv = 0.0f;
  for (size_t i = 0; i < kCells; ++i) dv = std::max(dv, std::fabs(simd_out[i] - scalar_out[i]));
  std::printf("simd vs scalar max|dprob| = %.3e  %s\n", dv, dv == 0.0f ? "BIT-IDENTICAL" : "DIVERGED");
  float ds = 0.0f, da = 0.0f;
  if (argc >= 4 && std::strcmp(argv[3], "-") != 0) {
    const std::string expected = read_all(argv[3]);
    if (expected.size() != kCells * sizeof(float)) { std::fprintf(stderr, "expected.f32 must hold %zu floats\n", kCells); return 4; }
    std::vector<float> want(kCells);
    std::memcpy(want.data(), expected.data(), expected.size());
    for (size_t i = 0; i < kCells; ++i) { ds = std::max(ds, std::fabs(scalar_out[i] - want[i])); da = std::max(da, std::fabs(simd_out[i] - want[i])); }
    std::printf("scalar max|got-expected| = %.3e\nsimd   max|got-expected| = %.3e\n", ds, da);
  }
  if (bin_mismatch != 0 || dv != 0.0f || ds > 1e-4f || da > 1e-4f) { std::printf("GATE FAILED\n"); return 5; }
  // -- early exit, as the model ships (or the override): the same scorer, stages applied ------------
  if (!stages.empty()) {
    ScoreOptions exit_opt;
    exit_opt.stages = &stages;
    exit_opt.lazy = true;
    size_t finished = 0;
    exit_opt.packs_finished = &finished;
    std::vector<float> exit_out(kCells);
    score_cells(model, tiled, xn.data(), offset, fine.get(), coarse.get(), exit_out.data(), exit_opt);
    size_t same = 0;
    float top_cut = 0.0f;
    for (size_t i = 0; i < kCells; ++i) {
      if (exit_out[i] == scalar_out[i]) ++same;
      else top_cut = std::max(top_cut, scalar_out[i]);
    }
    std::printf("early exit: %zu of %zu packs ran every tree; %zu of %zu cells bit-identical to the full scores; highest full probability among the rest %.4f\n",
                finished, kTiles / kPack, same, kCells, top_cut);
  }
  // -- timings --------------------------------------------------------------------------------
  const double probes = static_cast<double>(model.n_trees) * model.depth;
  const double ms_bin = time_it([&] { bin_tiles(model, tiled, xn.data(), offset, fine.get(), coarse.get()); }, iters);
  const double ms_scalar = time_it([&] { score_cells_scalar(model, full_bins.data(), kCells, scalar_out.data()); }, iters);
  const double ms_full = time_it([&] { score_cells(model, tiled, xn.data(), offset, fine.get(), coarse.get(), simd_out.data(), plain); }, iters);
  std::printf("tile binner      : %8.3f ms (p50, %d iters)\n", ms_bin, iters);
  std::printf("scalar traversal : %8.3f ms (%.3f ns/probe)\n", ms_scalar, 1e6 * ms_scalar / (probes * kCells));
  std::printf("simd traversal   : %8.3f ms (binning + every tree on every cell)\n", ms_full - ms_bin < 0 ? 0.0 : ms_full - ms_bin);
  std::printf("model total      : %8.3f ms (binning + every tree on every cell)\n", ms_full);
  if (!stages.empty()) {
    ScoreOptions exit_opt;
    exit_opt.stages = &stages;
    exit_opt.lazy = true;
    size_t finished = 0;
    exit_opt.packs_finished = &finished;
    std::vector<float> exit_out(kCells);
    const double ms_exit = time_it([&] { score_cells(model, tiled, xn.data(), offset, fine.get(), coarse.get(), exit_out.data(), exit_opt); }, iters);
    std::printf("model, as shipped: %8.3f ms (lazy binning, coarse tier per tile, early exit; %zu of %zu packs ran every tree)\n", ms_exit, finished, kTiles / kPack);
  }
  return 0;
}

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
//   * Pass               -- the shipped path.  Cells are scored in 4 x 4 tiles; the
//                           splits that are constant on a tile (most of them: the
//                           features form a pyramid) are evaluated once per tile and
//                           only choose which group of leaves the tile can reach,
//                           and the leaf values are fetched with byte shuffles
//                           rather than gathers.  A pass may run on several threads,
//                           with the output bit-identical at any thread count.  See
//                           "Tiled scorer" and "Threads" below.
// and one binner (bin_tiles) that bins each native value of each USED feature once.
// The vector path must be bit-identical to the scalar path.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"
// after highway.h: the vector exp for the logistic function
#include "hwy/contrib/math/math-inl.h"

namespace hn = hwy::HWY_NAMESPACE;

namespace {

constexpr size_t kGrid = 64;              // output grid side
constexpr size_t kCells = kGrid * kGrid;  // 4096 cells per image
constexpr uint8_t kMaxBin = 15;           // bins are table indices: at most 16 per feature
constexpr uint8_t kMaxShift = 12;         // level_shift of an image-wide (1x1) feature

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
  std::vector<uint8_t> codes;   // leaf codes, < 2**leaf_bits
  std::vector<float> offsets;   // per tree
  std::vector<uint8_t> shifts;  // per chunk: step = 2**(e_min + shift)
  std::vector<Stage> stages;    // early-exit stages calibrated at fit time
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
  auto rd32 = [&](size_t off) {
    uint32_t v;
    std::memcpy(&v, p + off, 4);
    return v;
  };
  if (size < 48 || std::memcmp(p, "IMSY", 4) != 0) return false;
  m->version = rd32(4);
  if (m->version != 4) {
    std::fprintf(stderr, "IMSY version %u; this scorer reads version 4\n", m->version);
    return false;
  }
  m->n_trees = rd32(8);
  m->n_features = rd32(12);
  m->n_leafs_total = rd32(16);
  m->depth = rd32(20);
  m->leaf_bits = rd32(24);
  m->leaf_chunk = rd32(28);
  m->coarse_trees = rd32(32);
  std::memcpy(&m->e_min, p + 36, 4);
  m->n_chunks = rd32(40);
  const uint32_t n_stages = rd32(44);
  size_t off = 48;
  auto take = [&](size_t bytes) {
    const uint8_t* q = p + off;
    off += bytes;
    return q;
  };
  if (m->depth == 0 || m->depth > 7 || (m->leaf_bits != 4 && m->leaf_bits != 8) || m->leaf_chunk == 0 ||
      m->leaf_chunk > 17)
    return false;
  m->tree_offsets.resize(m->n_trees + 1);
  std::memcpy(m->tree_offsets.data(), take(4 * (m->n_trees + 1)), 4 * (m->n_trees + 1));
  m->tree_base.resize(m->n_trees + 1);
  std::memcpy(m->tree_base.data(), take(4 * (m->n_trees + 1)), 4 * (m->n_trees + 1));
  const size_t n_splits = static_cast<size_t>(m->n_trees) * m->depth;
  m->split_feat.resize(n_splits);
  m->split_bin.resize(n_splits);
  for (size_t i = 0; i < n_splits; ++i) {
    const uint8_t* q = take(4);
    std::memcpy(&m->split_feat[i], q, 2);
    m->split_bin[i] = q[2];
  }
  m->codes.resize(m->n_leafs_total);
  std::memcpy(m->codes.data(), take(m->n_leafs_total), m->n_leafs_total);
  m->offsets.resize(m->n_trees);
  std::memcpy(m->offsets.data(), take(4 * m->n_trees), 4 * m->n_trees);
  m->shifts.resize(m->n_chunks);
  std::memcpy(m->shifts.data(), take(m->n_chunks), m->n_chunks);
  for (uint32_t i = 0; i < n_stages; ++i) {
    const uint8_t* q = take(8);
    Stage s;
    std::memcpy(&s.trees, q, 4);
    std::memcpy(&s.theta, q + 4, 4);
    m->stages.push_back(s);
  }
  m->n_borders.resize(m->n_features);
  std::memcpy(m->n_borders.data(), take(4 * m->n_features), 4 * m->n_features);
  m->border_offset.resize(m->n_features + 1);
  size_t total = 0;
  for (uint32_t f = 0; f < m->n_features; ++f) {
    m->border_offset[f] = static_cast<uint32_t>(total);
    total += m->n_borders[f];
    if (m->n_borders[f] > kMaxBin) return false;
  }
  m->border_offset[m->n_features] = static_cast<uint32_t>(total);
  m->borders.resize(total);
  std::memcpy(m->borders.data(), take(4 * total), 4 * total);
  m->level_shift.resize(m->n_features);
  std::memcpy(m->level_shift.data(), take(m->n_features), m->n_features);
  for (const uint8_t shift : m->level_shift)
    if (shift > kMaxShift || shift % 2) return false;  // sides 64, 32, ..., 1: even shifts up to 12
  if (off != size) {
    std::fprintf(stderr, "IMSY blob: %zu trailing bytes\n", size - off);
    return false;
  }
  // chunks: leaf_chunk trees each, restarting at coarse_trees
  m->chunk_of.resize(m->n_trees);
  m->chunk_starts.clear();
  const uint32_t starts[3] = {0, m->coarse_trees, m->n_trees};
  for (int s = 0; s < 2; ++s) {
    const uint32_t lo = starts[s], hi = starts[s + 1];
    for (uint32_t t = lo; t < hi; t += m->leaf_chunk) {
      for (uint32_t u = t; u < std::min(t + m->leaf_chunk, hi); ++u)
        m->chunk_of[u] = static_cast<uint32_t>(m->chunk_starts.size());
      m->chunk_starts.push_back(t);
    }
  }
  if (m->chunk_starts.size() != m->n_chunks) {
    std::fprintf(stderr, "IMSY blob: %zu chunks, header says %u\n", m->chunk_starts.size(), m->n_chunks);
    return false;
  }
  for (uint32_t t = 0; t < m->n_trees; ++t) m->offset_sum += m->offsets[t];
  return true;
}

inline uint8_t bin_one(const ImysModel& m, uint32_t f, float x) {
  const float* a = m.borders.data() + m.border_offset[f];
  const uint32_t n = m.n_borders[f];
  uint32_t lo = 0, hi = n;
  while (lo < hi) {
    const uint32_t mid = (lo + hi) >> 1;
    if (a[mid] < x)
      lo = mid + 1;
    else
      hi = mid;
  }
  return (uint8_t)lo;
}

void bin_full(const ImysModel& m, const float* X, size_t n_cells, uint8_t* B) {
  for (size_t c = 0; c < n_cells; ++c) {
    const float* row = X + c * m.n_features;
    for (uint32_t f = 0; f < m.n_features; ++f) B[(size_t)f * n_cells + c] = bin_one(m, f, row[f]);
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
    for (uint32_t j = 0; j < K; ++j) count = hn::Sub(count, hn::BitCast(di, hn::VecFromMask(df, hn::Gt(v, cut[j]))));
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
std::vector<float> expand_native(const ImysModel& m, const float* xn, const std::vector<size_t>& offset) {
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

// The logistic function of n raw scores (n a multiple of the vector width); Highway's Exp is
// within 1 ulp of std::exp.
void sigmoid(const float* raw, float* out, size_t n) {
  const hn::ScalableTag<float> df;
  const auto one = hn::Set(df, 1.0f);
  for (size_t i = 0; i < n; i += hn::Lanes(df))
    hn::StoreU(hn::Div(one, hn::Add(one, hn::Exp(df, hn::Neg(hn::LoadU(df, raw + i))))), df, out + i);
}

// Raw score of a cell from its integer total: the runtimes agree to the bit because the total
// is an integer and this is the one float expression that turns it into a score.
inline float raw_of(const ImysModel& m, int64_t total) {
  return static_cast<float>(m.offset_sum + std::ldexp(static_cast<double>(total), m.e_min));
}

// The probabilities of n cells (n a multiple of twice the int64 vector width) from their final
// totals: raw_of and the logistic function in vectors.  The scaling by 2^e_min is exact, so the
// fused multiply-add rounds once, like raw_of; bit-identical to raw_of followed by sigmoid.
void probabilities(const ImysModel& m, const int64_t* total, float* prob, size_t n) {
  const hn::ScalableTag<int64_t> d64;
  const hn::Rebind<double, decltype(d64)> dd;
  const hn::Rebind<float, decltype(d64)> dh;
  const hn::Twice<decltype(dh)> df;
  const size_t l64 = hn::Lanes(d64);
  const auto scale = hn::Set(dd, std::ldexp(1.0, m.e_min)), offset = hn::Set(dd, m.offset_sum);
  const auto one = hn::Set(df, 1.0f);
  auto raw = [&](size_t i) {
    return hn::DemoteTo(dh, hn::MulAdd(hn::ConvertTo(dd, hn::LoadU(d64, total + i)), scale, offset));
  };
  for (size_t i = 0; i < n; i += 2 * l64) {
    const auto x = hn::Combine(df, raw(i + l64), raw(i));
    hn::StoreU(hn::Div(one, hn::Add(one, hn::Exp(df, hn::Neg(x)))), df, prob + i);
  }
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
      for (uint32_t d = 0; d < m.depth; ++d)
        idx |= static_cast<uint32_t>(B[static_cast<size_t>(m.split_feat[s0 + d]) * n_cells + c] > m.split_bin[s0 + d])
               << d;
      total[c] += static_cast<int64_t>(code[idx]) << shift;
    }
  }
  std::vector<float> raw(n_cells);
  for (size_t c = 0; c < n_cells; ++c) raw[c] = raw_of(m, total[c]);
  sigmoid(raw.data(), out, n_cells);
}

// ---------------------------------------------------------------------------------------------
// Tiled scorer.

constexpr size_t kTile = 4;                       // tile side, in cells
constexpr size_t kTileCells = kTile * kTile;      // 16 cells: one 128-bit vector of byte lanes
constexpr size_t kTileGrid = kGrid / kTile;       // 16 x 16 tiles
constexpr size_t kTiles = kTileGrid * kTileGrid;  // 256
constexpr uint8_t kVaryShift = 2;                 // level_shift <= 2 (side 64, 32) varies in a tile

struct TiledModel {
  std::vector<int32_t> fine_slot, coarse_slot;  // per feature: its plane, or -1 if unused
  uint32_t n_fine = 0, n_coarse = 0;
  std::vector<uint8_t> nvary;                  // per tree: v (varying splits inside a tile)
  std::vector<uint32_t> plane;                 // per split, varying first: its plane slot
  hwy::AlignedFreeUniquePtr<uint8_t[]> table;  // per split: 16 bytes, bin -> its index bit
  std::vector<uint8_t> code;                   // leaf codes in permuted order
  std::vector<int32_t> code32;                 // the same, widened, for per-tile gathers
  hwy::AlignedFreeUniquePtr<uint8_t[]> nib;    // per leaf group: NP nibble planes of the codes
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
    // v >= 1: per leaf group, NP planes of its 2^v codes (what the fine-tree kernel shuffles);
    // v == 0: NP planes of all 2^depth codes (the tile-constant lookup shuffles those directly).
    uint8_t* planes = tm.nib.get() + static_cast<size_t>(m.tree_offsets[t]) * np;
    const uint32_t width = v ? 1u << v : n_leaves, groups = n_leaves / width;
    for (uint32_t g = 0; g < groups; ++g)
      for (uint32_t k = 0; k < np; ++k)
        for (uint32_t i = 0; i < width; ++i)
          planes[(g * np + k) * width + i] = static_cast<uint8_t>((dst[g * width + i] >> (4 * k)) & 15);
  }
  return tm;
}

#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
constexpr size_t kPack = 2;  // tiles per index vector (32 byte lanes)
#else
constexpr size_t kPack = 1;
#endif
using PackB = hn::FixedTag<uint8_t, 16 * kPack>;
constexpr size_t kPacks = kTiles / kPack;

// A unit is one full vector of tiles, one byte each: what the per-tile work (leaf groups, the
// tile-constant trees) is done in, and the grain the first phase of a pass splits between
// threads.  4 units of 64 tiles with 512-bit vectors, 8 of 32 with 256-bit, 16 of 16 with 128-bit.
constexpr size_t kUnit = HWY_HAVE_SCALABLE ? 16 : HWY_MAX_BYTES;
constexpr size_t kUnits = kTiles / kUnit;
using UnitB = hn::FixedTag<uint8_t, kUnit>;
static_assert(kUnit % (16 * kPack) == 0 && kTiles % kUnit == 0, "a unit is whole packs");

// A tile row of a coarse plane from one native row: byte c of the row is native value c >> shift.
constexpr uint8_t kExpand[4][16] = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                                    {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7},
                                    {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3},
                                    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}};

// Bins the used features f0, f0 + fstep, ... into the planes: fine planes tile-major (16 bytes
// per tile), coarse planes one byte per tile.  Each feature is binned in one run over its native
// values; threads take features round robin, so no two write the same plane.
void bin_tiles(const ImysModel& m, const TiledModel& tm, const float* xn, const std::vector<size_t>& offset,
               uint8_t* fine, uint8_t* coarse, size_t f0, size_t fstep, bool skip_side64) {
  uint8_t binned[kCells + 16];  // one feature's bins, native order (16 readable bytes past every row)
  uint8_t rows[2][kGrid];
  for (size_t f = f0; f < m.n_features; f += fstep) {
    const uint32_t k = m.n_borders[f];
    const float* cuts = m.borders.data() + m.border_offset[f];
    const float* xf = xn + offset[f];
    const uint32_t side = native_side(m, static_cast<uint32_t>(f));
    if (tm.fine_slot[f] >= 0 && !(skip_side64 && side == kGrid)) {
      // Four 64-cell rows hold 16 tiles; tile j is column group j of each row, so a block of
      // four tiles is a 4 x 4 transpose of 32-bit lanes.  A side-32 feature is first doubled
      // along the row, and each of its rows serves two cell rows.
      uint8_t* dst = fine + static_cast<size_t>(tm.fine_slot[f]) * kCells;
      const hn::FixedTag<uint8_t, 16> db;
      const hn::FixedTag<uint32_t, 4> du;
      const uint32_t per_tile_row = side / kTileGrid;  // native rows per tile row: 4 or 2
      bin_run(xf, side * side, cuts, k, binned);
      for (uint32_t tr = 0; tr < kTileGrid; ++tr) {
        const uint8_t* row[kTile];
        const uint8_t* src = binned + static_cast<size_t>(tr) * per_tile_row * side;
        if (side == kGrid) {
          for (uint32_t i = 0; i < kTile; ++i) row[i] = src + i * side;
        } else {
          for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t c = 0; c < kGrid / 2; c += 16) {
              const auto v = hn::LoadU(db, src + i * side + c);
              hn::StoreU(hn::InterleaveLower(db, v, v), db, rows[i] + 2 * c);
              hn::StoreU(hn::InterleaveUpper(db, v, v), db, rows[i] + 2 * c + 16);
            }
            row[2 * i] = row[2 * i + 1] = rows[i];
          }
        }
        uint8_t* tiles = dst + static_cast<size_t>(tr) * kTileGrid * kTileCells;
        for (uint32_t c = 0; c < kGrid; c += 16, tiles += 4 * kTileCells) {
          const auto r0 = hn::BitCast(du, hn::LoadU(db, row[0] + c)), r1 = hn::BitCast(du, hn::LoadU(db, row[1] + c));
          const auto r2 = hn::BitCast(du, hn::LoadU(db, row[2] + c)), r3 = hn::BitCast(du, hn::LoadU(db, row[3] + c));
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
      bin_run(xf, side * side, cuts, k, binned);
      const hn::FixedTag<uint8_t, 16> dr;
      const auto expand = hn::LoadU(dr, kExpand[shift]);  // tile column c reads native column c >> shift
      for (uint32_t r = 0; r < kTileGrid; ++r)
        hn::StoreU(hn::TableLookupBytes(hn::LoadU(dr, binned + static_cast<size_t>(r >> shift) * side), expand), dr,
                   dst + static_cast<size_t>(r) * kTileGrid);
    }
  }
}

// One tree's view of a tile pass: everything the inner loop touches, resolved to pointers.
struct TiledTree {
  uint32_t v = 0;
  const uint8_t* table = nullptr;  // 16 bytes per split, varying first
  const uint8_t* fine[8] = {};     // plane of each varying split (4096 bytes, tile-major)
  const uint8_t* nib = nullptr;    // this tree's nibble planes
  const uint8_t* group = nullptr;  // leaf group per tile
};

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
  for (uint32_t j = 0; j < V; ++j) {
    table[j] = hn::LoadDup128(db, tr.table + j * 16);
    plane[j] = tr.fine[j];
  }
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
      for (uint32_t b = 4; b < V; ++b)
        sel[b - 4] = hn::VecFromMask(db, hn::TestBit(idx, hn::Set(db, static_cast<uint8_t>(1u << b))));
      idx = hn::And(idx, low);
    }
    for (uint32_t k = 0; k < NP; ++k) {
      hn::VFromD<PackB> t[8];  // up to eight 16-entry sub-tables (V <= 7)
      for (size_t sub = 0; sub < subs; ++sub)
        t[sub] = hn::TableLookupBytes(pack_tables(base, k * width + sub * 16), idx);
      constexpr uint32_t levels = V > 4 ? V - 4 : 0;  // blend tree on the high index bits: 8 -> 4 -> 2 -> 1
      if constexpr (levels > 0) {
        for (uint32_t level = 0; level < levels; ++level) {
          const size_t pairs = subs >> (level + 1);
          for (size_t i = 0; i < pairs && 2 * i + 1 < 8; ++i)
            t[i] = hn::IfThenElse(hn::MaskFromVec(sel[level]), t[2 * i + 1], t[2 * i]);
        }
      }
      uint8_t* sum = bsum + k * kCells + cell0;
      hn::StoreU(hn::Add(hn::LoadU(db, sum), t[0]), db, sum);
    }
  }
}

// The leaf group of tree t per tile, for the tiles [lo, hi) (whole units): its tile-constant
// splits, each one byte shuffle of the split's coarse plane, ORed as the group's index bits.
void tile_groups(const TiledModel& tm, uint32_t t, uint32_t v, uint32_t depth, const uint8_t* coarse, uint8_t* group,
                 size_t lo, size_t hi) {
  const UnitB d8;
  const size_t s0 = static_cast<size_t>(t) * depth;
  if (v == depth) {
    for (size_t i = lo; i < hi; i += kUnit) hn::Store(hn::Zero(d8), d8, group + i);
    return;
  }
  for (uint32_t j = v; j < depth; ++j) {
    const auto table = hn::LoadDup128(d8, tm.table.get() + (s0 + j) * 16);
    const uint8_t* bins = coarse + static_cast<size_t>(tm.plane[s0 + j]) * kTiles;
    if (j == v)
      for (size_t i = lo; i < hi; i += kUnit)
        hn::Store(hn::TableLookupBytes(table, hn::LoadU(d8, bins + i)), d8, group + i);
    else
      for (size_t i = lo; i < hi; i += kUnit)
        hn::Store(hn::Or(hn::Load(d8, group + i), hn::TableLookupBytes(table, hn::LoadU(d8, bins + i))), d8, group + i);
  }
}

// The code of every tile of g (a leaf index < width per lane) from a byte table of width entries.
HWY_INLINE hn::VFromD<UnitB> lookup_codes(const uint8_t* table, uint32_t width, hn::VFromD<UnitB> g) {
  const UnitB d;
#if HWY_TARGET <= HWY_AVX3_DL
  // With VBMI a 128-entry byte table is one two-table permute (vpermi2b) per 64 tiles.
  const auto t0 = hn::LoadU(d, table);
  const auto t1 = width > 64 ? hn::LoadU(d, table + 64) : t0;
  return hn::TwoTablesLookupLanes(d, t0, t1, hn::IndicesFromVec(d, g));
#else
  // 16-entry sub-tables.  Instead of a blend tree, each sub-table's shuffle index gets bit 7 set
  // in the lanes that belong to another sub-table (a byte shuffle returns 0 there), and the
  // results are ORed: no blends, which cost two uops each on most x86 cores.
  if (width <= 16) return hn::TableLookupBytes(hn::LoadDup128(d, table), g);
  const auto low = hn::Set(d, 15), off = hn::Set(d, 0x80);
  const auto high = hn::ShiftRight<4>(g), idx = hn::And(g, low);
  auto v = hn::Zero(d);
  for (uint32_t sub = 0; sub < width / 16; ++sub) {
    const auto sel =
        hn::Or(idx, hn::AndNot(hn::VecFromMask(d, hn::Eq(high, hn::Set(d, static_cast<uint8_t>(sub)))), off));
    v = hn::Or(v, hn::TableLookupBytes(hn::LoadDup128(d, table + sub * 16), sel));
  }
  return v;
#endif
}

// The tile-constant trees of a chunk over the tiles [lo, hi), a unit at a time: the leaf group
// per tile of each tree IS its leaf index (one byte shuffle per split), which indexes the tree's
// code table; the codes are summed per tile in registers and stored once per unit, per plane.
template <uint32_t NP>
HWY_NOINLINE void coarse_trees_codes(const ImysModel& m, const TiledModel& tm, const uint8_t* coarse,
                                     const uint32_t* trees, size_t n, uint8_t* tsum, size_t lo, size_t hi) {
  const UnitB d8;
  const uint32_t depth = m.depth, width = 1u << depth;
  for (size_t i = lo; i < hi; i += kUnit) {
    hn::VFromD<UnitB> sum[NP];
    for (uint32_t k = 0; k < NP; ++k) sum[k] = hn::Zero(d8);
    for (size_t a = 0; a < n; ++a) {
      const size_t s0 = static_cast<size_t>(trees[a]) * depth;
      auto g = hn::TableLookupBytes(hn::LoadDup128(d8, tm.table.get() + s0 * 16),
                                    hn::LoadU(d8, coarse + static_cast<size_t>(tm.plane[s0]) * kTiles + i));
      for (uint32_t j = 1; j < depth; ++j)
        g = hn::Or(g, hn::TableLookupBytes(hn::LoadDup128(d8, tm.table.get() + (s0 + j) * 16),
                                           hn::LoadU(d8, coarse + static_cast<size_t>(tm.plane[s0 + j]) * kTiles + i)));
      const uint8_t* nib = tm.nib.get() + static_cast<size_t>(m.tree_offsets[trees[a]]) * NP;
      for (uint32_t k = 0; k < NP; ++k) sum[k] = hn::Add(sum[k], lookup_codes(nib + k * width, width, g));
    }
    for (uint32_t k = 0; k < NP; ++k) hn::StoreU(sum[k], d8, tsum + k * kTiles + i);
  }
}

template <uint32_t NP>
void run_tree(const TiledTree& tr, const uint16_t* active, size_t n_active, uint8_t* bsum) {
  switch (tr.v) {
    case 1:
      tree_codes_over_packs<1, NP>(tr, active, n_active, bsum);
      break;
    case 2:
      tree_codes_over_packs<2, NP>(tr, active, n_active, bsum);
      break;
    case 3:
      tree_codes_over_packs<3, NP>(tr, active, n_active, bsum);
      break;
    case 4:
      tree_codes_over_packs<4, NP>(tr, active, n_active, bsum);
      break;
    case 5:
      tree_codes_over_packs<5, NP>(tr, active, n_active, bsum);
      break;
    case 6:
      tree_codes_over_packs<6, NP>(tr, active, n_active, bsum);
      break;
    default:
      tree_codes_over_packs<7, NP>(tr, active, n_active, bsum);
      break;
  }
}

// Lazy binning: the side-64 features for the packs still alive, straight from the native fixture.
// Per feature, the packs' values are gathered tile-major into one buffer (4 rows x 4 floats per
// tile), binned in one run, and the bins land in the packs' slots of the plane.
constexpr size_t kPackCells = kPack * kTileCells;

void bin_side64_lazy(const ImysModel& m, const TiledModel& tm, const float* xn, const std::vector<size_t>& offset,
                     const uint16_t* active, size_t n_active, uint8_t* fine, float* vals, uint8_t* bins) {
  for (uint32_t f = 0; f < m.n_features; ++f) {
    if (tm.fine_slot[f] < 0 || native_side(m, f) != kGrid) continue;
    const float* xf = xn + offset[f];
    for (size_t i = 0; i < n_active; ++i) {
      const size_t tile = active[i], tr = tile / kTileGrid, tc = tile % kTileGrid;
      for (size_t p = 0; p < kPack; ++p)
        for (size_t r = 0; r < kTile; ++r)
          std::memcpy(vals + i * kPackCells + p * kTileCells + r * kTile,
                      xf + (kTile * tr + r) * kGrid + kTile * (tc + p), kTile * sizeof(float));
    }
    bin_run(vals, static_cast<uint32_t>(n_active * kPackCells), m.borders.data() + m.border_offset[f], m.n_borders[f],
            bins);
    uint8_t* plane = fine + static_cast<size_t>(tm.fine_slot[f]) * kCells;
    for (size_t i = 0; i < n_active; ++i)
      std::memcpy(plane + active[i] * kTileCells, bins + i * kPackCells, kPackCells);
  }
}

// ---------------------------------------------------------------------------------------------
// Threads.  A pass over an image has three phases, separated by barriers.  The threads bin the
// features round robin (whole planes); then each runs the coarse tier on its own units of tiles;
// then the packs still alive are dealt out again in balanced blocks (text clusters, so the fine
// tier's work does not follow the tile layout) and each thread runs the fine tier on its block.
// No thread reads what another writes within a phase, and every cell's total is the same integer
// sum whichever thread adds it up, so the output is bit-identical at any thread count.  Workers
// park on a condition variable between passes, as imfeat's do: a pass costs one wake-up, not
// thread creation.

class Pool {
 public:
  explicit Pool(size_t threads) : threads_(std::max<size_t>(1, threads)) {
    for (size_t i = 1; i < threads_; ++i) workers_.emplace_back([this, i] { worker(i); });
  }
  ~Pool() {
    {
      const std::lock_guard<std::mutex> lk(mu_);
      quit_ = true;
    }
    cv_go_.notify_all();
    for (std::thread& t : workers_) t.join();
  }
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  size_t size() const { return threads_; }

  // fn(i) on every thread i < size(), the caller taking i == 0; returns once all have finished.
  void run(std::function<void(size_t)> fn) {
    if (threads_ == 1) {
      fn(0);
      return;
    }
    {
      const std::lock_guard<std::mutex> lk(mu_);
      job_ = std::move(fn);
      pending_.store(threads_ - 1);
      ++epoch_;
    }
    cv_go_.notify_all();
    job_(0);
    // The workers finish within microseconds of the caller: spin a little before parking.
    for (int i = 0; i < kSpin && pending_.load() != 0; ++i) std::this_thread::yield();
    if (pending_.load() != 0) {
      std::unique_lock<std::mutex> lk(mu_);
      cv_done_.wait(lk, [&] { return pending_.load() == 0; });
    }
  }

  // Inside a job: wait until every thread of the job has arrived.  A spin, not a park: the threads
  // are all running, and the phase before the barrier is the same work on each of them.
  void barrier() {
    const uint64_t gen = generation_.load();
    if (arrived_.fetch_add(1) + 1 == threads_) {
      arrived_.store(0);
      generation_.store(gen + 1);
    } else {
      while (generation_.load() == gen) std::this_thread::yield();
    }
  }

 private:
  static constexpr int kSpin = 2000;

  void worker(size_t i) {
    uint64_t seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lk(mu_);
      cv_go_.wait(lk, [&] { return quit_ || epoch_ != seen; });
      if (quit_) return;
      seen = epoch_;
      lk.unlock();
      job_(i);
      if (pending_.fetch_sub(1) == 1) {
        const std::lock_guard<std::mutex> done(mu_);  // so the caller cannot miss the notify
        cv_done_.notify_one();
      }
    }
  }

  const size_t threads_;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_go_, cv_done_;
  std::function<void(size_t)> job_;
  uint64_t epoch_ = 0;
  bool quit_ = false;
  std::atomic<size_t> pending_{0};
  std::atomic<size_t> arrived_{0};
  std::atomic<uint64_t> generation_{0};
};

struct ScoreOptions {
  const std::vector<Stage>* stages = nullptr;  // nullptr: no early exit
  bool lazy = false;                           // bin side-64 features after the coarse tier
  bool prebinned = false;                      // planes already hold this image's bins (timing only)
  size_t* packs_finished = nullptr;
};

// What one thread writes during a pass and no other thread reads.
struct Scratch {
  hwy::AlignedFreeUniquePtr<int32_t[]> acc32;  // kCells: the current shift run's fine-tree sums, unshifted
  hwy::AlignedFreeUniquePtr<int32_t[]> tacc;   // kTiles: ... and its tile-constant-tree sums, per tile
  hwy::AlignedFreeUniquePtr<uint8_t[]> bsum;   // 2 * kCells: byte sums of a chunk's fine trees, per plane
  hwy::AlignedFreeUniquePtr<uint8_t[]> tsum;   // 2 * kTiles: ... of its tile-constant trees, per tile
  hwy::AlignedFreeUniquePtr<uint8_t[]> group;  // kTiles: the leaf group per tile of one tree
  hwy::AlignedFreeUniquePtr<float[]> vals;     // kCells: one side-64 feature's values for the active packs
  hwy::AlignedFreeUniquePtr<uint8_t[]> bins;   // kCells: ... and their bins
  std::vector<uint16_t> active;                // first tiles of the packs this thread is scoring

  Scratch()
      : acc32(hwy::AllocateAligned<int32_t>(kCells)),
        tacc(hwy::AllocateAligned<int32_t>(kTiles)),
        bsum(hwy::AllocateAligned<uint8_t>(2 * kCells)),
        tsum(hwy::AllocateAligned<uint8_t>(2 * kTiles + 64)),
        group(hwy::AllocateAligned<uint8_t>(kTiles + 64)),
        vals(hwy::AllocateAligned<float>(kCells)),
        bins(hwy::AllocateAligned<uint8_t>(kCells)) {
    active.reserve(kPacks);
  }
};

// What the threads of one pass share: the input, and outputs of which no two threads write the
// same cache line.
struct PassShared {
  const float* xn = nullptr;
  int64_t* total = nullptr;     // kCells, tile-major: the integer score of every cell
  float* prob = nullptr;        // kCells, tile-major: its probability, once final
  uint16_t* alive = nullptr;    // kPacks slots: after phase A, each thread's alive packs at its units' slots
  uint32_t* n_alive = nullptr;  // per thread
  uint32_t split_chunk = 0;     // the fine tier's first chunk: where phase B starts
  bool lazy = false;
  bool prebinned = false;
  const std::vector<Stage>* stages = nullptr;
  std::atomic<size_t> finished{0};  // packs that ran every tree
};

// One thread's share of a pass: chunk by chunk (the quantisation chunks are the passes), coarse
// trees once per tile, fine trees on the packs still active, integer totals, early exit at the
// blob's stages, each pack's probabilities written the moment its total is final.
template <uint32_t NP>
class Pass {
 public:
  Pass(const ImysModel& m, const TiledModel& tm, const std::vector<size_t>& offset, uint8_t* fine, uint8_t* coarse,
       PassShared& sh, Scratch& s)
      : m_(m), tm_(tm), offset_(offset), fine_(fine), coarse_(coarse), sh_(sh), s_(s), active_(s.active) {
    acc_shift_ = m.n_chunks ? m.shifts[0] : 0;
    std::fill(s.acc32.get(), s.acc32.get() + kCells, 0);
    std::fill(s.tacc.get(), s.tacc.get() + kTiles, 0);
  }

  void run(size_t t, size_t n_threads, Pool& pool) {
    // -- phase A: bin (features round robin), then the coarse tier on this thread's units ------
    if (!sh_.prebinned) {
      bin_tiles(m_, tm_, sh_.xn, offset_, fine_, coarse_, t, n_threads, /*skip_side64=*/sh_.lazy);
      pool.barrier();
    }
    const size_t lo = t * kUnits / n_threads * kUnit, hi = (t + 1) * kUnits / n_threads * kUnit;
    std::fill(sh_.total + lo * kTileCells, sh_.total + hi * kTileCells, int64_t{0});
    active_.clear();
    for (size_t tile = lo; tile < hi; tile += kPack) active_.push_back(static_cast<uint16_t>(tile));
    run_chunks(0, sh_.split_chunk);
    fold();
    std::copy(active_.begin(), active_.end(), sh_.alive + lo / kPack);
    sh_.n_alive[t] = static_cast<uint32_t>(active_.size());
    pool.barrier();
    // -- phase B: a balanced block of the packs alive after the coarse tier --------------------
    take_share(t, n_threads);
    if (sh_.lazy && !sh_.prebinned && !active_.empty())
      bin_side64_lazy(m_, tm_, sh_.xn, offset_, active_.data(), active_.size(), fine_, s_.vals.get(), s_.bins.get());
    run_chunks(sh_.split_chunk, m_.n_chunks);
    fold();
    for (const uint16_t first : active_) finish_pack(first);
    sh_.finished.fetch_add(active_.size());
  }

  // Block t of the alive packs, in tile order, cut only between cache lines of the tile-major
  // planes (4 tiles): the lazy binning then writes no line two threads share.
  void take_share(size_t t, size_t n_threads) {
    uint16_t all[kPacks];
    size_t n = 0;
    for (size_t i = 0; i < n_threads; ++i) {
      const uint16_t* from = sh_.alive + (i * kUnits / n_threads) * (kUnit / kPack);
      std::copy(from, from + sh_.n_alive[i], all + n);
      n += sh_.n_alive[i];
    }
    auto cut = [&](size_t b) {
      while (b > 0 && b < n && all[b] / 4 == all[b - 1] / 4) ++b;
      return b;
    };
    const size_t b0 = cut(n * t / n_threads), b1 = cut(n * (t + 1) / n_threads);
    active_.assign(all + b0, all + b1);
  }

  // The trees of chunks [c0, c1) on the active packs.  Their per-tile work covers the units the
  // active packs span (the list is in tile order).
  void run_chunks(uint32_t c0, uint32_t c1) {
    const uint32_t depth = m_.depth;
    uint8_t* const group = s_.group.get();
    uint8_t* const tsum = s_.tsum.get();
    uint8_t* const bsum = s_.bsum.get();
    for (uint32_t c = c0; c < c1 && !active_.empty(); ++c) {
      const uint32_t t0 = m_.chunk_start(c), t1 = m_.chunk_end(c);
      const size_t lo = (active_.front() / kUnit) * kUnit, hi = (active_.back() / kUnit + 1) * kUnit;
      if (m_.shifts[c] != acc_shift_) {
        fold();
        acc_shift_ = m_.shifts[c];
      }
      uint32_t flat[17];  // the chunk's tile-constant trees (leaf_chunk <= 17)
      size_t n_flat = 0;
      for (uint32_t t = t0; t < t1; ++t) n_flat += tm_.nvary[t] == 0;
      const bool fine_trees = n_flat < t1 - t0;
      if (fine_trees)
        for (const uint16_t first : active_)
          for (uint32_t k = 0; k < NP; ++k) std::memset(bsum + k * kCells + first * kTileCells, 0, kPackCells);
      if (n_flat == 0)  // otherwise coarse_trees_codes stores the tile sums of [lo, hi)
        for (uint32_t k = 0; k < NP; ++k) std::memset(tsum + k * kTiles + lo, 0, hi - lo);
      n_flat = 0;
      for (uint32_t t = t0; t < t1; ++t) {
        const size_t s0 = static_cast<size_t>(t) * depth;
        TiledTree tr;
        tr.v = tm_.nvary[t];
        if (tr.v == 0) {
          flat[n_flat++] = t;
          continue;
        }
        tr.table = tm_.table.get() + s0 * 16;
        for (uint32_t j = 0; j < tr.v; ++j) tr.fine[j] = fine_ + static_cast<size_t>(tm_.plane[s0 + j]) * kCells;
        tr.nib = tm_.nib.get() + static_cast<size_t>(m_.tree_offsets[t]) * NP;
        tr.group = group;
        tile_groups(tm_, t, tr.v, depth, coarse_, group, lo, hi);
        run_tree<NP>(tr, active_.data(), active_.size(), bsum);
      }
      if (n_flat) coarse_trees_codes<NP>(m_, tm_, coarse_, flat, n_flat, tsum, lo, hi);
      for (uint32_t t = t0; t < t1; ++t) offsets_done_ += m_.offsets[t];
      widen(fine_trees, lo, hi);
      if (sh_.stages && next_stage_ < sh_.stages->size() && (*sh_.stages)[next_stage_].trees == t1) {
        fold();
        exit_stage((*sh_.stages)[next_stage_++].theta);
      }
    }
  }

  // This chunk's tile sums (the tile-constant trees) into the per-tile accumulator and, when it
  // had fine trees, its byte sums per cell into the per-cell one.
  void widen(bool fine_trees, size_t lo, size_t hi) {
    const hn::ScalableTag<int32_t> d32;
    const hn::Rebind<uint8_t, decltype(d32)> dq;
    const size_t l32 = hn::Lanes(d32);
    const uint8_t* const bsum = s_.bsum.get();
    const uint8_t* const tsum = s_.tsum.get();
    int32_t* const acc32 = s_.acc32.get();
    int32_t* const tacc = s_.tacc.get();
    if (!fine_trees && active_.size() * kPack == hi - lo) {  // the coarse tier: every tile of the span
      for (size_t tile = lo; tile < hi; tile += l32) {
        auto v = hn::PromoteTo(d32, hn::LoadU(dq, tsum + tile));
        if constexpr (NP == 2)
          v = hn::Add(v, hn::ShiftLeft<4>(hn::PromoteTo(d32, hn::LoadU(dq, tsum + kTiles + tile))));
        hn::StoreU(hn::Add(hn::LoadU(d32, tacc + tile), v), d32, tacc + tile);
      }
      return;
    }
    for (const uint16_t first : active_) {
      for (size_t tile = first; tile < first + kPack; ++tile) {
        int32_t tile_code = tsum[tile];
        if constexpr (NP == 2) tile_code += static_cast<int32_t>(tsum[kTiles + tile]) << 4;
        tacc[tile] += tile_code;
        if (!fine_trees) continue;
        for (size_t i = 0; i < kTileCells; i += l32) {
          const size_t cell = tile * kTileCells + i;
          auto v = hn::PromoteTo(d32, hn::LoadU(dq, bsum + cell));
          if constexpr (NP == 2)
            v = hn::Add(v, hn::ShiftLeft<4>(hn::PromoteTo(d32, hn::LoadU(dq, bsum + kCells + cell))));
          hn::StoreU(hn::Add(hn::LoadU(d32, acc32 + cell), v), d32, acc32 + cell);
        }
      }
    }
  }

  // total += (acc32 + tacc) << shift on the active packs (the only cells with anything
  // accumulated); the accumulators are zeroed.
  void fold() {
    const hn::ScalableTag<int64_t> d64;
    const hn::Rebind<int32_t, decltype(d64)> dh32;
    const size_t l64 = hn::Lanes(d64);
    int32_t* const acc32 = s_.acc32.get();
    int32_t* const tacc = s_.tacc.get();
    for (const uint16_t first : active_) {
      for (size_t tile = first; tile < first + kPack; ++tile) {
        const auto per_tile = hn::Set(dh32, tacc[tile]);
        tacc[tile] = 0;
        for (size_t i = 0; i < kTileCells; i += l64) {
          const size_t cell = tile * kTileCells + i;
          const auto v = hn::PromoteTo(d64, hn::Add(hn::LoadU(dh32, acc32 + cell), per_tile));
          hn::StoreU(hn::Add(hn::LoadU(d64, sh_.total + cell), hn::ShiftLeftSame(v, acc_shift_)), d64,
                     sh_.total + cell);
          hn::StoreU(hn::Zero(dh32), dh32, acc32 + cell);
        }
      }
    }
  }

  // Drop every active pack whose cells are all below theta; their scores are final.
  void exit_stage(double theta) {
    // integer threshold: total * 2^e_min + offsets_done >= theta  <=>  total >= ceil((theta - offsets_done) / 2^e_min)
    const double lim = std::ceil(std::ldexp(theta - offsets_done_, -m_.e_min));
    const int64_t theta_int = lim <= -9.0e18 ? INT64_MIN : lim >= 9.0e18 ? INT64_MAX : static_cast<int64_t>(lim);
    const hn::ScalableTag<int64_t> d64;
    const size_t l64 = hn::Lanes(d64);
    const auto th = hn::Set(d64, theta_int);
    size_t kept = 0;
    for (const uint16_t first : active_) {
      auto top = hn::Set(d64, INT64_MIN);
      for (size_t i = 0; i < kPack * kTileCells; i += l64)
        top = hn::Max(top, hn::LoadU(d64, sh_.total + first * kTileCells + i));
      if (!hn::AllTrue(d64, hn::Lt(top, th)))
        active_[kept++] = first;
      else
        finish_pack(first);
    }
    active_.resize(kept);
  }

  void finish_pack(uint16_t first) {
    const size_t cell0 = static_cast<size_t>(first) * kTileCells;
    probabilities(m_, sh_.total + cell0, sh_.prob + cell0, kPackCells);
  }

  const ImysModel& m_;
  const TiledModel& tm_;
  const std::vector<size_t>& offset_;
  uint8_t* const fine_;
  uint8_t* const coarse_;
  PassShared& sh_;
  Scratch& s_;
  std::vector<uint16_t>& active_;
  int acc_shift_ = 0;
  double offsets_done_ = 0.0;
  size_t next_stage_ = 0;
};

constexpr size_t kMaxThreads = 16;  // beyond this a 64 x 64 grid has nothing left to share out

// A loaded model with everything a pass needs: the tiled tables, the bin planes, one scratch per
// thread and the pool.  A handle scores one image at a time.
struct FastdetHandle {
  ImysModel model;
  TiledModel tiled;
  std::vector<size_t> offset;
  hwy::AlignedFreeUniquePtr<uint8_t[]> fine, coarse;
  hwy::AlignedFreeUniquePtr<int64_t[]> total;
  hwy::AlignedFreeUniquePtr<float[]> prob;
  std::vector<uint16_t> alive;
  std::vector<uint32_t> n_alive;
  std::vector<Scratch> scratch;
  Pool pool;
  std::mutex busy;

  FastdetHandle(ImysModel m, size_t threads)
      : model(std::move(m)),
        tiled(build_tiled(model)),
        offset(native_offsets(model)),
        fine(hwy::AllocateAligned<uint8_t>(static_cast<size_t>(tiled.n_fine) * kCells + 64)),
        coarse(hwy::AllocateAligned<uint8_t>(static_cast<size_t>(tiled.n_coarse) * kTiles + 64)),
        total(hwy::AllocateAligned<int64_t>(kCells)),
        prob(hwy::AllocateAligned<float>(kCells)),
        alive(kPacks),
        n_alive(std::max<size_t>(1, std::min(threads, kMaxThreads))),
        scratch(n_alive.size()),
        pool(n_alive.size()) {}

  size_t threads() const { return pool.size(); }

  // Scores one native fixture into out (kCells probabilities, row-major grid).
  void score(const float* xn, float* out, const ScoreOptions& opt) {
    const std::lock_guard<std::mutex> lk(busy);
    PassShared sh;
    sh.xn = xn;
    sh.total = total.get();
    sh.prob = prob.get();
    sh.alive = alive.data();
    sh.n_alive = n_alive.data();
    sh.split_chunk = model.coarse_trees < model.n_trees ? model.chunk_of[model.coarse_trees] : model.n_chunks;
    sh.lazy = opt.lazy && model.coarse_trees > 0;
    sh.prebinned = opt.prebinned;
    sh.stages = opt.stages;
    const size_t n = pool.size();
    pool.run([&](size_t t) {
      if (model.leaf_bits == 4)
        Pass<1>(model, tiled, offset, fine.get(), coarse.get(), sh, scratch[t]).run(t, n, pool);
      else
        Pass<2>(model, tiled, offset, fine.get(), coarse.get(), sh, scratch[t]).run(t, n, pool);
    });
    if (opt.packs_finished) *opt.packs_finished = sh.finished.load();
    for (size_t r = 0; r < kGrid; ++r)  // tile-major probabilities to the row-major grid, a tile row at a time
      for (size_t c = 0; c < kGrid; c += kTile)
        std::memcpy(out + r * kGrid + c,
                    prob.get() + ((r / kTile) * kTileGrid + c / kTile) * kTileCells + (r % kTile) * kTile,
                    kTile * sizeof(float));
  }
};

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
    if (*end != ':') {
      *ok = false;
      break;
    }
    const float theta = std::strtof(end + 1, &end);
    stages.push_back({trees, theta});
    q = *end == ',' ? end + 1 : end;
  }
  return stages;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// C API wrapped by the nanobind module in bindings.cpp (the module and this file are one build).

// Parses an FDT1 container (or a bare IMSY blob) from memory and builds its scorer with `threads`
// threads (clamped to 1..16); nullptr on failure.
extern "C" void* fastdet_open(const uint8_t* bytes, size_t size, size_t threads) {
  const std::string raw(reinterpret_cast<const char*>(bytes), size);
  const uint8_t* blob = nullptr;
  size_t blob_size = 0;
  if (!find_blob(raw, &blob, &blob_size)) return nullptr;
  ImysModel model;
  if (!load_imys(blob, blob_size, &model)) return nullptr;
  return new FastdetHandle(std::move(model), threads);
}

extern "C" void fastdet_close(void* handle) {
  delete static_cast<FastdetHandle*>(handle);
}

// Number of floats the native fixture holds (Detector.native_matrix), and the grid's cell count.
extern "C" size_t fastdet_native_size(void* handle) {
  return static_cast<FastdetHandle*>(handle)->offset.back();
}
extern "C" size_t fastdet_cells(void*) {
  return kCells;
}
extern "C" size_t fastdet_threads(void* handle) {
  return static_cast<FastdetHandle*>(handle)->threads();
}
extern "C" const char* fastdet_target() {
  return hwy::TargetName(HWY_TARGET);
}

// Scores one image: `native` holds fastdet_native_size floats, `out` receives kCells probabilities
// in row-major grid order.  With use_exit the model's calibrated stages apply (lazy binning, coarse
// tier once per tile, early exit); without, every tree runs on every cell.  Returns 0 on success.
extern "C" int fastdet_score(void* handle, const float* native, float* out, int use_exit) {
  auto* h = static_cast<FastdetHandle*>(handle);
  ScoreOptions opt;
  if (use_exit && !h->model.stages.empty()) {
    opt.stages = &h->model.stages;
    opt.lazy = true;
  }
  h->score(native, out, opt);
  return 0;
}

#ifndef FASTDET_LIBRARY
int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s model.fdt fixture.f32 [expected.f32|-] [iters] [stages trees:theta,...|-] [threads]\n",
                 argv[0]);
    return 2;
  }
  const std::string container = read_all(argv[1]);
  if (container.empty()) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  const uint8_t* blob = nullptr;
  size_t blob_size = 0;
  if (!find_blob(container, &blob, &blob_size)) {
    std::fprintf(stderr, "no IMSY blob in %s\n", argv[1]);
    return 1;
  }
  ImysModel loaded;
  if (!load_imys(blob, blob_size, &loaded)) {
    std::fprintf(stderr, "bad IMSY blob\n");
    return 1;
  }
  const int iters = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 20;
  std::vector<Stage> stages = loaded.stages;
  if (argc >= 6 && std::strcmp(argv[5], "-") != 0) {
    bool ok = false;
    stages = parse_stages(argv[5], &ok);
    if (!ok) {
      std::fprintf(stderr, "bad stage list: want trees:theta,trees:theta,...\n");
      return 9;
    }
  }
  const size_t cores = std::max(1u, std::thread::hardware_concurrency());
  const size_t threads =
      argc >= 7 ? std::max<size_t>(1, std::strtoul(argv[6], nullptr, 10)) : std::min<size_t>(4, cores);
  FastdetHandle single(loaded, 1);
  FastdetHandle multi(std::move(loaded), threads);
  const ImysModel& model = single.model;
  const TiledModel& tiled = single.tiled;
  const std::vector<size_t>& offset = single.offset;
  const std::string fixture = read_all(argv[2]);
  const size_t native_values = offset.back();
  if (fixture.size() != native_values * sizeof(float)) {
    std::fprintf(stderr, "fixture has %zu floats, model wants %zu native values\n", fixture.size() / sizeof(float),
                 native_values);
    return 3;
  }
  std::vector<float> xn(native_values);
  std::memcpy(xn.data(), fixture.data(), fixture.size());
  std::printf("simd target: %s, %zu tiles per index vector, %u-bit leaf codes, %zu units of %zu tiles\n",
              hwy::TargetName(HWY_TARGET), kPack, model.leaf_bits, kUnits, kUnit);
  std::printf("model: %u trees x depth %u (%u tile-constant, then fine), %u features, %u borders, %zu exit stages\n",
              model.n_trees, model.depth, model.coarse_trees, model.n_features,
              static_cast<uint32_t>(model.borders.size()), stages.size());
  {
    uint32_t hist[8] = {};
    for (uint32_t t = 0; t < model.n_trees; ++t) hist[tiled.nvary[t]]++;
    std::printf("used features: %u vary inside a 4x4 tile, %u are constant on it; trees by varying splits:",
                tiled.n_fine, tiled.n_coarse);
    for (uint32_t v = 0; v < 8; ++v) std::printf(" %u:%u", v, hist[v]);
    std::printf("\n");
  }
  // -- reference: dense bins per cell, scalar traversal -------------------------------------
  const std::vector<float> dense = expand_native(model, xn.data(), offset);
  std::vector<uint8_t> full_bins(kCells * model.n_features);
  bin_full(model, dense.data(), kCells, full_bins.data());
  std::vector<float> scalar_out(kCells), simd_out(kCells), multi_out(kCells);
  score_cells_scalar(model, full_bins.data(), kCells, scalar_out.data());
  const ScoreOptions plain;
  single.score(xn.data(), simd_out.data(), plain);
  multi.score(xn.data(), multi_out.data(), plain);
  // binner gate: every plane against the dense bins, from both scorers
  size_t bin_mismatch = 0;
  for (const FastdetHandle* h : {&single, &multi}) {
    for (uint32_t f = 0; f < model.n_features; ++f) {
      if (tiled.fine_slot[f] < 0 && tiled.coarse_slot[f] < 0) continue;
      for (size_t r = 0; r < kGrid; ++r)
        for (size_t c = 0; c < kGrid; ++c) {
          const uint8_t expect = full_bins[static_cast<size_t>(f) * kCells + r * kGrid + c];
          const size_t tile = (r / kTile) * kTileGrid + c / kTile;
          if (tiled.fine_slot[f] >= 0)
            bin_mismatch += h->fine[static_cast<size_t>(tiled.fine_slot[f]) * kCells + tile * kTileCells +
                                    (r % kTile) * kTile + c % kTile] != expect;
          if (tiled.coarse_slot[f] >= 0)
            bin_mismatch += h->coarse[static_cast<size_t>(tiled.coarse_slot[f]) * kTiles + tile] != expect;
        }
    }
  }
  std::printf("tile binner == reference binner: %zu mismatches %s\n", bin_mismatch, bin_mismatch ? "FAIL" : "PASS");
  float dv = 0.0f, dt = 0.0f;
  for (size_t i = 0; i < kCells; ++i) {
    dv = std::max(dv, std::fabs(simd_out[i] - scalar_out[i]));
    dt = std::max(dt, std::fabs(multi_out[i] - simd_out[i]));
  }
  std::printf("simd vs scalar max|dprob| = %.3e  %s\n", dv, dv == 0.0f ? "BIT-IDENTICAL" : "DIVERGED");
  std::printf("%zu threads vs 1 max|dprob| = %.3e  %s\n", multi.threads(), dt,
              dt == 0.0f ? "BIT-IDENTICAL" : "DIVERGED");
  float ds = 0.0f, da = 0.0f;
  if (argc >= 4 && std::strcmp(argv[3], "-") != 0) {
    const std::string expected = read_all(argv[3]);
    if (expected.size() != kCells * sizeof(float)) {
      std::fprintf(stderr, "expected.f32 must hold %zu floats\n", kCells);
      return 4;
    }
    std::vector<float> want(kCells);
    std::memcpy(want.data(), expected.data(), expected.size());
    for (size_t i = 0; i < kCells; ++i) {
      ds = std::max(ds, std::fabs(scalar_out[i] - want[i]));
      da = std::max(da, std::fabs(simd_out[i] - want[i]));
    }
    std::printf("scalar max|got-expected| = %.3e\nsimd   max|got-expected| = %.3e\n", ds, da);
  }
  if (bin_mismatch != 0 || dv != 0.0f || dt != 0.0f || ds > 1e-4f || da > 1e-4f) {
    std::printf("GATE FAILED\n");
    return 5;
  }
  // -- early exit, as the model ships (or the override): the same scorer, stages applied ------------
  ScoreOptions exit_opt;
  size_t finished = 0;
  if (!stages.empty()) {
    exit_opt.stages = &stages;
    exit_opt.lazy = true;
    exit_opt.packs_finished = &finished;
    std::vector<float> exit_out(kCells), exit_multi(kCells);
    single.score(xn.data(), exit_out.data(), exit_opt);
    multi.score(xn.data(), exit_multi.data(), exit_opt);
    size_t same = 0;
    float top_cut = 0.0f, dm = 0.0f;
    for (size_t i = 0; i < kCells; ++i) {
      if (exit_out[i] == scalar_out[i])
        ++same;
      else
        top_cut = std::max(top_cut, scalar_out[i]);
      dm = std::max(dm, std::fabs(exit_multi[i] - exit_out[i]));
    }
    std::printf(
        "early exit: %zu of %zu packs ran every tree; %zu of %zu cells bit-identical to the full scores; highest full "
        "probability among the rest %.4f; %zu threads vs 1: %s\n",
        finished, kPacks, same, kCells, top_cut, multi.threads(), dm == 0.0f ? "BIT-IDENTICAL" : "DIVERGED");
    if (dm != 0.0f) {
      std::printf("GATE FAILED\n");
      return 5;
    }
  }
  // -- timings --------------------------------------------------------------------------------
  const double probes = static_cast<double>(model.n_trees) * model.depth;
  const double ms_bin = time_it(
      [&] { bin_tiles(model, tiled, xn.data(), offset, single.fine.get(), single.coarse.get(), 0, 1, false); }, iters);
  const double ms_scalar =
      time_it([&] { score_cells_scalar(model, full_bins.data(), kCells, scalar_out.data()); }, iters);
  const double ms_full = time_it([&] { single.score(xn.data(), simd_out.data(), plain); }, iters);
  std::printf("tile binner      : %8.3f ms (p50, %d iters)\n", ms_bin, iters);
  std::printf("scalar traversal : %8.3f ms (%.3f ns/probe)\n", ms_scalar, 1e6 * ms_scalar / (probes * kCells));
  ScoreOptions walk_only = plain;
  walk_only.prebinned = true;  // the planes hold this image's bins from the run above
  const double ms_walk = time_it([&] { single.score(xn.data(), simd_out.data(), walk_only); }, iters);
  std::printf("simd traversal   : %8.3f ms (every tree on every cell, planes already binned)\n", ms_walk);
  std::printf("model total      : %8.3f ms (binning + every tree on every cell)\n", ms_full);
  if (multi.threads() > 1) {
    const double ms_multi = time_it([&] { multi.score(xn.data(), multi_out.data(), plain); }, iters);
    std::printf("model total, %zu threads: %8.3f ms\n", multi.threads(), ms_multi);
  }
  if (!stages.empty()) {
    std::vector<float> exit_out(kCells);
    const double ms_exit = time_it([&] { single.score(xn.data(), exit_out.data(), exit_opt); }, iters);
    std::printf(
        "model, as shipped: %8.3f ms (lazy binning, coarse tier per tile, early exit; %zu of %zu packs ran every "
        "tree)\n",
        ms_exit, finished, kPacks);
    if (multi.threads() > 1) {
      const double ms_exit_multi = time_it([&] { multi.score(xn.data(), exit_out.data(), exit_opt); }, iters);
      std::printf("model, as shipped, %zu threads: %8.3f ms\n", multi.threads(), ms_exit_multi);
    }
  }
  return 0;
}
#endif  // FASTDET_LIBRARY

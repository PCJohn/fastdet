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

struct ImysModel {
  uint32_t n_trees = 0;
  uint32_t n_features = 0;
  uint32_t n_leafs_total = 0;
  uint32_t depth = 0;
  std::vector<uint32_t> tree_offsets;     // n_trees + 1 leaf-value starts
  std::vector<uint32_t> tree_base;        // n_trees + 1 split-record byte offsets
  std::vector<uint8_t> splits;            // n_trees * depth packed records
  std::vector<float> leaf_values;         // n_leafs_total
  std::vector<uint32_t> n_borders;        // per feature
  std::vector<uint32_t> border_offset;    // n_features + 1
  std::vector<float> borders;             // concatenated per-feature borders
  std::vector<uint8_t> level_shift;       // per feature (2 * log2(64 / level))
  std::vector<uint8_t> shuffle_tables;    // n_trees * depth * 16
};

// Extract the IMSY blob from either an FDT1 container or a bare IMSY blob.
bool locate_blob(const std::string& raw, const uint8_t** blob, size_t* blob_size) {
  if (raw.size() < 12) return false;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(raw.data());
  if (std::memcmp(p, "FDT1", 4) == 0) {
    const uint32_t header_len = rd_u32(p + 8);
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

bool load_imys(const uint8_t* p, size_t size, ImysModel* model) {
  if (size < 48 || std::memcmp(p, "IMSY", 4) != 0) return false;
  const uint32_t version = rd_u32(p + 4);
  if (version != 2 && version != 3) return false;
  model->n_trees = rd_u32(p + 8);
  model->n_features = rd_u32(p + 12);
  model->n_leafs_total = rd_u32(p + 16);
  model->depth = rd_u32(p + 20);

  const size_t n1 = 4 * (size_t)(model->n_trees + 1);
  size_t off = 24;
  if (size < off + 2 * n1) return false;
  model->tree_offsets.assign((const uint32_t*)(p + off), (const uint32_t*)(p + off) + model->n_trees + 1);
  off += n1;
  model->tree_base.assign((const uint32_t*)(p + off), (const uint32_t*)(p + off) + model->n_trees + 1);
  off += n1;
  if (model->tree_offsets[0] != 0 || model->tree_offsets[model->n_trees] != model->n_leafs_total) return false;
  if (model->tree_base[model->n_trees] != (size_t)model->n_trees * model->depth * 4) return false;

  const size_t split_bytes = (size_t)model->n_trees * model->depth * 4;
  if (size < off + split_bytes) return false;
  model->splits.assign(p + off, p + off + split_bytes);
  off += split_bytes;

  const size_t leaf_bytes = 4 * (size_t)model->n_leafs_total;
  if (size < off + leaf_bytes) return false;
  model->leaf_values.assign((const float*)(p + off), (const float*)(p + off) + model->n_leafs_total);
  off += leaf_bytes;

  const size_t nb_bytes = 4 * (size_t)model->n_features;
  if (size < off + nb_bytes) return false;
  model->n_borders.assign((const uint32_t*)(p + off), (const uint32_t*)(p + off) + model->n_features);
  off += nb_bytes;
  // A bin indexes a 16-entry table, so a feature has at most kMaxBin cuts (the exporter
  // enforces the same bound).
  const auto binnable = [](uint32_t cuts) { return cuts <= kMaxBin; };
  if (!std::all_of(model->n_borders.begin(), model->n_borders.end(), binnable)) return false;

  model->border_offset.assign(model->n_features + 1, 0);
  for (uint32_t f = 0; f < model->n_features; ++f)
    model->border_offset[f + 1] = model->border_offset[f] + model->n_borders[f];
  const size_t border_bytes = 4 * (size_t)model->border_offset[model->n_features];
  if (size < off + border_bytes) return false;
  model->borders.assign((const float*)(p + off),
                        (const float*)(p + off) + model->border_offset[model->n_features]);
  off += border_bytes;

  if (version == 2 || version == 3) {
    if (size < off + model->n_features) return false;
    model->level_shift.assign(p + off, p + off + model->n_features);
    const auto tileable = [](uint8_t shift) { return shift <= kMaxShift && !(shift & 1u); };
    if (!std::all_of(model->level_shift.begin(), model->level_shift.end(), tileable)) return false;
    off += model->n_features;
  }
  if (version == 3) {
    const size_t table_bytes = (size_t)model->n_trees * model->depth * 16;
    if (size < off + table_bytes) return false;
    model->shuffle_tables.assign(p + off, p + off + table_bytes);
    off += table_bytes;
  }
  return true;
}

// searchsorted-left: bin(x) = number of borders strictly below x.
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

// Reference binner: full column-major byte matrix. X is row-major (cell-major).
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
  for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp((double)-raw[i]));
}

// Reference scorer: one cell at a time, root split in the lowest leaf bit.
void score_cells_scalar(const ImysModel& m, const uint8_t* B, size_t n_cells, float* out) {
  std::vector<float> raw(n_cells, 0.0f);
  for (size_t c = 0; c < n_cells; ++c) {
    float acc = 0.0f;
    for (uint32_t t = 0; t < m.n_trees; ++t) {
      const uint8_t* sp = m.splits.data() + m.tree_base[t];
      const float* lf = m.leaf_values.data() + m.tree_offsets[t];
      uint32_t idx = 0;
      for (uint32_t d = 0; d < m.depth; ++d) {
        const uint16_t feat = *(const uint16_t*)(sp + d * 4);
        const uint8_t bin = sp[d * 4 + 2];
        idx |= (uint32_t)(B[(size_t)feat * n_cells + c] > bin) << d;
      }
      acc += lf[idx];
    }
    raw[c] = acc;
  }
  sigmoid(raw, out, n_cells);
}

// ---- Tiled scorer --------------------------------------------------------------------------
// The feature pyramid makes most splits coarse: a feature of side s is constant on blocks of
// (64 / s)^2 cells.  Cells are therefore scored in 4 x 4 tiles (one 16-lane byte vector), and a
// tree's splits are divided into those that vary inside a tile (side 64 or 32) and those that
// are constant on it (side <= 16, and image-wide features):
//   * the constant splits are evaluated once per tile, at 16 x 16 resolution, and only select
//     WHICH group of leaves the tile can reach;
//   * the v varying splits index inside that group.  With v <= 4 the group has at most 16
//     leaves, so the leaf value is fetched with four byte shuffles (one per byte of the
//     float) instead of a gather; v == 0 is a broadcast; v > 4 falls back to a gather.
// Leaves are permuted per tree so that the varying splits are the low index bits.  Every cell
// still adds its trees in order, so the result is bit-identical to score_cells_scalar.
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
  std::vector<uint8_t> nvary;                       // per tree: v
  std::vector<uint32_t> plane;                      // per split, varying first: its plane slot
  hwy::AlignedFreeUniquePtr<uint8_t[]> table;       // per split: 16 bytes, bin -> its index bit
  hwy::AlignedFreeUniquePtr<uint8_t[]> leaf_bytes;  // 1 <= v <= 4: per group, 4 byte planes
  std::vector<float> leaf_f32;                      // leaves in permuted order (v == 0, v > 4)
};

bool varies_in_tile(const ImysModel& m, uint32_t f) {
  return m.level_shift.empty() || m.level_shift[f] <= kVaryShift;
}

TiledModel build_tiled(const ImysModel& m) {
  TiledModel tm;
  const uint32_t depth = m.depth, leaves = 1u << depth;
  tm.fine_slot.assign(m.n_features, -1);
  tm.coarse_slot.assign(m.n_features, -1);
  tm.nvary.resize(m.n_trees);
  tm.plane.resize(static_cast<size_t>(m.n_trees) * depth);
  tm.table = hwy::AllocateAligned<uint8_t>(static_cast<size_t>(m.n_trees) * depth * 16);
  tm.leaf_bytes = hwy::AllocateAligned<uint8_t>(static_cast<size_t>(m.n_leafs_total) * 4 + 16);
  tm.leaf_f32.resize(m.n_leafs_total);
  std::memset(tm.leaf_bytes.get(), 0, static_cast<size_t>(m.n_leafs_total) * 4 + 16);
  std::vector<uint32_t> order(depth);
  for (uint32_t t = 0; t < m.n_trees; ++t) {
    const uint8_t* sp = m.splits.data() + m.tree_base[t];
    uint32_t v = 0, n = 0;  // varying splits take the low index bits, in their original order
    for (uint32_t d = 0; d < depth; ++d)
      if (varies_in_tile(m, rd_u16(sp + d * 4))) order[n++] = d;
    v = n;
    for (uint32_t d = 0; d < depth; ++d)
      if (!varies_in_tile(m, rd_u16(sp + d * 4))) order[n++] = d;
    tm.nvary[t] = static_cast<uint8_t>(v);
    for (uint32_t j = 0; j < depth; ++j) {
      const uint32_t f = rd_u16(sp + order[j] * 4);
      const uint8_t bin = sp[order[j] * 4 + 2];
      const bool varying = j < v;
      std::vector<int32_t>& slot = varying ? tm.fine_slot : tm.coarse_slot;
      if (slot[f] < 0) slot[f] = static_cast<int32_t>((varying ? tm.n_fine : tm.n_coarse)++);
      tm.plane[static_cast<size_t>(t) * depth + j] = static_cast<uint32_t>(slot[f]);
      const uint32_t bit = j < v ? j : j - v;  // within the low (varying) or high (constant) part
      uint8_t* tab = tm.table.get() + (static_cast<size_t>(t) * depth + j) * 16;
      for (uint32_t b = 0; b < 16; ++b) tab[b] = b > bin ? static_cast<uint8_t>(1u << bit) : 0;
    }
    const float* old_leaf = m.leaf_values.data() + m.tree_offsets[t];
    float* leaf = tm.leaf_f32.data() + m.tree_offsets[t];
    for (uint32_t idx = 0; idx < leaves; ++idx) {
      uint32_t old_idx = 0;
      for (uint32_t j = 0; j < depth; ++j) old_idx |= ((idx >> j) & 1u) << order[j];
      leaf[idx] = old_leaf[old_idx];
    }
    if (v >= 1 && v <= kBlendBits) {  // byte planes: group g holds leaves [g << v, (g + 1) << v)
      const uint32_t group = 1u << v;
      uint8_t* out = tm.leaf_bytes.get() + static_cast<size_t>(m.tree_offsets[t]) * 4;
      for (uint32_t g = 0; g < leaves / group; ++g)
        for (uint32_t k = 0; k < 4; ++k)
          for (uint32_t i = 0; i < group; ++i) {
            uint8_t bytes[4];
            std::memcpy(bytes, &leaf[g * group + i], 4);
            out[(static_cast<size_t>(g) * 4 + k) * group + i] = bytes[k];
          }
    }
  }
  return tm;
}

// Bins of the used features: fine planes (4096 bytes, tile-major: tile T = (r/4)*16 + c/4 at
// T*16 + (r%4)*4 + c%4) and tile-constant planes (256 bytes, one per tile).
void bin_tiles(const ImysModel& m, const TiledModel& tm, const float* xn,
               const std::vector<size_t>& offset, uint8_t* fine, uint8_t* coarse) {
  uint8_t rows[kTile][kGrid];
  uint8_t small[kTileGrid * kTileGrid];
  for (uint32_t f = 0; f < m.n_features; ++f) {
    if (tm.fine_slot[f] < 0 && tm.coarse_slot[f] < 0) continue;
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
  uint32_t v;                 // splits that vary inside a tile
  const uint8_t* table;       // v 16-byte tables, bin -> index bit
  const uint8_t* fine[8];     // the v fine planes those splits read
  const uint8_t* group;       // per tile: which leaf group it reaches (kTiles bytes)
  const uint8_t* leaf_bytes;  // 1 <= v <= 4: byte planes of the permuted leaves
  const float* leaf;          // permuted leaves
};

// Tiles are scored kPack at a time, one per 128-bit block of a vector: byte shuffles and
// interleaves work per block, so each tile uses its own leaf tables.  The running sums of a
// pack are kept as four float vectors; vector k holds cells 4k..4k+3 of every tile in it.
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
constexpr size_t kPack = 2;
#else
constexpr size_t kPack = 1;
#endif
using PackB = hn::FixedTag<uint8_t, 16 * kPack>;
using PackW = hn::FixedTag<uint16_t, 8 * kPack>;
using PackF = hn::FixedTag<float, 4 * kPack>;

// One 16-byte table per tile of the pack, from the same offset of each tile's own base.
HWY_INLINE hn::VFromD<PackB> pack_tables(const uint8_t* const* base, size_t offset) {
  const hn::FixedTag<uint8_t, 16> db;
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
  return hn::Combine(PackB(), hn::LoadU(db, base[1] + offset), hn::LoadU(db, base[0] + offset));
#else
  return hn::LoadU(db, base[0] + offset);
#endif
}

// Five varying splits: the fifth index bit chooses between two 16-entry tables per byte plane.
HWY_INLINE void pack_leaves5(const TiledTree& tr, size_t cell0, const uint8_t* g,
                             hn::VFromD<PackF>* acc) {
  const PackB db;
  const PackW dw;
  const PackF df;
  auto idx = hn::TableLookupBytes(hn::LoadDup128(db, tr.table), hn::LoadU(db, tr.fine[0] + cell0));
  for (uint32_t j = 1; j < kBlendBits; ++j)
    idx = hn::Or(idx, hn::TableLookupBytes(hn::LoadDup128(db, tr.table + j * 16),
                                           hn::LoadU(db, tr.fine[j] + cell0)));
  constexpr size_t width = static_cast<size_t>(1) << kBlendBits;
  const uint8_t* base[kPack];
  for (size_t k = 0; k < kPack; ++k)
    base[k] = tr.leaf_bytes + static_cast<size_t>(g[k]) * width * 4;
  const auto upper = hn::TestBit(idx, hn::Set(db, 16));
  const auto low = hn::And(idx, hn::Set(db, 15));
  auto plane = [&](size_t k) {
    return hn::IfThenElse(upper, hn::TableLookupBytes(pack_tables(base, k * width + 16), low),
                          hn::TableLookupBytes(pack_tables(base, k * width), low));
  };
  const auto b0 = plane(0), b1 = plane(1), b2 = plane(2), b3 = plane(3);
  const auto lo01 = hn::BitCast(dw, hn::InterleaveLower(db, b0, b1));
  const auto hi01 = hn::BitCast(dw, hn::InterleaveUpper(db, b0, b1));
  const auto lo23 = hn::BitCast(dw, hn::InterleaveLower(db, b2, b3));
  const auto hi23 = hn::BitCast(dw, hn::InterleaveUpper(db, b2, b3));
  acc[0] = hn::Add(acc[0], hn::BitCast(df, hn::InterleaveLower(dw, lo01, lo23)));
  acc[1] = hn::Add(acc[1], hn::BitCast(df, hn::InterleaveUpper(dw, lo01, lo23)));
  acc[2] = hn::Add(acc[2], hn::BitCast(df, hn::InterleaveLower(dw, hi01, hi23)));
  acc[3] = hn::Add(acc[3], hn::BitCast(df, hn::InterleaveUpper(dw, hi01, hi23)));
}

// Where cell `p` (0..15) of tile `tile` lives in the pack-ordered running sums.
inline size_t pack_slot(size_t tile, size_t p) {
  const size_t lanes = 4 * kPack;
  // kPack is 1 on 128-bit targets, where `tile % kPack` is a constant zero by design.
  // cppcheck-suppress moduloofone
  return (tile / kPack) * kTileCells * kPack + (p / 4) * lanes + (tile % kPack) * 4 + p % 4;
}

// Adds one tree with 1 <= V <= 4 varying splits to every pack still being scored.  The loop
// order is tree-outer on purpose: the running sums live in memory (16 KB, in L1) rather than
// in registers, but everything that depends only on the tree -- the dispatch on V, the split
// tables, the plane pointers -- is hoisted out of the pack loop, and a tree streams its few
// planes sequentially.  Measured 11% faster than holding the sums in registers across a group
// of trees.
template <uint32_t V>
HWY_NOINLINE void tree_over_packs(const TiledTree& tr, const uint16_t* active, size_t n_active,
                                  float* raw) {
  const PackB db;
  const PackW dw;
  const PackF df;
  const size_t lanes = hn::Lanes(df);
  hn::VFromD<PackB> table[V];
  for (uint32_t j = 0; j < V; ++j) table[j] = hn::LoadDup128(db, tr.table + j * 16);
  constexpr size_t width = static_cast<size_t>(1) << V;
  for (size_t a = 0; a < n_active; ++a) {
    const size_t tile = active[a], cell0 = tile * kTileCells;
    auto idx = hn::TableLookupBytes(table[0], hn::LoadU(db, tr.fine[0] + cell0));
    for (uint32_t j = 1; j < V; ++j)
      idx = hn::Or(idx, hn::TableLookupBytes(table[j], hn::LoadU(db, tr.fine[j] + cell0)));
    const uint8_t* base[kPack];
    for (size_t k = 0; k < kPack; ++k)
      base[k] = tr.leaf_bytes + static_cast<size_t>(tr.group[tile + k]) * width * 4;
    // Four byte shuffles fetch the four bytes of each of the pack's leaf values ...
    const auto b0 = hn::TableLookupBytes(pack_tables(base, 0), idx);
    const auto b1 = hn::TableLookupBytes(pack_tables(base, width), idx);
    const auto b2 = hn::TableLookupBytes(pack_tables(base, 2 * width), idx);
    const auto b3 = hn::TableLookupBytes(pack_tables(base, 3 * width), idx);
    // ... and two rounds of interleaves put them back together as floats.
    const auto lo01 = hn::BitCast(dw, hn::InterleaveLower(db, b0, b1));
    const auto hi01 = hn::BitCast(dw, hn::InterleaveUpper(db, b0, b1));
    const auto lo23 = hn::BitCast(dw, hn::InterleaveLower(db, b2, b3));
    const auto hi23 = hn::BitCast(dw, hn::InterleaveUpper(db, b2, b3));
    float* sum = raw + cell0;
    hn::Store(hn::Add(hn::Load(df, sum), hn::BitCast(df, hn::InterleaveLower(dw, lo01, lo23))), df,
              sum);
    hn::Store(
        hn::Add(hn::Load(df, sum + lanes), hn::BitCast(df, hn::InterleaveUpper(dw, lo01, lo23))),
        df, sum + lanes);
    hn::Store(hn::Add(hn::Load(df, sum + 2 * lanes),
                      hn::BitCast(df, hn::InterleaveLower(dw, hi01, hi23))),
              df, sum + 2 * lanes);
    hn::Store(hn::Add(hn::Load(df, sum + 3 * lanes),
                      hn::BitCast(df, hn::InterleaveUpper(dw, hi01, hi23))),
              df, sum + 3 * lanes);
  }
}

// Early exit (optional): after `trees` trees, a pack of tiles whose every running sum is below
// `theta` is dropped; its cells keep the partial sum.  A pack that survives to the end has run
// every tree in order, so its scores are bit-identical to the full evaluation.
struct Stage {
  uint32_t trees;
  float theta;  // on the raw (pre-sigmoid) sum
};

void score_cells_simd(const ImysModel& m, const TiledModel& tm, const uint8_t* fine,
                      const uint8_t* coarse, float* out, const std::vector<Stage>* stages = nullptr,
                      size_t* packs_finished = nullptr) {
  const hn::ScalableTag<uint8_t> d8;
  const PackB db;
  const PackF df;
  const hn::RebindToUnsigned<PackF> du;
  const hn::RebindToSigned<PackF> di;
  const hn::FixedTag<float, 4> d4;
  const hn::FixedTag<uint32_t, 4> du4;
  const hn::FixedTag<uint8_t, 16> d16;
  const hn::FixedTag<uint8_t, 4> dq;
  const uint32_t depth = m.depth;
  const size_t lanes = hn::Lanes(df);
  auto raw = hwy::AllocateAligned<float>(kCells);
  auto group = hwy::AllocateAligned<uint8_t>(kTiles);
  std::fill(raw.get(), raw.get() + kCells, 0.0f);
  std::vector<uint16_t> active(kTiles / kPack);  // first tile of every pack still being scored
  for (size_t a = 0; a < active.size(); ++a) active[a] = static_cast<uint16_t>(a * kPack);
  size_t next_stage = 0;
  for (uint32_t t = 0; t < m.n_trees; ++t) {
    const size_t s0 = static_cast<size_t>(t) * depth;
    TiledTree tr;
    tr.v = tm.nvary[t];
    tr.table = tm.table.get() + s0 * 16;
    for (uint32_t j = 0; j < tr.v; ++j)
      tr.fine[j] = fine + static_cast<size_t>(tm.plane[s0 + j]) * kCells;
    tr.leaf = tm.leaf_f32.data() + m.tree_offsets[t];
    tr.leaf_bytes = tm.leaf_bytes.get() + static_cast<size_t>(m.tree_offsets[t]) * 4;
    // Which leaf group each tile reaches, from the tile-constant splits.
    uint8_t* g = group.get();
    tr.group = g;
    std::memset(g, 0, kTiles);
    for (uint32_t j = tr.v; j < depth; ++j) {
      const auto table = hn::LoadDup128(d8, tm.table.get() + (s0 + j) * 16);
      const uint8_t* bins = coarse + static_cast<size_t>(tm.plane[s0 + j]) * kTiles;
      for (size_t c = 0; c < kTiles; c += hn::Lanes(d8))
        hn::Store(hn::Or(hn::Load(d8, g + c), hn::TableLookupBytes(table, hn::LoadU(d8, bins + c))),
                  d8, g + c);
    }
    switch (tr.v) {
      case 1:
        tree_over_packs<1>(tr, active.data(), active.size(), raw.get());
        break;
      case 2:
        tree_over_packs<2>(tr, active.data(), active.size(), raw.get());
        break;
      case 3:
        tree_over_packs<3>(tr, active.data(), active.size(), raw.get());
        break;
      case 4:
        tree_over_packs<4>(tr, active.data(), active.size(), raw.get());
        break;
      default:  // the rarer shapes, a pack at a time
        for (const size_t tile : active) {
          const size_t cell0 = tile * kTileCells;
          float* sum = raw.get() + cell0;
          hn::VFromD<PackF> acc[4] = {hn::Load(df, sum), hn::Load(df, sum + lanes),
                                      hn::Load(df, sum + 2 * lanes), hn::Load(df, sum + 3 * lanes)};
          const uint8_t* gp = tr.group + tile;
          if (tr.v == 0) {  // each tile lands in one leaf
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
            const auto leaf =
                hn::Combine(df, hn::Set(d4, tr.leaf[gp[1]]), hn::Set(d4, tr.leaf[gp[0]]));
#else
            const auto leaf = hn::Set(d4, tr.leaf[gp[0]]);
#endif
            for (auto& x : acc) x = hn::Add(x, leaf);
          } else if (tr.v == kBlendBits) {
            pack_leaves5(tr, cell0, gp, acc);
          } else {  // too many varying splits for a byte shuffle: full index, then gather
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
            auto idx = hn::Combine(db, hn::Set(d16, static_cast<uint8_t>(gp[1] << tr.v)),
                                   hn::Set(d16, static_cast<uint8_t>(gp[0] << tr.v)));
#else
            auto idx = hn::Set(d16, static_cast<uint8_t>(gp[0] << tr.v));
#endif
            for (uint32_t j = 0; j < tr.v; ++j)
              idx = hn::Or(idx, hn::TableLookupBytes(hn::LoadDup128(db, tr.table + j * 16),
                                                     hn::LoadU(db, tr.fine[j] + cell0)));
            uint8_t bytes[kTileCells * kPack];
            hn::StoreU(idx, db, bytes);
            for (size_t k = 0; k < 4;
                 ++k) {  // cells 4k..4k+3 of each tile, as the sums are laid out
#if HWY_MAX_BYTES >= 32 && !HWY_HAVE_SCALABLE
              const auto at =
                  hn::Combine(du, hn::PromoteTo(du4, hn::LoadU(dq, bytes + kTileCells + 4 * k)),
                              hn::PromoteTo(du4, hn::LoadU(dq, bytes + 4 * k)));
#else
              const auto at = hn::PromoteTo(du4, hn::LoadU(dq, bytes + 4 * k));
#endif
              acc[k] = hn::Add(acc[k], hn::GatherIndex(df, tr.leaf, hn::BitCast(di, at)));
            }
          }
          for (size_t k = 0; k < 4; ++k) hn::Store(acc[k], df, sum + k * lanes);
        }
    }
    if (stages && next_stage < stages->size() && (*stages)[next_stage].trees == t + 1) {
      const auto theta = hn::Set(df, (*stages)[next_stage++].theta);
      size_t kept = 0;
      for (const uint16_t tile : active) {
        const float* sum = raw.get() + static_cast<size_t>(tile) * kTileCells;
        const auto top =
            hn::Max(hn::Max(hn::Load(df, sum), hn::Load(df, sum + lanes)),
                    hn::Max(hn::Load(df, sum + 2 * lanes), hn::Load(df, sum + 3 * lanes)));
        if (!hn::AllTrue(df, hn::Lt(top, theta))) active[kept++] = tile;
      }
      active.resize(kept);
    }
  }
  if (packs_finished) *packs_finished = active.size();
  std::vector<float> ordered(kCells);  // pack order back to row-major
  for (size_t r = 0; r < kGrid; ++r)
    for (size_t c = 0; c < kGrid; ++c)
      ordered[r * kGrid + c] =
          raw[pack_slot((r / kTile) * kTileGrid + c / kTile, (r % kTile) * kTile + c % kTile)];
  sigmoid(ordered, out, kCells);
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float d = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
  return d;
}

template <typename Fn>
double time_it(Fn fn, int iters) {
  for (int r = 0; r < 5; ++r) fn();
  std::vector<double> t;
  t.reserve(iters);
  for (int r = 0; r < iters; ++r) {
    const Clock::time_point t0 = Clock::now();
    fn();
    t.push_back(ms_since(t0));
  }
  return median(t);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(
        stderr,
        "usage: fastdet_score model.fdt fixture.f32 [expected.f32] [iters=50] [trees:theta,...]\n");
    return 2;
  }
  const int iters = (argc >= 5) ? std::atoi(argv[4]) : 50;

  const std::string model_raw = read_all(argv[1]);
  const uint8_t* blob = nullptr;
  size_t blob_size = 0;
  if (!locate_blob(model_raw, &blob, &blob_size)) {
    std::fprintf(stderr, "not an FDT1/IMSY model: %s\n", argv[1]);
    return 3;
  }
  ImysModel model;
  if (!load_imys(blob, blob_size, &model)) {
    std::fprintf(stderr, "malformed IMSY blob\n");
    return 4;
  }
  if (model.depth > 8 || model.depth == 0) {
    std::fprintf(stderr, "depth %u unsupported: a leaf index must fit a byte lane\n", model.depth);
    return 5;
  }

  const std::string fixture_raw = read_all(argv[2]);
  const size_t n_features = model.n_features;
  const std::vector<size_t> offset = native_offsets(model);
  if (fixture_raw.size() != 4 * offset[n_features]) {
    std::fprintf(stderr, "fixture size mismatch: %zu bytes, expected %zu (native layout)\n",
                 fixture_raw.size(), 4 * offset[n_features]);
    return 6;
  }
  std::vector<float> xn(offset[n_features]);
  std::memcpy(xn.data(), fixture_raw.data(), xn.size() * 4);
  const std::vector<float> X = expand_native(model, xn.data(), offset);

  // Reference: every feature binned at every cell.  Shipped: used features, at tile layout.
  std::vector<uint8_t> full_bins(n_features * kCells);
  bin_full(model, X.data(), kCells, full_bins.data());
  const TiledModel tiled = build_tiled(model);
  auto fine = hwy::AllocateAligned<uint8_t>(static_cast<size_t>(tiled.n_fine) * kCells + 64);
  auto coarse = hwy::AllocateAligned<uint8_t>(static_cast<size_t>(tiled.n_coarse) * kTiles + 64);
  bin_tiles(model, tiled, xn.data(), offset, fine.get(), coarse.get());

  // Gate: the tile planes must hold the reference bins, cell for cell.
  size_t bin_mismatch = 0;
  for (uint32_t f = 0; f < n_features; ++f)
    for (size_t r = 0; r < kGrid; ++r)
      for (size_t c = 0; c < kGrid; ++c) {
        const uint8_t expect = full_bins[(size_t)f * kCells + r * kGrid + c];
        const size_t tile = (r / kTile) * kTileGrid + c / kTile;
        if (tiled.fine_slot[f] >= 0)
          bin_mismatch += fine[(size_t)tiled.fine_slot[f] * kCells + tile * kTileCells +
                               (r % kTile) * kTile + c % kTile] != expect;
        if (tiled.coarse_slot[f] >= 0)
          bin_mismatch += coarse[(size_t)tiled.coarse_slot[f] * kTiles + tile] != expect;
      }

  std::vector<float> scalar_out(kCells), simd_out(kCells);
  score_cells_scalar(model, full_bins.data(), kCells, scalar_out.data());
  score_cells_simd(model, tiled, fine.get(), coarse.get(), simd_out.data());

  std::printf("simd target: %s, %zu tiles per vector\n", hwy::TargetName(HWY_TARGET), kPack);
  std::printf("model: %u trees x depth %u, %u leaves, %u features, %u borders\n",
              model.n_trees, model.depth, model.n_leafs_total, model.n_features,
              model.border_offset[model.n_features]);
  uint32_t by_v[9] = {0};
  for (uint32_t t = 0; t < model.n_trees; ++t) by_v[std::min<uint32_t>(tiled.nvary[t], 8)]++;
  std::printf(
      "used features: %u vary inside a 4x4 tile, %u are constant on it; trees by varying splits:",
      tiled.n_fine, tiled.n_coarse);
  for (uint32_t v = 0; v <= model.depth; ++v) std::printf(" %u:%u", v, by_v[v]);
  std::printf("\ntile binner == reference binner: %zu mismatches %s\n", bin_mismatch,
              bin_mismatch == 0 ? "PASS" : "FAIL");
  const float dv = max_abs_diff(scalar_out, simd_out);
  std::printf("simd vs scalar max|dprob| = %.3e  %s\n", dv,
              dv == 0.0f ? "BIT-IDENTICAL" : "DIVERGED");

  float ds = 0.0f, da = 0.0f;
  if (argc >= 4) {
    const std::string expected_raw = read_all(argv[3]);
    if (expected_raw.size() != 4 * kCells) {
      std::fprintf(stderr, "expected size mismatch: %zu bytes\n", expected_raw.size());
      return 7;
    }
    std::vector<float> expected(kCells);
    std::memcpy(expected.data(), expected_raw.data(), expected.size() * 4);
    ds = max_abs_diff(scalar_out, expected);
    da = max_abs_diff(simd_out, expected);
    std::printf("scalar max|got-expected| = %.3e\n", ds);
    std::printf("simd   max|got-expected| = %.3e\n", da);
  }

  if (bin_mismatch != 0 || dv != 0.0f || ds > 1e-4f || da > 1e-4f) {
    std::fprintf(stderr, "GATE FAILED\n");
    return 8;
  }

  std::vector<Stage> stages;
  if (argc >= 6) {  // "trees:theta,trees:theta,..."
    for (const char* q = argv[5]; *q;) {
      char* end = nullptr;
      const uint32_t trees = static_cast<uint32_t>(std::strtoul(q, &end, 10));
      if (*end != ':') {
        std::fprintf(stderr, "bad stage list: want trees:theta,trees:theta,...\n");
        return 9;
      }
      const float theta = std::strtof(end + 1, &end);
      stages.push_back({trees, theta});
      q = *end == ',' ? end + 1 : end;
    }
  }

  const double probes = (double)model.n_trees * model.depth;
  const double ms_bin =
      time_it([&] { bin_tiles(model, tiled, xn.data(), offset, fine.get(), coarse.get()); }, iters);
  const double ms_scalar = time_it([&] { score_cells_scalar(model, full_bins.data(), kCells, scalar_out.data()); }, iters);
  const double ms_simd = time_it(
      [&] { score_cells_simd(model, tiled, fine.get(), coarse.get(), simd_out.data()); }, iters);
  std::printf("tile binner      : %8.3f ms (p50, %d iters)\n", ms_bin, iters);
  std::printf("scalar traversal : %8.3f ms (%.3f ns/probe)\n", ms_scalar,
              ms_scalar * 1e6 / (probes * kCells));
  std::printf("simd traversal   : %8.3f ms (%.3f ns/probe)\n", ms_simd,
              ms_simd * 1e6 / (probes * kCells));
  std::printf("model total      : %8.3f ms (binner + simd traversal)\n", ms_bin + ms_simd);
  if (!stages.empty()) {
    std::vector<float> early_out(kCells);
    size_t finished = 0;
    score_cells_simd(model, tiled, fine.get(), coarse.get(), early_out.data(), &stages, &finished);
    size_t same = 0;
    float top_dropped = 0.0f;  // the highest TRUE probability among cells whose score was cut short
    for (size_t i = 0; i < kCells; ++i) {
      if (early_out[i] == simd_out[i])
        ++same;
      else
        top_dropped = std::max(top_dropped, simd_out[i]);
    }
    const double ms_early = time_it(
        [&] {
          score_cells_simd(model, tiled, fine.get(), coarse.get(), early_out.data(), &stages,
                           &finished);
        },
        iters);
    std::printf(
        "early exit       : %8.3f ms traversal, %8.3f ms with binning; %zu of %zu packs ran every "
        "tree;\n"
        "                   %zu of %zu cells bit-identical to the full scores; highest full "
        "probability among "
        "the rest: %.4f\n",
        ms_early, ms_bin + ms_early, finished, kTiles / kPack, same, kCells, top_dropped);
  }
  return 0;
}

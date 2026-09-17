// fastdet_score.cpp -- C++17/AVX2 runtime for a fastdet single-file model.
//
// Reads the FDT1 container produced by fastdet (a JSON header followed by an
// IMSY symmetric-tree blob), bins a float fixture, scores every cell, and
// checks the two retained scoring paths against each other (and, when given,
// against a recorded expected output).
//
// Build (MSVC 2022 BuildTools):
//   vcvars64.bat && cl /O2 /EHsc /std:c++17 /arch:AVX2 fastdet_score.cpp
//
// Usage:
//   fastdet_score model.fdt fixture.f32 [expected.f32] [iters=50]
//
// There are exactly two scoring paths:
//   * score_cells_scalar       -- straightforward reference, one cell at a time.
//   * score_cells_avx2_nibble  -- shipped 4-bit nibble-packed shuffle traversal.
// and one fused binner (bin_coarse_nibble) that writes the packed nibble matrix
// directly.  The AVX2 path must be bit-identical to the scalar path.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <string>
#include <vector>

namespace {

constexpr size_t kGrid = 64;                // output grid side
constexpr size_t kCells = kGrid * kGrid;    // 4096 cells per image
constexpr size_t kLanes = 32;               // cells per AVX2 vector (2 x 16 bytes)
constexpr uint8_t kNibbleMask = 0x0F;       // 4-bit bin selector
constexpr uint8_t kMaxBin = 15;             // largest bin the nibble path can encode

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

// Fused packed binner. A level-L feature is constant on L x L blocks of the
// grid, so only the L^2 block representatives are scanned and each result is
// broadcast to its block. Cell j goes in the low nibble, cell j + half in the
// high nibble, so one 32-byte load serves 32 cells.
void bin_coarse_nibble(const ImysModel& m, const float* X, size_t n_cells, uint8_t* Bp) {
  const uint32_t nf = m.n_features;
  const size_t half = n_cells / 2;
  const __m256i one = _mm256_set1_epi32(1);
  std::memset(Bp, 0, (size_t)nf * half);
  for (uint32_t rr = 0; rr < kGrid; ++rr) {
    const float* xrow = X + (size_t)rr * kGrid * nf;
    const bool high_half = (rr >= kGrid / 2);
    for (uint32_t f = 0; f < nf; ++f) {
      const uint8_t shift = m.level_shift.empty() ? 0 : m.level_shift[f];
      const uint32_t t = shift >> 1;
      const uint32_t L = kGrid >> t;
      const uint32_t factor = kGrid / L;
      if (rr % factor) continue;
      const uint32_t K = m.n_borders[f];
      if (K == 0) continue;
      const float* cuts = m.borders.data() + m.border_offset[f];
      uint8_t* dst = Bp + (size_t)f * half;
      uint32_t cc = 0;
      for (; cc + 8 <= L; cc += 8) {
        const __m256i gi = _mm256_setr_epi32(
            (int)((cc + 0) * factor * nf + f), (int)((cc + 1) * factor * nf + f),
            (int)((cc + 2) * factor * nf + f), (int)((cc + 3) * factor * nf + f),
            (int)((cc + 4) * factor * nf + f), (int)((cc + 5) * factor * nf + f),
            (int)((cc + 6) * factor * nf + f), (int)((cc + 7) * factor * nf + f));
        const __m256 x = _mm256_i32gather_ps(xrow, gi, 4);
        __m256i acc = _mm256_setzero_si256();
        for (uint32_t j = 0; j < K; ++j) {
          const __m256 mask = _mm256_cmp_ps(x, _mm256_set1_ps(cuts[j]), _CMP_GT_OQ);
          acc = _mm256_add_epi32(acc, _mm256_and_si256(_mm256_castps_si256(mask), one));
        }
        const __m128i p16 = _mm_packus_epi32(_mm256_castsi256_si128(acc),
                                             _mm256_extracti128_si256(acc, 1));
        const __m128i p8 = _mm_packus_epi16(p16, p16);
        uint8_t bins[8];
        _mm_storel_epi64((__m128i*)bins, p8);
        for (uint32_t k = 0; k < 8; ++k) {
          const uint8_t b = high_half ? (uint8_t)(bins[k] << 4) : bins[k];
          const uint32_t c_fine = (cc + k) * factor;
          for (uint32_t dr = 0; dr < factor; ++dr) {
            uint8_t* d = dst + (size_t)((rr + dr) & (kGrid / 2 - 1)) * kGrid + c_fine;
            for (uint32_t dd = 0; dd < factor; ++dd) d[dd] |= b;
          }
        }
      }
      for (; cc < L; ++cc) {
        const uint32_t c_fine = cc * factor;
        const float x = xrow[(size_t)c_fine * nf + f];
        uint32_t count = 0;
        for (uint32_t j = 0; j < K; ++j) count += (x > cuts[j]) ? 1u : 0u;
        const uint8_t b = high_half ? (uint8_t)(count << 4) : (uint8_t)count;
        for (uint32_t dr = 0; dr < factor; ++dr) {
          uint8_t* d = dst + (size_t)((rr + dr) & (kGrid / 2 - 1)) * kGrid + c_fine;
          for (uint32_t dd = 0; dd < factor; ++dd) d[dd] |= b;
        }
      }
    }
  }
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

// Shipped scorer: 32 cells per vector via the 4-bit nibble-packed matrix and
// the per-split shuffle tables. The low nibble holds cell j, the high nibble
// cell j + half; vpshufb then maps each 4-bit bin to its leaf-index bit.
void score_cells_avx2_nibble(const ImysModel& m, const uint8_t* Bp, size_t n_cells, float* out) {
  const uint32_t depth = m.depth;
  const size_t half = n_cells / 2;
  std::vector<float> raw(n_cells, 0.0f);
  const __m256i nibble_mask = _mm256_set1_epi8((char)kNibbleMask);
  for (size_t c = 0; c + kLanes <= half; c += kLanes) {
    __m256 lo0 = _mm256_setzero_ps(), lo1 = _mm256_setzero_ps();
    __m256 lo2 = _mm256_setzero_ps(), lo3 = _mm256_setzero_ps();
    __m256 hi0 = _mm256_setzero_ps(), hi1 = _mm256_setzero_ps();
    __m256 hi2 = _mm256_setzero_ps(), hi3 = _mm256_setzero_ps();
    for (uint32_t t = 0; t < m.n_trees; ++t) {
      const uint8_t* sp = m.splits.data() + m.tree_base[t];
      const float* lf = m.leaf_values.data() + m.tree_offsets[t];
      const uint8_t* tp = m.shuffle_tables.data() + (size_t)t * depth * 16;
      __m256i il = _mm256_setzero_si256(), ih = _mm256_setzero_si256();
      for (uint32_t d = 0; d < depth; ++d) {
        const uint16_t feat = *(const uint16_t*)(sp + d * 4);
        const __m256i v = _mm256_loadu_si256((const __m256i*)(Bp + (size_t)feat * half + c));
        const __m256i tv = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tp + d * 16)));
        il = _mm256_or_si256(il, _mm256_shuffle_epi8(tv, _mm256_and_si256(v, nibble_mask)));
        ih = _mm256_or_si256(
            ih, _mm256_shuffle_epi8(tv, _mm256_and_si256(_mm256_srli_epi16(v, 4), nibble_mask)));
      }
      const __m128i ll = _mm256_castsi256_si128(il), lh = _mm256_extracti128_si256(il, 1);
      const __m128i hl = _mm256_castsi256_si128(ih), hh = _mm256_extracti128_si256(ih, 1);
      lo0 = _mm256_add_ps(lo0, _mm256_i32gather_ps(lf, _mm256_cvtepu8_epi32(ll), 4));
      lo1 = _mm256_add_ps(lo1, _mm256_i32gather_ps(lf, _mm256_cvtepu8_epi32(_mm_srli_si128(ll, 8)), 4));
      lo2 = _mm256_add_ps(lo2, _mm256_i32gather_ps(lf, _mm256_cvtepu8_epi32(lh), 4));
      lo3 = _mm256_add_ps(lo3, _mm256_i32gather_ps(lf, _mm256_cvtepu8_epi32(_mm_srli_si128(lh, 8)), 4));
      hi0 = _mm256_add_ps(hi0, _mm256_i32gather_ps(lf, _mm256_cvtepu8_epi32(hl), 4));
      hi1 = _mm256_add_ps(hi1, _mm256_i32gather_ps(lf, _mm256_cvtepu8_epi32(_mm_srli_si128(hl, 8)), 4));
      hi2 = _mm256_add_ps(hi2, _mm256_i32gather_ps(lf, _mm256_cvtepu8_epi32(hh), 4));
      hi3 = _mm256_add_ps(hi3, _mm256_i32gather_ps(lf, _mm256_cvtepu8_epi32(_mm_srli_si128(hh, 8)), 4));
    }
    _mm256_storeu_ps(raw.data() + c, lo0);
    _mm256_storeu_ps(raw.data() + c + 8, lo1);
    _mm256_storeu_ps(raw.data() + c + 16, lo2);
    _mm256_storeu_ps(raw.data() + c + 24, lo3);
    _mm256_storeu_ps(raw.data() + half + c, hi0);
    _mm256_storeu_ps(raw.data() + half + c + 8, hi1);
    _mm256_storeu_ps(raw.data() + half + c + 16, hi2);
    _mm256_storeu_ps(raw.data() + half + c + 24, hi3);
  }
  // Scalar tail over the remaining cells of both halves.
  for (size_t c = (half & ~(size_t)(kLanes - 1)); c < half; ++c) {
    for (size_t w = 0; w < 2; ++w) {
      float acc = 0.0f;
      for (uint32_t t = 0; t < m.n_trees; ++t) {
        const uint8_t* sp = m.splits.data() + m.tree_base[t];
        const float* lf = m.leaf_values.data() + m.tree_offsets[t];
        uint32_t idx = 0;
        for (uint32_t d = 0; d < m.depth; ++d) {
          const uint16_t feat = *(const uint16_t*)(sp + d * 4);
          const uint8_t bin = sp[d * 4 + 2];
          const uint8_t nib = (Bp[(size_t)feat * half + c] >> (4 * w)) & kNibbleMask;
          idx |= (uint32_t)(nib > bin) << d;
        }
        acc += lf[idx];
      }
      raw[c + w * half] = acc;
    }
  }
  sigmoid(raw, out, n_cells);
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
    std::fprintf(stderr, "usage: fastdet_score model.fdt fixture.f32 [expected.f32] [iters=50]\n");
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
  if (model.shuffle_tables.empty()) {
    std::fprintf(stderr, "model has no v3 shuffle tables; the AVX2 path needs them\n");
    return 5;
  }

  const std::string fixture_raw = read_all(argv[2]);
  const size_t n_features = model.n_features;
  if (fixture_raw.size() != 4 * kCells * n_features) {
    std::fprintf(stderr, "fixture size mismatch: %zu bytes, expected %zu\n",
                 fixture_raw.size(), 4 * kCells * n_features);
    return 6;
  }
  std::vector<float> X(kCells * n_features);
  std::memcpy(X.data(), fixture_raw.data(), X.size() * 4);

  std::vector<uint8_t> full_bins(n_features * kCells);
  std::vector<uint8_t> packed_bins(n_features * (kCells / 2));
  bin_full(model, X.data(), kCells, full_bins.data());
  bin_coarse_nibble(model, X.data(), kCells, packed_bins.data());

  // Gate: the fused packed binner must pack the reference byte matrix exactly.
  size_t bin_mismatch = 0;
  for (uint32_t f = 0; f < n_features; ++f) {
    for (size_t j = 0; j < kCells / 2; ++j) {
      const uint8_t expect = (uint8_t)((full_bins[(size_t)f * kCells + j] & kNibbleMask) |
                                       ((full_bins[(size_t)f * kCells + j + kCells / 2] & kNibbleMask) << 4));
      bin_mismatch += (packed_bins[(size_t)f * (kCells / 2) + j] != expect);
    }
  }

  std::vector<float> scalar_out(kCells), avx2_out(kCells);
  score_cells_scalar(model, full_bins.data(), kCells, scalar_out.data());
  score_cells_avx2_nibble(model, packed_bins.data(), kCells, avx2_out.data());

  std::printf("model: %u trees x depth %u, %u leaves, %u features, %u borders\n",
              model.n_trees, model.depth, model.n_leafs_total, model.n_features,
              model.border_offset[model.n_features]);
  std::printf("fused packed binner == pack(reference binner): %zu mismatches %s\n",
              bin_mismatch, bin_mismatch == 0 ? "PASS" : "FAIL");
  const float dv = max_abs_diff(scalar_out, avx2_out);
  std::printf("avx2 nibble vs scalar max|dprob| = %.3e  %s\n", dv,
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
    da = max_abs_diff(avx2_out, expected);
    std::printf("scalar     max|got-expected| = %.3e\n", ds);
    std::printf("avx2 nibble max|got-expected| = %.3e\n", da);
  }

  if (bin_mismatch != 0 || dv != 0.0f || ds > 1e-4f || da > 1e-4f) {
    std::fprintf(stderr, "GATE FAILED\n");
    return 8;
  }

  const double probes = (double)model.n_trees * model.depth;
  const double ms_bin = time_it([&] { bin_coarse_nibble(model, X.data(), kCells, packed_bins.data()); }, iters);
  const double ms_scalar = time_it([&] { score_cells_scalar(model, full_bins.data(), kCells, scalar_out.data()); }, iters);
  const double ms_avx2 = time_it([&] { score_cells_avx2_nibble(model, packed_bins.data(), kCells, avx2_out.data()); }, iters);
  std::printf("fused packed binner : %8.3f ms (p50, %d iters)\n", ms_bin, iters);
  std::printf("scalar traversal    : %8.3f ms (%.3f ns/probe)\n", ms_scalar, ms_scalar * 1e6 / (probes * kCells));
  std::printf("avx2 nibble traversal: %8.3f ms (%.3f ns/probe)\n", ms_avx2, ms_avx2 * 1e6 / (probes * kCells));
  std::printf("fused model total   : %8.3f ms (binner + avx2 traversal)\n", ms_bin + ms_avx2);
  return 0;
}

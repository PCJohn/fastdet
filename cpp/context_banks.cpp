// The context banks of one pyramid level, in one call, bit for bit what the OpenCV calls
// they replace produce (features.compute_context_values / compute_context2_values):
//
//   plane 2  ctx_surr3   pooled - box3(pooled)
//   plane 3  ctx_ring35  box5(pooled) - box3(pooled)
//   plane 4  ctx_range3  dilate3(pooled) - erode3(pooled)
//   plane 5  ctx_surr9   pooled - box9(pooled)
//   plane 6  ctx_range5  dilate5(pooled) - erode5(pooled)
//
// (planes 0 and 1, the cell coordinates, are constants the caller fills once). The maps are at
// most 64 x 64, so the cv2 route's cost was its 32 calls a frame and their allocations, not
// arithmetic; this is one call per level with no allocation, its loops written in Highway so
// that every compiler vectorises them (the static target the module is built for, as the
// scorer's).
//
// The box filter follows cv::boxFilter(CV_32F, normalize=true, BORDER_REFLECT) exactly: row
// sums in double -- a 3- or 5-tap window summed left to right, a 9-tap one as a running sum
// (s += S[i + 9] - S[i]) -- a running column sum in double, and one rounding to float of
// sum * (1.0 / (k * k)). The morphology is a plain max / min over the window with pixels
// outside the map ignored, as cv2.dilate / cv2.erode with their default border do (done
// separably: the window's max is the max of its rows' maxima). The subtractions are float32,
// as NumPy's. Every vector lane does the scalar operation in the scalar order, so the bytes
// are the same at any vector width, and the same as the scalar tails'.
#include <cstddef>
#include <cstdint>

#include "hwy/highway.h"

namespace fastdet_banks {

namespace hn = hwy::HWY_NAMESPACE;

constexpr int kMaxGrid = 64;
constexpr int kMaxPad = 4;  // the 9-tap window's reach
constexpr int kMaxW = kMaxGrid + 2 * kMaxPad;

// cv::borderInterpolate(p, len, BORDER_REFLECT): fedcba|abcdefgh|hgfedcb
static int reflect(int p, int len) {
  if (len == 1) return 0;
  do {
    if (p < 0)
      p = -p - 1;
    else
      p = 2 * len - p - 1;
  } while (static_cast<unsigned>(p) >= static_cast<unsigned>(len));
  return p;
}

struct Scratch {
  // the reflected index of every padded position; the padded rows of the source, as doubles;
  // the row sums of every padded row; the running column sums; three g x g maps for the
  // morphology (a slack vector past each, for the vector loops' last store)
  int at[kMaxW];
  double pads[kMaxW * kMaxW];
  double rows[kMaxW * kMaxGrid];
  double acc[kMaxGrid + 8];
  float a[kMaxGrid * kMaxGrid + 16], b[kMaxGrid * kMaxGrid + 16], t[kMaxGrid * kMaxGrid + 16];
};

using DD = hn::ScalableTag<double>;
using DF = hn::Rebind<float, DD>;    // as many floats as doubles: the float side of a conversion
using DFW = hn::ScalableTag<float>;  // a whole vector of floats, for the float-only passes

// dst = boxFilter(src, k x k, normalize, BORDER_REFLECT) on a g x g float32 map.
static void box_filter(const float* HWY_RESTRICT src, int g, int k, float* HWY_RESTRICT dst, Scratch& sc) {
  const DD dd;
  const DF df;
  const size_t N = hn::Lanes(dd);
  const int r = k / 2, padded = g + 2 * r, w = g + 2 * r;
  int* at = sc.at;
  for (int i = 0; i < w; ++i) at[i] = reflect(i - r, g);
  // the padded rows: the interior a straight widening of the source row, the 2r border
  // values gathered by their reflected index
  for (int pr = 0; pr < padded; ++pr) {
    const float* s = src + static_cast<size_t>(at[pr]) * g;
    double* pad = sc.pads + static_cast<size_t>(pr) * w;
    int i = 0;
    for (; i + static_cast<int>(N) <= g; i += static_cast<int>(N))
      hn::StoreU(hn::PromoteTo(dd, hn::LoadU(df, s + i)), dd, pad + r + i);
    for (; i < g; ++i) pad[r + i] = static_cast<double>(s[i]);
    for (int j = 0; j < r; ++j) {
      pad[j] = static_cast<double>(s[at[j]]);
      pad[r + g + j] = static_cast<double>(s[at[r + g + j]]);
    }
  }
  // the row sums: a 3- or 5-tap window added left to right, in every lane as in the scalar
  // tail; a 9-tap one as a running sum along the row, four rows abreast so that their
  // dependent additions overlap
  if (k == 3 || k == 5) {
    for (int pr = 0; pr < padded; ++pr) {
      const double* pad = sc.pads + static_cast<size_t>(pr) * w;
      double* d = sc.rows + static_cast<size_t>(pr) * g;
      int i = 0;
      for (; i + static_cast<int>(N) <= g; i += static_cast<int>(N)) {
        auto s = hn::Add(hn::LoadU(dd, pad + i), hn::LoadU(dd, pad + i + 1));
        s = hn::Add(s, hn::LoadU(dd, pad + i + 2));
        if (k == 5) {
          s = hn::Add(s, hn::LoadU(dd, pad + i + 3));
          s = hn::Add(s, hn::LoadU(dd, pad + i + 4));
        }
        hn::StoreU(s, dd, d + i);
      }
      for (; i < g; ++i) {
        double s = pad[i] + pad[i + 1] + pad[i + 2];
        if (k == 5) s = s + pad[i + 3] + pad[i + 4];
        d[i] = s;
      }
    }
  } else {
    int pr = 0;
    for (; pr + 4 <= padded; pr += 4) {
      const double* p0 = sc.pads + static_cast<size_t>(pr) * w;
      const double *p1 = p0 + w, *p2 = p1 + w, *p3 = p2 + w;
      double* d0 = sc.rows + static_cast<size_t>(pr) * g;
      double *d1 = d0 + g, *d2 = d1 + g, *d3 = d2 + g;
      double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
      for (int t = 0; t < k; ++t) {
        s0 += p0[t];
        s1 += p1[t];
        s2 += p2[t];
        s3 += p3[t];
      }
      d0[0] = s0, d1[0] = s1, d2[0] = s2, d3[0] = s3;
      for (int i = 0; i + 1 < g; ++i) {
        s0 += p0[i + k] - p0[i];
        s1 += p1[i + k] - p1[i];
        s2 += p2[i + k] - p2[i];
        s3 += p3[i + k] - p3[i];
        d0[i + 1] = s0, d1[i + 1] = s1, d2[i + 1] = s2, d3[i + 1] = s3;
      }
    }
    for (; pr < padded; ++pr) {
      const double* pad = sc.pads + static_cast<size_t>(pr) * w;
      double* d = sc.rows + static_cast<size_t>(pr) * g;
      double sum = 0.0;
      for (int t = 0; t < k; ++t) sum += pad[t];
      d[0] = sum;
      for (int i = 0; i + 1 < g; ++i) {
        sum += pad[i + k] - pad[i];
        d[i + 1] = sum;
      }
    }
  }
  // the column sums: a running sum down the rows, one rounding to float of sum * scale
  const double scale = 1.0 / static_cast<double>(k * k);
  const auto vscale = hn::Set(dd, scale);
  double* acc = sc.acc;
  for (int i = 0; i < g; ++i) acc[i] = 0.0;
  for (int t = 0; t < k - 1; ++t) {
    const double* d = sc.rows + static_cast<size_t>(t) * g;
    int i = 0;
    for (; i + static_cast<int>(N) <= g; i += static_cast<int>(N))
      hn::StoreU(hn::Add(hn::LoadU(dd, acc + i), hn::LoadU(dd, d + i)), dd, acc + i);
    for (; i < g; ++i) acc[i] += d[i];
  }
  for (int j = 0; j < g; ++j) {
    const double* in = sc.rows + static_cast<size_t>(j + k - 1) * g;
    const double* out = sc.rows + static_cast<size_t>(j) * g;
    float* d = dst + static_cast<size_t>(j) * g;
    int i = 0;
    for (; i + static_cast<int>(N) <= g; i += static_cast<int>(N)) {
      const auto s0 = hn::Add(hn::LoadU(dd, acc + i), hn::LoadU(dd, in + i));
      hn::StoreU(hn::DemoteTo(df, hn::Mul(s0, vscale)), df, d + i);
      hn::StoreU(hn::Sub(s0, hn::LoadU(dd, out + i)), dd, acc + i);
    }
    for (; i < g; ++i) {
      const double s0 = acc[i] + in[i];
      d[i] = static_cast<float>(s0 * scale);
      acc[i] = s0 - out[i];
    }
  }
}

// dst = dilate (max) or erode (min) over a k x k window, pixels outside the map ignored:
// the extreme along each row's window, then along each column's, through `tmp`.
template <bool MAX>
static void morph(const float* HWY_RESTRICT src, int g, int k, float* HWY_RESTRICT dst, float* HWY_RESTRICT tmp) {
  const DFW d;
  const int N = static_cast<int>(hn::Lanes(d));
  const int r = k / 2;
  auto pick = [](float v, float s) { return MAX ? (s > v ? s : v) : (s < v ? s : v); };
  auto vpick = [&](auto v, auto s) { return MAX ? hn::Max(v, s) : hn::Min(v, s); };
  for (int y = 0; y < g; ++y) {
    const float* s = src + static_cast<size_t>(y) * g;
    float* t = tmp + static_cast<size_t>(y) * g;
    for (int x = 0; x < r && x < g; ++x) {  // the left edge: a window clipped to the map
      float v = s[0];
      for (int xx = 1; xx <= x + r && xx < g; ++xx) v = pick(v, s[xx]);
      t[x] = v;
    }
    int x = r;  // the interior: a full window
    for (; x + N <= g - r; x += N) {
      auto v = hn::LoadU(d, s + x - r);
      for (int o = 1; o < k; ++o) v = vpick(v, hn::LoadU(d, s + x - r + o));
      hn::StoreU(v, d, t + x);
    }
    for (; x < g - r; ++x) {
      float v = s[x - r];
      for (int o = 1; o < k; ++o) v = pick(v, s[x - r + o]);
      t[x] = v;
    }
    for (x = g - r < r ? r : g - r; x < g; ++x) {  // the right edge
      const int x0 = x - r < 0 ? 0 : x - r;
      float v = s[x0];
      for (int xx = x0 + 1; xx < g; ++xx) v = pick(v, s[xx]);
      t[x] = v;
    }
  }
  for (int y = 0; y < g; ++y) {
    const int y0 = y - r < 0 ? 0 : y - r, y1 = y + r >= g ? g - 1 : y + r;
    float* dr = dst + static_cast<size_t>(y) * g;
    const float* first = tmp + static_cast<size_t>(y0) * g;
    int x = 0;
    if (y1 - y0 == 2) {  // a whole 3-row window: the rows' extremes in one expression
      const float *r1 = first + g, *r2 = r1 + g;
      for (; x + N <= g; x += N)
        hn::StoreU(vpick(vpick(hn::LoadU(d, first + x), hn::LoadU(d, r1 + x)), hn::LoadU(d, r2 + x)), d, dr + x);
    } else if (y1 - y0 == 4) {  // a whole 5-row window
      const float *r1 = first + g, *r2 = r1 + g, *r3 = r2 + g, *r4 = r3 + g;
      for (; x + N <= g; x += N) {
        auto v = vpick(vpick(hn::LoadU(d, first + x), hn::LoadU(d, r1 + x)), hn::LoadU(d, r2 + x));
        hn::StoreU(vpick(vpick(v, hn::LoadU(d, r3 + x)), hn::LoadU(d, r4 + x)), d, dr + x);
      }
    }
    for (; x + N <= g; x += N) {  // a window clipped by the map's edge
      auto v = hn::LoadU(d, first + x);
      for (int yy = y0 + 1; yy <= y1; ++yy) v = vpick(v, hn::LoadU(d, tmp + static_cast<size_t>(yy) * g + x));
      hn::StoreU(v, d, dr + x);
    }
    for (; x < g; ++x) {
      float v = first[x];
      for (int yy = y0 + 1; yy <= y1; ++yy) v = pick(v, tmp[static_cast<size_t>(yy) * g + x]);
      dr[x] = v;
    }
  }
}

// out[i] = a[i] - b[i] over n floats, in float32
static void subtract(const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, size_t n, float* HWY_RESTRICT out) {
  const DFW d;
  const size_t N = hn::Lanes(d);
  size_t i = 0;
  for (; i + N <= n; i += N) hn::StoreU(hn::Sub(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
  for (; i < n; ++i) out[i] = a[i] - b[i];
}

}  // namespace fastdet_banks

extern "C" {

// pooled: g x g float32 (g <= 64); out: 7 planes of g x g float32, planes 2..6 written (see
// above). ctx (planes 2..4) needs g >= 3 and ctx2 (5..6) g >= 5; below that the planes are
// zero, as the NumPy route leaves them. Returns 0, or 1 for a grid it cannot take.
int fastdet_context_banks(const float* pooled, int g, float* out) {
  using namespace fastdet_banks;
  if (g < 1 || g > kMaxGrid) return 1;
  const size_t n = static_cast<size_t>(g) * g;
  float* surr3 = out + 2 * n;
  float* ring35 = out + 3 * n;
  float* range3 = out + 4 * n;
  float* surr9 = out + 5 * n;
  float* range5 = out + 6 * n;
  if (g < 3) {
    for (size_t i = 0; i < 5 * n; ++i) surr3[i] = 0.0f;
    return 0;
  }
  thread_local Scratch sc;  // ~130 KB, sized for the largest grid: no allocation per call
  box_filter(pooled, g, 3, sc.a, sc);
  subtract(pooled, sc.a, n, surr3);
  box_filter(pooled, g, 5, sc.b, sc);
  subtract(sc.b, sc.a, n, ring35);
  morph<true>(pooled, g, 3, sc.a, sc.t);
  morph<false>(pooled, g, 3, sc.b, sc.t);
  subtract(sc.a, sc.b, n, range3);
  if (g < 5) {
    for (size_t i = 0; i < 2 * n; ++i) surr9[i] = 0.0f;
    return 0;
  }
  box_filter(pooled, g, 9, sc.a, sc);
  subtract(pooled, sc.a, n, surr9);
  morph<true>(pooled, g, 5, sc.a, sc.t);
  morph<false>(pooled, g, 5, sc.b, sc.t);
  subtract(sc.a, sc.b, n, range5);
  return 0;
}
}

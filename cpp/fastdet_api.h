// fastdet_api.h -- the C API of the scorer in fastdet_score.cpp, as the nanobind module
// (bindings.cpp) and any other host sees it.
#ifndef FASTDET_API_H_
#define FASTDET_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One feature's native values for the image being scored: value (r, c) of its side x side grid
// is base[r * row_stride + c * col_stride], the strides in floats.  A packed fixture has every
// feature contiguous and row-major (row_stride = side, col_stride = 1); a host's cell-major map,
// one record of every feature per cell, has col_stride = the record's width and row_stride =
// side times that; an image-wide feature (side 1) is base[0] whatever the strides.  The scorer
// reads any of these as it lies.
struct fastdet_source {
  const float* base;
  ptrdiff_t row_stride;
  ptrdiff_t col_stride;
};

// Parses an FDT1 container (or a bare IMSY blob) from memory and builds its scorer with
// `threads` threads (clamped to 1..16); NULL on failure.
void* fastdet_open(const uint8_t* bytes, size_t size, size_t threads);
void fastdet_close(void* handle);

size_t fastdet_native_size(void* handle);  // floats in a packed fixture (Detector.native_matrix)
size_t fastdet_cells(void* handle);        // cells in the output grid
size_t fastdet_threads(void* handle);      // threads a pass runs on
size_t fastdet_n_features(void* handle);   // features the model splits on (the blob's count)
// Side of feature f's native grid: 64, 32, ..., 1 (0 when f is out of range).
unsigned fastdet_feature_side(void* handle, size_t f);
const char* fastdet_target(void);  // the SIMD target this build runs

// Scores one image: `native` holds fastdet_native_size floats (every feature contiguous, in
// model order), `out` receives fastdet_cells probabilities in row-major grid order.  With
// use_exit the model's calibrated stages apply (lazy binning, coarse tier once per tile, early
// exit); without, every tree runs on every cell.  Returns 0 on success.
int fastdet_score(void* handle, const float* native, float* out, int use_exit);

// The same, each feature read from its own source (fastdet_n_features of them, in model order):
// the host's maps as they lie, no packed copy.  Bit for bit fastdet_score's output on the same
// values.
int fastdet_score_sources(void* handle, const struct fastdet_source* sources, float* out, int use_exit);

// A level's context banks the pass computes itself before binning the features that read
// them: fastdet_context_banks_from(base, row_stride, col_stride, side, out), run on the calling
// thread at the start of the pass while the other threads bin the rest.
struct fastdet_bank_job {
  const float* base;
  ptrdiff_t row_stride;
  ptrdiff_t col_stride;
  int side;
  float* out;
};

// fastdet_score_sources with `n_jobs` bank jobs done first on the calling thread; a feature
// whose after_jobs[f] is nonzero reads what a job writes and is binned by that thread once
// the jobs are done (after_jobs may be NULL when n_jobs is 0).  The other features are dealt
// out to every thread as they come free.  Same bytes as fastdet_score_sources on the sources
// with the banks already in place.
int fastdet_score_sources_banks(void* handle, const struct fastdet_source* sources, const struct fastdet_bank_job* jobs,
                                size_t n_jobs, const uint8_t* after_jobs, float* out, int use_exit);

// The context banks of one level (context_banks.cpp): planes 2..6 of out, 7 planes of g x g
// floats, from the g x g pooled luminance means -- contiguous, or where they lie in a
// cell-major map (value (r, c) at base[r * row_stride + c * col_stride], strides in floats).
// Returns 0, or 1 for a grid it cannot take (g < 1 or g > 64).
int fastdet_context_banks(const float* pooled, int g, float* out);
int fastdet_context_banks_from(const float* base, ptrdiff_t row_stride, ptrdiff_t col_stride, int g, float* out);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // FASTDET_API_H_

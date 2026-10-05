// nanobind module `fastdet._native_ext`: the C++ scorer as a Python extension, built by pip.
//
// Wraps the C API of fastdet_score.cpp (fastdet_api.h; compiled into this module with
// FASTDET_LIBRARY=1) so `pip install .` yields an in-process scorer with no separate build step.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fastdet_api.h"

namespace nb = nanobind;

namespace {

// Any float32 array on the CPU, whatever its rank and strides: a host's map as it lies.
using AnyFloat = nb::ndarray<const float, nb::device::cpu>;
// A contiguous vector of int32: the layout tables set_sources takes.
using Int32Vector = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

class Scorer {
 public:
  Scorer(const nb::bytes& blob, size_t threads)
      : handle_(fastdet_open(reinterpret_cast<const uint8_t*>(blob.c_str()), blob.size(), threads)) {
    if (handle_ == nullptr) throw std::invalid_argument("the C++ scorer rejected the model blob");
    const size_t n = fastdet_n_features(handle_);
    side_.resize(n);
    for (size_t f = 0; f < n; ++f) side_[f] = static_cast<int32_t>(fastdet_feature_side(handle_, f));
  }
  ~Scorer() {
    if (handle_ != nullptr) fastdet_close(handle_);  // joins the scorer's worker threads
  }
  Scorer(const Scorer&) = delete;
  Scorer& operator=(const Scorer&) = delete;

  size_t native_size() const { return fastdet_native_size(handle_); }
  size_t cells() const { return fastdet_cells(handle_); }
  size_t threads() const { return fastdet_threads(handle_); }
  size_t n_features() const { return side_.size(); }
  size_t n_slots() const { return slot_side_.size(); }

  // native: Detector.native_matrix as a contiguous float32 array; returns kCells probabilities.
  // The GIL is released while the scorer runs.
  nb::ndarray<nb::numpy, float, nb::ndim<1>> score(
      nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu> native, bool use_exit) {
    if (native.shape(0) != native_size())
      throw std::invalid_argument("native matrix has " + std::to_string(native.shape(0)) + " values, the model wants " +
                                  std::to_string(native_size()));
    return run([&](float* out) { return fastdet_score(handle_, native.data(), out, use_exit ? 1 : 0); });
  }

  // Where each feature's values are in a host's arrays, for score_maps: the arrays it will be
  // given are `slots`, each a 3-D (side, side, n) map whose last axis is features (any strides)
  // or a 1-D vector for the image-wide features (side 1).  Feature f is feature_index[f] of
  // array feature_slot[f]; its side must be the slot's.  Set once, checked here, so a call
  // only checks each array's shape against its slot.
  void set_sources(const Int32Vector& slot_side, const Int32Vector& feature_slot, const Int32Vector& feature_index) {
    const size_t n = n_features(), n_slots = slot_side.shape(0);
    if (feature_slot.shape(0) != n || feature_index.shape(0) != n)
      throw std::invalid_argument("feature_slot and feature_index must have one entry per model feature (" +
                                  std::to_string(n) + ")");
    std::vector<int32_t> sides(slot_side.data(), slot_side.data() + n_slots), max_index(n_slots, -1);
    for (size_t s = 0; s < n_slots; ++s)
      if (sides[s] < 1 || sides[s] > 64 || (sides[s] & (sides[s] - 1)))
        throw std::invalid_argument("slot " + std::to_string(s) + ": side must be 64, 32, ..., 1");
    for (size_t f = 0; f < n; ++f) {
      const int32_t s = feature_slot.data()[f], i = feature_index.data()[f];
      if (s < 0 || static_cast<size_t>(s) >= n_slots || i < 0)
        throw std::invalid_argument("feature " + std::to_string(f) + ": slot or index out of range");
      if (sides[static_cast<size_t>(s)] != side_[f])
        throw std::invalid_argument("feature " + std::to_string(f) + " is a side-" + std::to_string(side_[f]) +
                                    " feature, slot " + std::to_string(s) + " holds side " +
                                    std::to_string(sides[static_cast<size_t>(s)]));
      max_index[static_cast<size_t>(s)] = std::max(max_index[static_cast<size_t>(s)], i);
    }
    slot_side_ = std::move(sides);
    slot_max_index_ = std::move(max_index);
    feature_slot_.assign(feature_slot.data(), feature_slot.data() + n);
    feature_index_.assign(feature_index.data(), feature_index.data() + n);
  }

  // Probabilities from the host's arrays, one per slot of set_sources, read where they lie: the
  // same bytes as score() on the packed matrix of the same values, and no packed copy.
  nb::ndarray<nb::numpy, float, nb::ndim<1>> score_maps(const nb::sequence& arrays, bool use_exit) {
    const size_t n_slots = this->n_slots();
    if (n_slots == 0) throw std::runtime_error("set_sources() first: the scorer does not know the arrays' layout");
    if (nb::len(arrays) != n_slots)
      throw std::invalid_argument("expected " + std::to_string(n_slots) + " arrays, got " +
                                  std::to_string(nb::len(arrays)));
    struct View {
      const float* data;
      int64_t row, col, feature;  // strides, in floats
    };
    std::vector<View> views(n_slots);
    std::vector<AnyFloat> keep;  // the arrays stay alive (and their buffers) until the pass is over
    keep.reserve(n_slots);
    for (size_t s = 0; s < n_slots; ++s) {
      AnyFloat a;
      try {
        a = nb::cast<AnyFloat>(arrays[s], /*convert=*/false);
      } catch (const nb::cast_error&) {
        throw std::invalid_argument("array " + std::to_string(s) + " must be a float32 array on the CPU");
      }
      const int64_t side = slot_side_[s], last = slot_max_index_[s];
      if (a.ndim() == 3) {
        if (a.shape(0) != static_cast<size_t>(side) || a.shape(1) != static_cast<size_t>(side))
          throw std::invalid_argument("array " + std::to_string(s) + " must be (" + std::to_string(side) + ", " +
                                      std::to_string(side) + ", n), got (" + std::to_string(a.shape(0)) + ", " +
                                      std::to_string(a.shape(1)) + ", " + std::to_string(a.shape(2)) + ")");
        if (static_cast<int64_t>(a.shape(2)) <= last)
          throw std::invalid_argument("array " + std::to_string(s) + " has " + std::to_string(a.shape(2)) +
                                      " features, the model reads index " + std::to_string(last));
        views[s] = {a.data(), a.stride(0), a.stride(1), a.stride(2)};
      } else if (a.ndim() == 1) {
        if (side != 1)
          throw std::invalid_argument("array " + std::to_string(s) + " is 1-D but holds side-" + std::to_string(side) +
                                      " features");
        if (static_cast<int64_t>(a.shape(0)) <= last)
          throw std::invalid_argument("array " + std::to_string(s) + " has " + std::to_string(a.shape(0)) +
                                      " values, the model reads index " + std::to_string(last));
        views[s] = {a.data(), 0, 0, a.stride(0)};
      } else {
        throw std::invalid_argument("array " + std::to_string(s) + " must be 3-D (side, side, n) or 1-D");
      }
      keep.push_back(std::move(a));
    }
    // per call, not a member: the GIL is released during the pass, and a second caller must
    // not rewrite the sources under it (the scorer itself serialises passes)
    std::vector<fastdet_source> sources(feature_slot_.size());
    for (size_t f = 0; f < sources.size(); ++f) {
      const View& v = views[static_cast<size_t>(feature_slot_[f])];
      sources[f] = {v.data + static_cast<ptrdiff_t>(feature_index_[f]) * v.feature, static_cast<ptrdiff_t>(v.row),
                    static_cast<ptrdiff_t>(v.col)};
    }
    return run([&](float* out) { return fastdet_score_sources(handle_, sources.data(), out, use_exit ? 1 : 0); });
  }

 private:
  template <class Fn>
  nb::ndarray<nb::numpy, float, nb::ndim<1>> run(Fn fn) {
    const size_t n = cells();
    float* out = new float[n];
    nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    int status = 0;
    {
      const nb::gil_scoped_release nogil;
      status = fn(out);
    }
    if (status != 0) throw std::runtime_error("the C++ scorer failed with status " + std::to_string(status));
    return nb::ndarray<nb::numpy, float, nb::ndim<1>>(out, {n}, owner);
  }

  void* handle_;
  std::vector<int32_t> side_;  // per feature: 64, 32, ..., 1
  std::vector<int32_t> slot_side_, slot_max_index_, feature_slot_, feature_index_;
};

}  // namespace

NB_MODULE(_native_ext, m) {
  m.doc() = "fastdet's C++ scorer (Highway SIMD), built with the package";
  m.def("target", [] { return std::string(fastdet_target()); }, "the SIMD target this module was compiled for");
  nb::class_<Scorer>(m, "Scorer")
      .def(nb::init<nb::bytes, size_t>(), nb::arg("blob"), nb::arg("threads") = 1,
           "load an FDT1 container or bare IMSY blob; threads >= 1 score each image together")
      .def_prop_ro("native_size", &Scorer::native_size, "floats in Detector.native_matrix for this model")
      .def_prop_ro("cells", &Scorer::cells, "cells in the output grid (64 x 64)")
      .def_prop_ro("threads", &Scorer::threads, "threads a pass runs on (the request, capped by the SIMD width)")
      .def_prop_ro("n_features", &Scorer::n_features, "features the model splits on")
      .def_prop_ro("n_slots", &Scorer::n_slots, "arrays score_maps takes (0 until set_sources)")
      .def("score", &Scorer::score, nb::arg("native"), nb::arg("use_exit") = true,
           "probabilities (row-major grid) for one native-resolution feature matrix")
      .def("set_sources", &Scorer::set_sources, nb::arg("slot_side"), nb::arg("feature_slot"), nb::arg("feature_index"),
           "where score_maps finds each feature: array feature_slot[f], its last axis at feature_index[f]; "
           "slot_side gives each array's grid side (1 for a vector of image-wide features)")
      .def("score_maps", &Scorer::score_maps, nb::arg("arrays"), nb::arg("use_exit") = true,
           "probabilities (row-major grid) read straight from the arrays set_sources described: "
           "the same bytes as score() on the packed matrix, without making it");
  m.def(
      "context_banks",
      [](const nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>& pooled,
         const nb::ndarray<float, nb::ndim<3>, nb::c_contig, nb::device::cpu>& out) {
        const size_t g = pooled.shape(0);
        if (pooled.shape(1) != g) throw std::invalid_argument("pooled must be square");
        if (out.shape(0) != 7 || out.shape(1) != g || out.shape(2) != g)
          throw std::invalid_argument("out must be (7, g, g) for a g x g pooled map");
        if (fastdet_context_banks(pooled.data(), static_cast<int>(g), out.data()) != 0)
          throw std::invalid_argument("context_banks takes maps of 1 to 64 cells a side");
      },
      // no implicit conversion: a converted copy of `out` would take the results with it
      nb::arg("pooled").noconvert(), nb::arg("out").noconvert(),
      "the context banks of one level: planes 2..6 of out (surr3, ring35, range3, surr9, range5) "
      "from the g x g pooled luminance means, bit for bit the cv2 route's (see context_banks.cpp)");
}

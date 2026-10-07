// nanobind module `fastdet._native_ext`: the C++ scorer as a Python extension, built by pip.
//
// Wraps the C API of fastdet_score.cpp (fastdet_api.h; compiled into this module with
// FASTDET_LIBRARY=1) so `pip install .` yields an in-process scorer with no separate build step.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <climits>
#include <cmath>
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
// A (7, side, side) float32 buffer the scorer writes a level's context banks into.
using BankBuffer = nb::ndarray<float, nb::ndim<3>, nb::c_contig, nb::device::cpu>;

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
    const size_t n = n_features(), slots = slot_side.shape(0);
    if (feature_slot.shape(0) != n || feature_index.shape(0) != n)
      throw std::invalid_argument("feature_slot and feature_index must have one entry per model feature (" +
                                  std::to_string(n) + ")");
    std::vector<int32_t> sides(slot_side.data(), slot_side.data() + slots), max_index(slots, -1);
    for (size_t s = 0; s < slots; ++s)
      if (sides[s] < 1 || sides[s] > 64 || (sides[s] & (sides[s] - 1)))
        throw std::invalid_argument("slot " + std::to_string(s) + ": side must be 64, 32, ..., 1");
    for (size_t f = 0; f < n; ++f) {
      const int32_t s = feature_slot.data()[f], i = feature_index.data()[f];
      if (s < 0 || static_cast<size_t>(s) >= slots || i < 0)
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
    banks_.assign(slots, Bank{});
    after_jobs_.assign(n, 0);
  }

  // Declares slot `slot` as a level's context banks that the scorer computes itself, from the
  // luminance means at index `column` of the host array in `raw_slot` (a 3-D map of the same
  // side), into `buffer`, a (7, side, side) float32 array it keeps: planes 2..6 are written on
  // every call, planes 0 and 1 (the cell coordinates) are the caller's to fill once.  The
  // slot's features index the planes.  score_maps then takes None for the slot.
  void set_bank_slot(size_t slot, size_t raw_slot, size_t column, const BankBuffer& buffer) {
    const size_t slots = n_slots();
    if (slots == 0) throw std::runtime_error("set_sources() first");
    if (slot >= slots || raw_slot >= slots || raw_slot == slot)
      throw std::invalid_argument("set_bank_slot: slot and raw_slot must be distinct slots of set_sources");
    const int64_t side = slot_side_[slot];
    if (slot_side_[raw_slot] != side)
      throw std::invalid_argument("set_bank_slot: the raw slot must hold the same side as the bank slot");
    if (buffer.shape(0) != 7 || buffer.shape(1) != static_cast<size_t>(side) ||
        buffer.shape(2) != static_cast<size_t>(side))
      throw std::invalid_argument("set_bank_slot: buffer must be (7, side, side) for a side-" + std::to_string(side) +
                                  " slot");
    if (slot_max_index_[slot] >= 7)
      throw std::invalid_argument("set_bank_slot: the slot's features index beyond the 7 planes");
    if (column > static_cast<size_t>(INT32_MAX)) throw std::invalid_argument("set_bank_slot: column out of range");
    banks_[slot] = Bank{true, raw_slot, column, buffer};
    slot_max_index_[raw_slot] = std::max(slot_max_index_[raw_slot], static_cast<int32_t>(column));
    for (size_t f = 0; f < feature_slot_.size(); ++f)
      if (static_cast<size_t>(feature_slot_[f]) == slot) after_jobs_[f] = 1;
  }

  // Probabilities from the host's arrays, one per slot of set_sources, read where they lie: the
  // same bytes as score() on the packed matrix of the same values, and no packed copy.
  nb::ndarray<nb::numpy, float, nb::ndim<1>> score_maps(const nb::sequence& arrays, bool use_exit) {
    const size_t slots = n_slots();
    if (slots == 0) throw std::runtime_error("set_sources() first: the scorer does not know the arrays' layout");
    if (nb::len(arrays) != slots)
      throw std::invalid_argument("expected " + std::to_string(slots) + " arrays, got " +
                                  std::to_string(nb::len(arrays)));
    struct View {
      const float* data;
      int64_t row, col, feature;  // strides, in floats
    };
    std::vector<View> views(slots);
    std::vector<AnyFloat> keep;  // the arrays stay alive (and their buffers) until the pass is over
    keep.reserve(slots);
    for (size_t s = 0; s < slots; ++s) {
      const int64_t side = slot_side_[s], last = slot_max_index_[s];
      if (banks_[s].set) {  // the scorer's own buffer: its planes are the features
        if (!arrays[s].is_none())
          throw std::invalid_argument("array " + std::to_string(s) + " is computed by the scorer: pass None");
        views[s] = {banks_[s].buffer.data(), side, 1, side * side};
        continue;
      }
      AnyFloat a;
      try {
        a = nb::cast<AnyFloat>(arrays[s], /*convert=*/false);
      } catch (const nb::cast_error&) {
        throw std::invalid_argument("array " + std::to_string(s) + " must be a float32 array on the CPU");
      }
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
    // the banks the pass computes first, from the raw slots' arrays as given this call
    std::vector<fastdet_bank_job> jobs;
    for (size_t s = 0; s < slots; ++s) {
      const Bank& bank = banks_[s];
      if (!bank.set) continue;
      const View& raw = views[bank.raw_slot];
      jobs.push_back({raw.data + static_cast<ptrdiff_t>(bank.column) * raw.feature, static_cast<ptrdiff_t>(raw.row),
                      static_cast<ptrdiff_t>(raw.col), static_cast<int>(slot_side_[s]),
                      const_cast<float*>(views[s].data)});
    }
    const uint8_t* after = jobs.empty() ? nullptr : after_jobs_.data();
    return run([&](float* out) {
      return fastdet_score_sources_banks(handle_, sources.data(), jobs.data(), jobs.size(), after, out,
                                         use_exit ? 1 : 0);
    });
  }

 private:
  template <class Fn>
  nb::ndarray<nb::numpy, float, nb::ndim<1>> run(Fn fn) {
    const size_t n = cells();
    float* out = new float[n];
    const nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    int status = 0;
    {
      const nb::gil_scoped_release nogil;
      status = fn(out);
    }
    if (status != 0) throw std::runtime_error("the C++ scorer failed with status " + std::to_string(status));
    return nb::ndarray<nb::numpy, float, nb::ndim<1>>(out, {n}, owner);
  }

  // A slot the scorer computes: a level's context banks from a raw slot's column, into a
  // buffer the Python side owns (held here too, so it outlives the layout).
  struct Bank {
    bool set = false;
    size_t raw_slot = 0, column = 0;
    BankBuffer buffer;
  };

  void* handle_;
  std::vector<int32_t> side_;  // per feature: 64, 32, ..., 1
  std::vector<int32_t> slot_side_, slot_max_index_, feature_slot_, feature_index_;
  std::vector<Bank> banks_;          // per slot
  std::vector<uint8_t> after_jobs_;  // per feature: read from a bank slot
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
      .def("set_bank_slot", &Scorer::set_bank_slot, nb::arg("slot"), nb::arg("raw_slot"), nb::arg("column"),
           nb::arg("buffer").noconvert(),
           "a slot whose planes the scorer computes itself: the level's context banks, from the "
           "luminance means at `column` of the raw slot's map, into `buffer` (7, side, side); "
           "score_maps takes None for it")
      .def("score_maps", &Scorer::score_maps, nb::arg("arrays"), nb::arg("use_exit") = true,
           "probabilities (row-major grid) read straight from the arrays set_sources described "
           "(None for a slot set_bank_slot declared): the same bytes as score() on the packed "
           "matrix, without making it");
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
  m.def(
      "global_stats",
      [](const nb::ndarray<const float, nb::ndim<1>, nb::device::cpu>& block, double aspect, double log_area) {
        // the block's values with NaN and +-inf zeroed, then the two scalars rounded to float32
        // once: what compute_global_stats writes, in one call and no NumPy temporaries
        const size_t n = block.shape(0);
        float* out = new float[n + 2];
        const nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<float*>(p); });
        const float* src = block.data();
        const int64_t stride = block.stride(0);
        for (size_t i = 0; i < n; ++i) {
          const float v = src[static_cast<ptrdiff_t>(i) * stride];
          out[i] = std::isfinite(v) ? v : 0.0f;
        }
        out[n] = static_cast<float>(aspect);
        out[n + 1] = static_cast<float>(log_area);
        return nb::ndarray<nb::numpy, float, nb::ndim<1>>(out, {n + 2}, owner);
      },
      nb::arg("block"), nb::arg("aspect"), nb::arg("log_area"),
      "the broadcast global vector: the whole-image block (non-finite values zeroed) followed by "
      "the aspect ratio and the log area as float32");
  m.def(
      "context_banks_all",
      [](const nb::sequence& maps, size_t column, const nb::sequence& outs) {
        const size_t n = nb::len(maps);
        if (nb::len(outs) != n) throw std::invalid_argument("maps and outs must have one entry per level");
        struct Level {
          const float* base;
          int64_t row_stride, col_stride;
          int g;
          float* out;
        };
        std::vector<Level> levels;
        std::vector<nb::ndarray<const float, nb::ndim<3>, nb::device::cpu>> keep_maps;
        std::vector<nb::ndarray<float, nb::ndim<3>, nb::c_contig, nb::device::cpu>> keep_outs;
        levels.reserve(n), keep_maps.reserve(n), keep_outs.reserve(n);
        for (size_t i = 0; i < n; ++i) {
          nb::ndarray<const float, nb::ndim<3>, nb::device::cpu> map;
          nb::ndarray<float, nb::ndim<3>, nb::c_contig, nb::device::cpu> out;
          try {
            map = nb::cast<decltype(map)>(maps[i], /*convert=*/false);
            out = nb::cast<decltype(out)>(outs[i], /*convert=*/false);
          } catch (const nb::cast_error&) {
            throw std::invalid_argument("level " + std::to_string(i) +
                                        ": the map must be a 3-D float32 array and out a C-contiguous one");
          }
          const size_t g = map.shape(0);
          if (map.shape(1) != g || column >= map.shape(2))
            throw std::invalid_argument("level " + std::to_string(i) + ": the map must be (g, g, n) with column < n");
          if (out.shape(0) != 7 || out.shape(1) != g || out.shape(2) != g)
            throw std::invalid_argument("level " + std::to_string(i) + ": out must be (7, g, g) for a g x g map");
          levels.push_back({map.data() + static_cast<ptrdiff_t>(column) * map.stride(2), map.stride(0), map.stride(1),
                            static_cast<int>(g), out.data()});
          keep_maps.push_back(std::move(map));
          keep_outs.push_back(std::move(out));
        }
        int status = 0;
        {
          const nb::gil_scoped_release nogil;
          for (const Level& level : levels) {
            status = fastdet_context_banks_from(level.base, static_cast<ptrdiff_t>(level.row_stride),
                                                static_cast<ptrdiff_t>(level.col_stride), level.g, level.out);
            if (status != 0) break;
          }
        }
        if (status != 0) throw std::invalid_argument("context_banks_all takes maps of 1 to 64 cells a side");
      },
      nb::arg("maps"), nb::arg("column"), nb::arg("outs"),
      "the context banks of every level in one call: for each (g, g, n) map (any strides) the "
      "luminance means are its column `column`, read where they lie, and planes 2..6 of the "
      "matching (7, g, g) out are written, bit for bit context_banks on a copy of that column");
}

// nanobind module `fastdet._native_ext`: the C++ scorer as a Python extension, built by pip.
//
// Wraps the C API at the end of fastdet_score.cpp (compiled into this module with
// FASTDET_LIBRARY=1) so `pip install .` yields an in-process scorer with no separate build
// step; fastdet.native prefers it and falls back to the ctypes-loaded shared library.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

extern "C" {
void* fastdet_open(const uint8_t* bytes, size_t size);
void fastdet_close(void* handle);
size_t fastdet_native_size(void* handle);
size_t fastdet_cells(void* handle);
const char* fastdet_target();
int fastdet_score(void* handle, const float* native, float* out, int use_exit);
}

namespace {

class Scorer {
 public:
  explicit Scorer(nb::bytes blob) : handle_(fastdet_open(reinterpret_cast<const uint8_t*>(blob.c_str()), blob.size())) {
    if (handle_ == nullptr) throw std::invalid_argument("the C++ scorer rejected the model blob");
  }
  ~Scorer() {
    if (handle_ != nullptr) fastdet_close(handle_);
  }
  Scorer(const Scorer&) = delete;
  Scorer& operator=(const Scorer&) = delete;

  size_t native_size() const { return fastdet_native_size(handle_); }
  size_t cells() const { return fastdet_cells(handle_); }

  // native: Detector.native_matrix as a contiguous float32 array; returns kCells probabilities.
  nb::ndarray<nb::numpy, float, nb::ndim<1>> score(
      nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu> native, bool use_exit) {
    if (native.shape(0) != native_size())
      throw std::invalid_argument("native matrix has " + std::to_string(native.shape(0)) + " values, the model wants " +
                                  std::to_string(native_size()));
    const size_t n = cells();
    float* out = new float[n];
    nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    const int status = fastdet_score(handle_, native.data(), out, use_exit ? 1 : 0);
    if (status != 0) throw std::runtime_error("the C++ scorer failed with status " + std::to_string(status));
    return nb::ndarray<nb::numpy, float, nb::ndim<1>>(out, {n}, owner);
  }

 private:
  void* handle_;
};

}  // namespace

NB_MODULE(_native_ext, m) {
  m.doc() = "fastdet's C++ scorer (Highway SIMD), built with the package";
  m.def("target", [] { return std::string(fastdet_target()); }, "the SIMD target this module was compiled for");
  nb::class_<Scorer>(m, "Scorer")
      .def(nb::init<nb::bytes>(), nb::arg("blob"), "load an FDT1 container or bare IMSY blob")
      .def_prop_ro("native_size", &Scorer::native_size, "floats in Detector.native_matrix for this model")
      .def_prop_ro("cells", &Scorer::cells, "cells in the output grid (64 x 64)")
      .def("score", &Scorer::score, nb::arg("native"), nb::arg("use_exit") = true,
           "probabilities (row-major grid) for one native-resolution feature matrix");
}

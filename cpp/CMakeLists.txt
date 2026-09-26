cmake_minimum_required(VERSION 3.15)
project(fastdet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Highway picks its SIMD target from these flags (static dispatch, as in imfeat).
# No fast-math: the scorer is gated bit-for-bit against its scalar reference.
# Override FASTDET_ARCH_FLAGS for a portable or cross build, e.g.
# "-march=haswell -maes -mpclmul" (GCC's haswell omits AES, which Highway's AVX2
# target requires) or "-march=armv8.2-a".
if(MSVC)
  set(FASTDET_ARCH_FLAGS "/arch:AVX2" CACHE STRING "SIMD target flags")
  add_compile_options(/O2)
else()
  set(FASTDET_ARCH_FLAGS "-march=native" CACHE STRING "SIMD target flags")
  add_compile_options(-O3)
endif()
separate_arguments(_fastdet_arch NATIVE_COMMAND "${FASTDET_ARCH_FLAGS}")
add_compile_options(${_fastdet_arch})

# Google Highway, same pin as imfeat. For offline builds point CMake at a local
# checkout: -DFETCHCONTENT_SOURCE_DIR_HIGHWAY=/path/to/highway
include(FetchContent)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG 1.2.0
  GIT_SHALLOW TRUE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(highway)

add_executable(fastdet_score fastdet_score.cpp)
target_link_libraries(fastdet_score PRIVATE hwy)

# The Python extension (fastdet._native_ext), built when scikit-build-core drives CMake during
# `pip install` (SKBUILD is set) or when asked for explicitly.
option(FASTDET_BUILD_PYTHON "Build the nanobind extension module" OFF)
if(SKBUILD OR FASTDET_BUILD_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(nanobind CONFIG REQUIRED)
  # NOMINSIZE: nanobind builds for size (-Os) by default; the scorer is compute-bound.
  nanobind_add_module(_native_ext NOMINSIZE fastdet_score.cpp bindings.cpp)
  target_compile_definitions(_native_ext PRIVATE FASTDET_LIBRARY=1)
  target_link_libraries(_native_ext PRIVATE hwy)
  install(TARGETS _native_ext LIBRARY DESTINATION fastdet)
endif()

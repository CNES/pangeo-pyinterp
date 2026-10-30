# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
include(ProcessorCount)
include_directories(include)

ProcessorCount(NUM_CORES)

# Bake the resolved version into a generated header. It lands in the build tree
# so the source tree stays free of generated files, and is exposed through the
# library's interface include directories so consumers see it too.
set(PYINTERP_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/pyinterp/version.hpp.in"
               "${PYINTERP_GENERATED_INCLUDE_DIR}/pyinterp/version.hpp" @ONLY)
include_directories("${PYINTERP_GENERATED_INCLUDE_DIR}")

# Create the pure C++ library used by pybind
file(GLOB_RECURSE LIB_SRC "src/library/*.cpp")
add_library(pyinterp ${LIB_SRC})
# Namespaced alias, so a parent project can link against pyinterp::pyinterp
# whether it pulled the sources in with add_subdirectory/FetchContent or found
# an installed copy.
add_library(pyinterp::pyinterp ALIAS pyinterp)
if(ENABLE_COVERAGE)
  add_coverage(pyinterp)
endif()
target_include_directories(
  pyinterp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
                     ${PYINTERP_GENERATED_INCLUDE_DIR})
set_target_properties(pyinterp PROPERTIES VERSION "${PROJECT_VERSION}")
if(MSVC)
  target_compile_options(pyinterp PRIVATE /MD)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(pyinterp PRIVATE _ITERATOR_DEBUG_LEVEL=0)
  endif()
endif()

if(PYINTERP_BUILD_PYTHON_BINDINGS)
  file(GLOB_RECURSE NB_SRC "src/pybind/*.cpp")
  # NOMINSIZE: in Release, RelWithDebInfo and MinSizeRel builds, nanobind
  # compiles the module with -Os, which disables the vectorization and inlining
  # of the Eigen kernels.
  if(PYINTERP_NANOBIND_BACKEND_MODULE)
    # Split mode: no nanobind library is linked in, the backend module supplies
    # it at import time and the extension targets the stable ABI. The abi3 floor
    # is the oldest supported Python, which setup.py writes in the wheel tag
    # (cp312-abi3). A free-threaded build needs the abi3t variant of Python
    # 3.15, nanobind's default floor there (it rejects older interpreters).
    if(NB_ABI MATCHES "[0-9]t")
      set(PYINTERP_STABLE_ABI_VERSION)
    else()
      set(PYINTERP_STABLE_ABI_VERSION STABLE_ABI_VERSION 3.12)
    endif()
    nanobind_add_module(
      core
      FREE_THREADED
      NOMINSIZE
      BACKEND_MODULE
      ${PYINTERP_NANOBIND_BACKEND_MODULE}
      ${PYINTERP_STABLE_ABI_VERSION}
      ${NB_SRC})
  else()
    nanobind_add_module(core FREE_THREADED NOMINSIZE ${NB_SRC})
  endif()
  target_link_libraries(
    core
    PRIVATE BLAS::BLAS pyinterp
    PUBLIC cpp_coverage)
  # Most translation units parse the same heavy third-party headers (Boost
  # Geometry, Eigen, nanobind); precompiling them roughly halves the CPU time of
  # the build. clang-tidy cannot load a precompiled header produced by GCC, so
  # the header is skipped when the analysis is requested.
  if(PYINTERP_USE_PCH AND NOT ENABLE_CLANG_TIDY)
    target_precompile_headers(core PRIVATE
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/pch.hpp)
    # By default, Clang records the modification time of the precompiled headers
    # and rejects the PCH as soon as one of them is newer, which prevents ccache
    # from reusing a cached PCH once the dependencies have been reinstalled
    # (e.g. a fresh conda environment on CI). CMake already rebuilds the PCH
    # whenever one of these headers changes.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      target_compile_options(core PRIVATE "SHELL:-Xclang -fno-pch-timestamp")
    endif()
  endif()
  if(ENABLE_COVERAGE)
    add_coverage(core)
  endif()
  if(MSVC)
    target_compile_options(core PRIVATE /MD)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      target_compile_definitions(core PRIVATE _ITERATOR_DEBUG_LEVEL=0)
    endif()
  endif()
endif()

if(PYINTERP_BUILD_TESTS AND GTest_FOUND)
  add_subdirectory(tests)
endif()

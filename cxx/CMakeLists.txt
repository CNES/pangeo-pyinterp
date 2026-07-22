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
  nanobind_add_module(core FREE_THREADED ${NB_SRC})
  target_link_libraries(
    core
    PRIVATE BLAS::BLAS pyinterp
    PUBLIC cpp_coverage)
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

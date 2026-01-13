# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
include(ProcessorCount)
include_directories(include)

ProcessorCount(NUM_CORES)

# Create the pure C++ library used by pybind
file(GLOB_RECURSE LIB_SRC "src/library/*.cpp")
add_library(pyinterp ${LIB_SRC})
if(ENABLE_COVERAGE)
  add_coverage(pyinterp)
endif()
target_include_directories(pyinterp
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MSVC)
  target_compile_options(pyinterp PRIVATE /MD)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(pyinterp PRIVATE _ITERATOR_DEBUG_LEVEL=0)
  endif()
endif()

file(GLOB_RECURSE NB_SRC "src/pybind/*.cpp")
nanobind_add_module(core ${NB_SRC})
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

if(GTest_FOUND)
  add_subdirectory(tests)
endif()

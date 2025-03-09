# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
include(ProcessorCount)
include_directories(include)

ProcessorCount(NUM_CORES)

file(GLOB_RECURSE IMPLEMENT "lib/*.cpp")
add_library(pyinterp STATIC ${IMPLEMENT})
target_include_directories(pyinterp PUBLIC ${Python3_INCLUDE_DIRS})
target_link_libraries(
  pyinterp
  PRIVATE
  PUBLIC cpp_coverage)

file(GLOB_RECURSE SOURCES "module/*.cpp")
pybind11_add_module(core ${SOURCES})
target_link_libraries(
  core
  PRIVATE pyinterp ${BLAS_LIBRARIES}
  PUBLIC cpp_coverage)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set_property(
    TARGET core
    APPEND
    PROPERTY COMPILE_OPTIONS -flto=${NUM_CORES})
  set_property(
    TARGET core
    APPEND
    PROPERTY LINK_OPTIONS -flto=${NUM_CORES})
elseif(MSVC)
  set_property(
    TARGET core
    APPEND
    PROPERTY COMPILE_OPTIONS /GL)
  set_property(
    TARGET core
    APPEND
    PROPERTY LINK_OPTIONS /LTCG)
endif()

if(GTest_FOUND)
  add_subdirectory(tests)
endif()

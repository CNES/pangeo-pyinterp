# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
include_directories(include)

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
  PRIVATE pyinterp GSL::gsl
  PUBLIC cpp_coverage)

if(GTest_FOUND)
  add_subdirectory(tests)
endif()

# Copyright (c) 2020 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
find_package(MKL)
include_directories(include)

file(GLOB_RECURSE IMPLEMENT "detail/*.cpp")
add_library(pyinterp STATIC ${IMPLEMENT})
target_link_libraries(pyinterp PUBLIC cpp_coverage)


file(GLOB_RECURSE SOURCES "module/*.cpp")
pybind11_add_module(core ${SOURCES})
if(MKL_FOUND)
    add_definitions(-DEIGEN_USE_MKL_ALL)
    include_directories(${MKL_INCLUDE_DIRS})
    target_link_libraries(core PUBLIC ${MKL_LIBRARIES})
endif()

target_link_libraries(
    core
        PRIVATE pyinterp GSL::gsl GSL::gslcblas
        PUBLIC cpp_coverage ${MKL_LIBRARIES}
)

add_subdirectory(tests)

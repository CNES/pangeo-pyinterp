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
    set_target_properties(GSL::gsl PROPERTIES
        IMPORTED_LOCATION                 "${GSL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES     "${GSL_INCLUDE_DIRS}"
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        INTERFACE_LINK_LIBRARIES          "${MKL_LIBRARIES}" )
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    add_definitions(-DEIGEN_USE_BLAS)
    add_definitions(-DEIGEN_USE_LAPACKE)
    set_target_properties(GSL::gsl PROPERTIES
        IMPORTED_LOCATION                 "${GSL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES     "${GSL_INCLUDE_DIRS}"
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        INTERFACE_LINK_LIBRARIES          "${BLAS_LIBRARIES}" )
endif()

target_link_libraries(
    core
        PRIVATE pyinterp GSL::gsl
        PUBLIC cpp_coverage
)

add_subdirectory(tests)

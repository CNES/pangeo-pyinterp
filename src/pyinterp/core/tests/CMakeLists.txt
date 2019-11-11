# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
macro (add_testcase testname)
    set(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${testname}.cpp")
    add_executable(test_${testname} ${FILES})
    target_link_libraries(test_${testname} pyinterp gtest_main ${ARGN} cpp_coverage)
    add_test(NAME test_${testname} COMMAND ${EXECUTABLE_OUTPUT_PATH}/test_${testname})
endmacro()

add_testcase(axis)
add_testcase(axis_container)
add_testcase(geodetic_coordinates)
add_testcase(geodetic_rtree)
add_testcase(geodetic_system)
add_testcase(geometry_rtree)
add_testcase(gsl GSL::gsl GSL::gslcblas)
add_testcase(math)
add_testcase(math_bicubic GSL::gsl GSL::gslcblas)
add_testcase(math_binning)
add_testcase(math_bivariate)
add_testcase(math_linear)
add_testcase(math_trivariate)
add_testcase(thread)

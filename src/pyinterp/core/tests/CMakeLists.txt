# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
enable_testing()

# Create a new test suite.
macro(ADD_TESTCASE testname)
  set(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${testname}.cpp")
  add_executable(test_${testname} ${FILES})
  target_link_libraries(test_${testname} pyinterp GTest::gtest_main ${ARGN}
                        cpp_coverage)
  add_test(NAME test_${testname}
           COMMAND ${EXECUTABLE_OUTPUT_PATH}/test_${testname})
endmacro()

add_testcase(axis_container)
add_testcase(axis)
add_testcase(geodetic_coordinates)
add_testcase(geodetic_system)
add_testcase(geometry_rtree ${BLAS_LIBRARIES})
add_testcase(interpolation_akima)
add_testcase(interpolation_bicubic ${BLAS_LIBRARIES})
add_testcase(interpolation_bilinear)
add_testcase(interpolation_cspline ${BLAS_LIBRARIES})
add_testcase(interpolation_linear)
add_testcase(interpolation_polynomial)
add_testcase(interpolation_search)
add_testcase(interpolation_steffen)
add_testcase(math_bicubic ${BLAS_LIBRARIES})
add_testcase(math_binning)
add_testcase(math_bivariate ${BLAS_LIBRARIES})
add_testcase(math_descriptive_statistics)
add_testcase(math_kriging ${BLAS_LIBRARIES})
add_testcase(math_linear)
add_testcase(math_rbf ${BLAS_LIBRARIES})
add_testcase(math_spline ${BLAS_LIBRARIES})
add_testcase(math_streaming_histogram)
add_testcase(math_trivariate)
add_testcase(math_window_function)
add_testcase(math)
add_testcase(thread)

// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include <boost/geometry.hpp>

#include "pyinterp/detail/math/window_functions.hpp"

namespace math = pyinterp::detail::math;

TEST(math_window_function, hamming) {
  auto wi = math::window::hamming(0.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 1.0, 1e-6);
  wi = math::window::hamming(1.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.9118346052832507, 1e-6);
  wi = math::window::hamming(2.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.6810146052832508, 1e-6);
  wi = math::window::hamming(3.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.3957053947167493, 1e-6);
  wi = math::window::hamming(4.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.1648853947167493, 1e-6);
  wi = math::window::hamming(5.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.07671999999999995, 1e-6);
}

TEST(math_window_function, blackman) {
  auto wi = math::window::blackman(0.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 1.0, 1e-6);
  wi = math::window::blackman(1.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.8520642374237258, 1e-6);
  wi = math::window::blackman(2.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.5178645059151086, 1e-6);
  wi = math::window::blackman(3.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.2109731658436862, 1e-6);
  wi = math::window::blackman(4.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.04861227826373925, 1e-6);
  wi = math::window::blackman(5.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.006878761822871851, 1e-6);
}

TEST(math_window_function, flat_top) {
  auto wi = math::window::flat_top(0.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 1.000000003, 1e-6);
  wi = math::window::flat_top(1.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.6068721525762121, 1e-6);
  wi = math::window::flat_top(2.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.05454464816043305, 1e-6);
  wi = math::window::flat_top(3.0, 5.0, 0.0);
  EXPECT_NEAR(wi, -0.06771425207621193, 1e-6);
  wi = math::window::flat_top(4.0, 5.0, 0.0);
  EXPECT_NEAR(wi, -0.01559727466043301, 1e-6);
  wi = math::window::flat_top(5.0, 5.0, 0.0);
  EXPECT_NEAR(wi, -0.0004210510000000013, 1e-6);
}

TEST(math_window_function, nuttall) {
  auto wi = math::window::nuttall(0.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.9893589, 1e-6);
  wi = math::window::nuttall(1.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.8015463776889715, 1e-6);
  wi = math::window::nuttall(2.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.40423474384273034, 1e-6);
  wi = math::window::nuttall(3.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.1019064223110286, 1e-6);
  wi = math::window::nuttall(4.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.010040556157269807, 1e-6);
  wi = math::window::nuttall(5.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.011003900000000039, 1e-6);
}

TEST(math_window_function, blackman_harris) {
  auto wi = math::window::blackman_harris(0.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 1.0, 1e-6);
  wi = math::window::blackman_harris(1.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.7938335106543364, 1e-6);
  wi = math::window::blackman_harris(2.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.3858926687237512, 1e-6);
  wi = math::window::blackman_harris(3.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.1030114893456638, 1e-6);
  wi = math::window::blackman_harris(4.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.01098233127624889, 1e-6);
  wi = math::window::blackman_harris(5.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 6.0000000000001025e-05, 1e-6);
}

TEST(math_window_function, parzen) {
  auto wi = math::window::parzen(0.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 1.0, 1e-6);
  wi = math::window::parzen(1.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.8079999999999999, 1e-6);
  wi = math::window::parzen(2.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.42399999999999993, 1e-6);
  wi = math::window::parzen(3.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.12800000000000003, 1e-6);
  wi = math::window::parzen(4.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.01599999999999999, 1e-6);
  wi = math::window::parzen(5.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.0, 1e-6);
}

TEST(math_window_function, parzen_swot) {
  auto wi = math::window::parzen_swot(0.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 1.0, 1e-6);
  wi = math::window::parzen_swot(1.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.808, 1e-6);
  wi = math::window::parzen_swot(2.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.42399999999999993, 1e-6);
  wi = math::window::parzen_swot(3.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.12800000000000003, 1e-6);
  wi = math::window::parzen_swot(4.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.01599999999999999, 1e-6);
  wi = math::window::parzen_swot(5.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.0, 1e-6);
}

TEST(math_window_function, lanczos) {
  auto wi = math::window::lanczos(0.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 1.0, 1e-6);
  wi = math::window::lanczos(1.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.9201768612999938, 1e-6);
  wi = math::window::lanczos(2.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.7080032943281469, 1e-6);
  wi = math::window::lanczos(3.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.43310352619879655, 1e-6);
  wi = math::window::lanczos(4.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.17700082358203678, 1e-6);
  wi = math::window::lanczos(5.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 2.4816532646682024e-17, 1e-6);
  wi = math::window::lanczos(6.0, 5.0, 2.0);
  EXPECT_NEAR(wi, -0.07866703270312743, 1e-6);
  wi = math::window::lanczos(7.0, 5.0, 2.0);
  EXPECT_NEAR(wi, -0.07954962726100345, 1e-6);
  wi = math::window::lanczos(8.0, 5.0, 2.0);
  EXPECT_NEAR(wi, -0.044250205895509195, 1e-6);
  wi = math::window::lanczos(9.0, 5.0, 2.0);
  EXPECT_NEAR(wi, -0.011360208164197461, 1e-6);
  wi = math::window::lanczos(10.0, 5.0, 2.0);
  EXPECT_NEAR(wi, -1.5195743635847466e-33, 1e-6);
  wi = math::window::lanczos(11.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.0, 1e-6);
  wi = math::window::lanczos(12.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.0, 1e-6);
  wi = math::window::lanczos(13.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.0, 1e-6);
  wi = math::window::lanczos(14.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.0, 1e-6);
  wi = math::window::lanczos(15.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.0, 1e-6);
}

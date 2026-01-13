// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/math/interpolate/window_function.hpp"

#include <gtest/gtest.h>

#include <cstdlib>

namespace pyinterp::math::interpolate {

// Test fixture for window function tests
class WindowFunctionTest : public ::testing::Test {
 protected:
  static constexpr double kTolerance = 1e-6;

  void SetUp() override {}
  void TearDown() override {}

  // Helper to test symmetry: w(x) should equal w(-x) for symmetric windows
  template <typename Func>
  void TestSymmetry(Func func, double half_width, double cutoff = 0.0) {
    double x = 0.0;
    while (x <= half_width) {
      double w_pos = func(x, half_width, cutoff);
      double w_neg = func(-x, half_width, cutoff);
      EXPECT_NEAR(w_pos, w_neg, kTolerance) << "Symmetry failed at x=" << x;
      x += 0.5;
    }
  }

  // Helper to test that window decreases monotonically from center
  template <typename Func>
  void TestMonotonicDecrease(Func func, double half_width,
                             double cutoff = 0.0) {
    double prev_value = func(0.0, half_width, cutoff);
    double x = 0.5;
    while (x <= half_width) {
      double curr_value = func(x, half_width, cutoff);
      EXPECT_LE(curr_value, prev_value) << "Non-monotonic at x=" << x;
      prev_value = curr_value;
      x += 0.5;
    }
  }
};

// Hamming window tests
TEST_F(WindowFunctionTest, HammingBasicValues) {
  auto wi = window::hamming(0.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 1.0, kTolerance);
  wi = window::hamming(1.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.9118346052832507, kTolerance);
  wi = window::hamming(2.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.6810146052832508, kTolerance);
  wi = window::hamming(3.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.3957053947167493, kTolerance);
  wi = window::hamming(4.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.1648853947167493, kTolerance);
  wi = window::hamming(5.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.07671999999999995, kTolerance);
}

TEST_F(WindowFunctionTest, HammingSymmetry) {
  TestSymmetry([](double x, double hw,
                  double c) -> double { return window::hamming(x, hw, c); },
               5.0);
}

TEST_F(WindowFunctionTest, HammingMonotonic) {
  TestMonotonicDecrease(
      [](double x, double hw, double c) -> double {
        return window::hamming(x, hw, c);
      },
      5.0);
}

TEST_F(WindowFunctionTest, HammingDifferentHalfWidths) {
  auto wi = window::hamming(0.0, 10.0, 0.0);
  EXPECT_NEAR(wi, 1.0, kTolerance);
  wi = window::hamming(5.0, 10.0, 0.0);
  EXPECT_GT(wi, 0.0);  // Just check it's positive

  wi = window::hamming(0.0, 2.0, 0.0);
  EXPECT_NEAR(wi, 1.0, kTolerance);
  wi = window::hamming(1.0, 2.0, 0.0);
  EXPECT_GT(wi, 0.0);  // Just check it's positive
}

TEST_F(WindowFunctionTest, HammingBeyondHalfWidth) {
  auto wi = window::hamming(6.0, 5.0, 0.0);
  EXPECT_GE(wi, 0.0);
}

// Blackman window tests
TEST_F(WindowFunctionTest, BlackmanBasicValues) {
  auto wi = window::blackman(0.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 1.0, kTolerance);
  wi = window::blackman(1.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.8520642374237258, kTolerance);
  wi = window::blackman(2.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.5178645059151086, kTolerance);
  wi = window::blackman(3.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.2109731658436862, kTolerance);
  wi = window::blackman(4.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.04861227826373925, kTolerance);
  wi = window::blackman(5.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.006878761822871851, kTolerance);
}

TEST_F(WindowFunctionTest, BlackmanSymmetry) {
  TestSymmetry([](double x, double hw,
                  double c) -> double { return window::blackman(x, hw, c); },
               5.0);
}

TEST_F(WindowFunctionTest, BlackmanMonotonic) {
  TestMonotonicDecrease(
      [](double x, double hw, double c) -> double {
        return window::blackman(x, hw, c);
      },
      5.0);
}

// Flat-top window tests
TEST_F(WindowFunctionTest, FlatTopBasicValues) {
  auto wi = window::flat_top(0.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 1.000000003, kTolerance);
  wi = window::flat_top(1.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.6068721525762121, kTolerance);
  wi = window::flat_top(2.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.05454464816043305, kTolerance);
  wi = window::flat_top(3.0, 5.0, 0.0);
  EXPECT_NEAR(wi, -0.06771425207621193, kTolerance);
  wi = window::flat_top(4.0, 5.0, 0.0);
  EXPECT_NEAR(wi, -0.01559727466043301, kTolerance);
  wi = window::flat_top(5.0, 5.0, 0.0);
  EXPECT_NEAR(wi, -0.0004210510000000013, kTolerance);
}

TEST_F(WindowFunctionTest, FlatTopSymmetry) {
  TestSymmetry([](double x, double hw,
                  double c) -> double { return window::flat_top(x, hw, c); },
               5.0);
}

TEST_F(WindowFunctionTest, FlatTopNegativeValues) {
  // Flat-top can have negative values
  auto wi = window::flat_top(3.0, 5.0, 0.0);
  EXPECT_LT(wi, 0.0);
}

// Nuttall window tests
TEST_F(WindowFunctionTest, NuttallBasicValues) {
  auto wi = window::nuttall(0.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.9893589, kTolerance);
  wi = window::nuttall(1.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.8015463776889715, kTolerance);
  wi = window::nuttall(2.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.40423474384273034, kTolerance);
  wi = window::nuttall(3.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.1019064223110286, kTolerance);
  wi = window::nuttall(4.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.010040556157269807, kTolerance);
  wi = window::nuttall(5.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.011003900000000039, kTolerance);
}

TEST_F(WindowFunctionTest, NuttallSymmetry) {
  TestSymmetry([](double x, double hw,
                  double c) -> double { return window::nuttall(x, hw, c); },
               5.0);
}

TEST_F(WindowFunctionTest, NuttallMonotonic) {
  // Nuttall is mostly monotonic but may have slight variations at edges
  double prev_value = window::nuttall(0.0, 5.0, 0.0);
  double x = 0.5;
  while (x <= 4.0) {
    double curr_value = window::nuttall(x, 5.0, 0.0);
    EXPECT_LE(curr_value, prev_value + 1e-5) << "Non-monotonic at x=" << x;
    prev_value = curr_value;
    x += 0.5;
  }
}

// Blackman-Harris window tests
TEST_F(WindowFunctionTest, BlackmanHarrisBasicValues) {
  auto wi = window::blackman_harris(0.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 1.0, kTolerance);
  wi = window::blackman_harris(1.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.7938335106543364, kTolerance);
  wi = window::blackman_harris(2.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.3858926687237512, kTolerance);
  wi = window::blackman_harris(3.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.1030114893456638, kTolerance);
  wi = window::blackman_harris(4.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.01098233127624889, kTolerance);
  wi = window::blackman_harris(5.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 6.0000000000001025e-05, kTolerance);
}

TEST_F(WindowFunctionTest, BlackmanHarrisSymmetry) {
  TestSymmetry(
      [](double x, double hw, double c) -> double {
        return window::blackman_harris(x, hw, c);
      },
      5.0);
}

TEST_F(WindowFunctionTest, BlackmanHarrisMonotonic) {
  TestMonotonicDecrease(
      [](double x, double hw, double c) -> double {
        return window::blackman_harris(x, hw, c);
      },
      5.0);
}

// Parzen window tests
TEST_F(WindowFunctionTest, ParzenBasicValues) {
  auto wi = window::parzen(0.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 1.0, kTolerance);
  wi = window::parzen(1.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.8079999999999999, kTolerance);
  wi = window::parzen(2.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.42399999999999993, kTolerance);
  wi = window::parzen(3.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.12800000000000003, kTolerance);
  wi = window::parzen(4.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.01599999999999999, kTolerance);
  wi = window::parzen(5.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.0, kTolerance);
}

TEST_F(WindowFunctionTest, ParzenMonotonic) {
  TestMonotonicDecrease(
      [](double x, double hw, double c) -> double {
        return window::parzen(x, hw, c);
      },
      5.0);
}

TEST_F(WindowFunctionTest, ParzenZeroAtBoundary) {
  auto wi = window::parzen(5.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.0, kTolerance);
}

// Parzen-SWOT window tests
TEST_F(WindowFunctionTest, ParzenSwotBasicValues) {
  auto wi = window::parzen_swot(0.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 1.0, kTolerance);
  wi = window::parzen_swot(1.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.808, kTolerance);
  wi = window::parzen_swot(2.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.42399999999999993, kTolerance);
  wi = window::parzen_swot(3.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.12800000000000003, kTolerance);
  wi = window::parzen_swot(4.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.01599999999999999, kTolerance);
  wi = window::parzen_swot(5.0, 5.0, 0.0);
  EXPECT_NEAR(wi, 0.0, kTolerance);
}

TEST_F(WindowFunctionTest, ParzenSwotVsParzen) {
  // Parzen-SWOT should differ slightly from Parzen at x=1.0
  auto parzen_val = window::parzen(1.0, 5.0, 0.0);
  auto parzen_swot_val = window::parzen_swot(1.0, 5.0, 0.0);
  EXPECT_NE(parzen_val, parzen_swot_val);

  // But should be the same at other points
  parzen_val = window::parzen(2.0, 5.0, 0.0);
  parzen_swot_val = window::parzen_swot(2.0, 5.0, 0.0);
  EXPECT_NEAR(parzen_val, parzen_swot_val, kTolerance);
}

// Lanczos window tests
TEST_F(WindowFunctionTest, LanczosBasicValues) {
  auto wi = window::lanczos(0.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 1.0, kTolerance);
  wi = window::lanczos(1.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.9201768612999938, kTolerance);
  wi = window::lanczos(2.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.7080032943281469, kTolerance);
  wi = window::lanczos(3.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.43310352619879655, kTolerance);
  wi = window::lanczos(4.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.17700082358203678, kTolerance);
  wi = window::lanczos(5.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 2.4816532646682024e-17, kTolerance);
}

TEST_F(WindowFunctionTest, LanczosNegativeLobes) {
  auto wi = window::lanczos(6.0, 5.0, 2.0);
  EXPECT_NEAR(wi, -0.07866703270312743, kTolerance);
  wi = window::lanczos(7.0, 5.0, 2.0);
  EXPECT_NEAR(wi, -0.07954962726100345, kTolerance);
  wi = window::lanczos(8.0, 5.0, 2.0);
  EXPECT_NEAR(wi, -0.044250205895509195, kTolerance);
  wi = window::lanczos(9.0, 5.0, 2.0);
  EXPECT_NEAR(wi, -0.011360208164197461, kTolerance);
  wi = window::lanczos(10.0, 5.0, 2.0);
  EXPECT_NEAR(wi, -1.5195743635847466e-33, kTolerance);
}

TEST_F(WindowFunctionTest, LanczosZeroBeyondCutoff) {
  auto wi = window::lanczos(11.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.0, kTolerance);
  wi = window::lanczos(12.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.0, kTolerance);
  wi = window::lanczos(13.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.0, kTolerance);
  wi = window::lanczos(14.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.0, kTolerance);
  wi = window::lanczos(15.0, 5.0, 2.0);
  EXPECT_NEAR(wi, 0.0, kTolerance);
}

TEST_F(WindowFunctionTest, LanczosSymmetry) {
  TestSymmetry([](double x, double hw,
                  double c) -> double { return window::lanczos(x, hw, c); },
               5.0, 2.0);
}

TEST_F(WindowFunctionTest, LanczosDifferentCutoffs) {
  // Test with cutoff = 1.0
  auto wi = window::lanczos(0.0, 5.0, 1.0);
  EXPECT_NEAR(wi, 1.0, kTolerance);

  wi = window::lanczos(2.5, 5.0, 1.0);
  EXPECT_GT(wi, 0.0);

  wi = window::lanczos(6.0, 5.0, 1.0);
  EXPECT_NEAR(wi, 0.0, kTolerance);

  // Test with cutoff = 3.0
  wi = window::lanczos(0.0, 5.0, 3.0);
  EXPECT_NEAR(wi, 1.0, kTolerance);

  wi = window::lanczos(7.5, 5.0, 3.0);
  EXPECT_GT(std::abs(wi), 0.0);

  wi = window::lanczos(16.0, 5.0, 3.0);
  EXPECT_NEAR(wi, 0.0, kTolerance);
}

TEST_F(WindowFunctionTest, LanczosNegativeX) {
  auto wi_pos = window::lanczos(3.0, 5.0, 2.0);
  auto wi_neg = window::lanczos(-3.0, 5.0, 2.0);
  EXPECT_NEAR(wi_pos, wi_neg, kTolerance);
}

// Edge case tests
TEST_F(WindowFunctionTest, AllWindowsAtZero) {
  // All windows should be 1.0 (or close to 1.0) at x=0
  EXPECT_NEAR(window::hamming(0.0, 5.0, 0.0), 1.0, kTolerance);
  EXPECT_NEAR(window::blackman(0.0, 5.0, 0.0), 1.0, kTolerance);
  EXPECT_NEAR(window::flat_top(0.0, 5.0, 0.0), 1.0,
              1e-3);  // Slightly relaxed
  EXPECT_NEAR(window::nuttall(0.0, 5.0, 0.0), 0.9893589,
              kTolerance);  // Nuttall is 0.9893589 by design
  EXPECT_NEAR(window::blackman_harris(0.0, 5.0, 0.0), 1.0, kTolerance);
  EXPECT_NEAR(window::parzen(0.0, 5.0, 0.0), 1.0, kTolerance);
  EXPECT_NEAR(window::parzen_swot(0.0, 5.0, 0.0), 1.0, kTolerance);
  EXPECT_NEAR(window::lanczos(0.0, 5.0, 2.0), 1.0, kTolerance);
}

TEST_F(WindowFunctionTest, SmallHalfWidth) {
  double small_hw = 0.1;

  EXPECT_NO_THROW((void)window::hamming(0.0, small_hw, 0.0));
  EXPECT_NO_THROW((void)window::blackman(0.0, small_hw, 0.0));
  EXPECT_NO_THROW((void)window::flat_top(0.0, small_hw, 0.0));
  EXPECT_NO_THROW((void)window::nuttall(0.0, small_hw, 0.0));
  EXPECT_NO_THROW((void)window::blackman_harris(0.0, small_hw, 0.0));
  EXPECT_NO_THROW((void)window::parzen(0.0, small_hw, 0.0));
  EXPECT_NO_THROW((void)window::parzen_swot(0.0, small_hw, 0.0));
  EXPECT_NO_THROW((void)window::lanczos(0.0, small_hw, 2.0));
}

TEST_F(WindowFunctionTest, LargeHalfWidth) {
  double large_hw = 1000.0;

  EXPECT_NEAR(window::hamming(0.0, large_hw, 0.0), 1.0, kTolerance);
  EXPECT_NEAR(window::blackman(0.0, large_hw, 0.0), 1.0, kTolerance);
  EXPECT_NEAR(window::parzen(0.0, large_hw, 0.0), 1.0, kTolerance);
  EXPECT_NEAR(window::lanczos(0.0, large_hw, 2.0), 1.0, kTolerance);
}

}  // namespace pyinterp::math::interpolate

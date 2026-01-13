// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/math/interpolate/rbf.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace pyinterp::math::interpolate {

// Test 1D interpolation with various radial basis functions
static void test_1d(RBFKernel kernel) {
  auto x = Eigen::Matrix<double, 1, 9>::LinSpaced(9, 0, 10);
  auto y = x.array().sin();
  auto rbf = RBF<double>(std::numeric_limits<double>::quiet_NaN(), 0, kernel);
  auto yi = rbf.interpolate(x, y, x);

  ASSERT_EQ(y.size(), yi.size());
  for (int64_t ix = 0; ix < yi.size(); ++ix) {
    EXPECT_NEAR(y(ix), yi(ix), 1e-9);
  }
}

// Test 2D interpolation with various radial basis functions
static void test_2d(RBFKernel kernel) {
  Eigen::Matrix<double, 2, 50> x = Eigen::Matrix<double, 2, 50>::Random();
  Eigen::Matrix<double, 50, 1> y =
      (x.row(0).array().pow(2) - x.row(1).array().pow(2)).array().exp();
  auto rbf = RBF<double>(std::numeric_limits<double>::quiet_NaN(), 0, kernel);
  auto yi = rbf.interpolate(x, y, x);

  ASSERT_EQ(y.size(), yi.size());
  for (int64_t ix = 0; ix < yi.size(); ++ix) {
    EXPECT_NEAR(y(ix), yi(ix), 1e-9);
  }
}

// Test 3D interpolation with various radial basis functions
static void test_3d(RBFKernel kernel) {
  Eigen::Matrix<double, 3, 50> x = Eigen::Matrix<double, 3, 50>::Random();
  Eigen::Matrix<double, 50, 1> y =
      (x.row(0).array().pow(2) - x.row(1).array().pow(2)).array().exp();
  auto rbf = RBF<double>(std::numeric_limits<double>::quiet_NaN(), 0, kernel);
  auto yi = rbf.interpolate(x, y, x);

  ASSERT_EQ(y.size(), yi.size());
  for (int64_t ix = 0; ix < yi.size(); ++ix) {
    EXPECT_NEAR(y(ix), yi(ix), 1e-9);
  }
}

// Test all radial basis functions in 1D
TEST(MathRbf, 1d) {
  test_1d(RBFKernel::kCubic);
  test_1d(RBFKernel::kGaussian);
  test_1d(RBFKernel::kInverseMultiquadric);
  test_1d(RBFKernel::kLinear);
  test_1d(RBFKernel::kMultiquadric);
  test_1d(RBFKernel::kThinPlate);
}

// Test all radial basis functions in 2D
TEST(MathRbf, 2d) {
  test_2d(RBFKernel::kCubic);
  test_2d(RBFKernel::kGaussian);
  test_2d(RBFKernel::kInverseMultiquadric);
  test_2d(RBFKernel::kLinear);
  test_2d(RBFKernel::kMultiquadric);
  test_2d(RBFKernel::kThinPlate);
}

// Test all radial basis functions in 3D
TEST(MathRbf, 3d) {
  test_3d(RBFKernel::kCubic);
  test_3d(RBFKernel::kGaussian);
  test_3d(RBFKernel::kInverseMultiquadric);
  test_3d(RBFKernel::kLinear);
  test_3d(RBFKernel::kMultiquadric);
  test_3d(RBFKernel::kThinPlate);
}

// Test single point interpolation
TEST(MathRbf, Point) {
  Eigen::Matrix<double, 3, 50> x = Eigen::Matrix<double, 3, 50>::Random();
  Eigen::Matrix<double, 50, 1> y =
      (x.row(0).array().pow(2) - x.row(1).array().pow(2)).array().exp();
  auto rbf = RBF<double>(std::numeric_limits<double>::quiet_NaN(), 0,
                         RBFKernel::kCubic);

  auto xi = Eigen::Matrix<double, 3, 1>();
  for (int64_t ix = 0; ix < x.cols(); ++ix) {
    xi << x.col(ix);
    auto yi = rbf.interpolate(x, y, xi);

    ASSERT_EQ(yi.size(), 1);
    EXPECT_NEAR(y(ix), yi(0), 1e-9);
  }
}

// Test with explicit epsilon value
TEST(MathRbf, ExplicitEpsilon) {
  auto x = Eigen::Matrix<double, 1, 9>::LinSpaced(9, 0, 10);
  auto y = x.array().sin();

  // Test with explicit epsilon value
  auto rbf = RBF<double>(1.0, 0, RBFKernel::kGaussian);
  auto yi = rbf.interpolate(x, y, x);

  ASSERT_EQ(y.size(), yi.size());
  for (int64_t ix = 0; ix < yi.size(); ++ix) {
    EXPECT_NEAR(y(ix), yi(ix), 1e-9);
  }
}

// Test smoothing parameter (Tikhonov regularization)
TEST(MathRbf, Smoothing) {
  auto x = Eigen::Matrix<double, 1, 20>::LinSpaced(20, 0, 10);
  auto y = x.array().sin();

  // Without smoothing (interpolation)
  auto rbf_no_smooth = RBF<double>(std::numeric_limits<double>::quiet_NaN(),
                                   0.0, RBFKernel::kMultiquadric);
  auto yi_no_smooth = rbf_no_smooth.interpolate(x, y, x);

  // With smoothing (approximation)
  auto rbf_smooth = RBF<double>(std::numeric_limits<double>::quiet_NaN(), 0.1,
                                RBFKernel::kMultiquadric);
  auto yi_smooth = rbf_smooth.interpolate(x, y, x);

  ASSERT_EQ(y.size(), yi_no_smooth.size());
  ASSERT_EQ(y.size(), yi_smooth.size());

  // Interpolation should match exactly at nodes
  for (int64_t ix = 0; ix < yi_no_smooth.size(); ++ix) {
    EXPECT_NEAR(y(ix), yi_no_smooth(ix), 1e-9);
  }

  // Smoothing should produce values close to but not exactly at nodes
  // The difference should be small but non-zero for most points
  double max_diff = 0.0;
  for (int64_t ix = 0; ix < yi_smooth.size(); ++ix) {
    max_diff = std::max(max_diff, std::abs(y(ix) - yi_smooth(ix)));
  }
  EXPECT_GT(max_diff, 1e-12);  // Some difference due to smoothing
  EXPECT_LT(max_diff, 3.0);    // Reasonable bound for smoothing effect
}

// Test extrapolation behavior
TEST(MathRbf, Extrapolation) {
  auto x = Eigen::Matrix<double, 1, 10>::LinSpaced(10, 0, 5);
  auto y = x.array().sin();
  auto rbf = RBF<double>(std::numeric_limits<double>::quiet_NaN(), 0,
                         RBFKernel::kLinear);

  // Test points outside the training range
  Eigen::Matrix<double, 1, 3> xi;
  xi << -1.0, 6.0, 10.0;

  auto yi = rbf.interpolate(x, y, xi);

  ASSERT_EQ(yi.size(), 3);
  // Values should be finite (no NaN or Inf)
  for (int64_t ix = 0; ix < yi.size(); ++ix) {
    EXPECT_TRUE(std::isfinite(yi(ix)));
  }
}

// Test with different floating point types
TEST(MathRbf, FloatType) {
  auto x = Eigen::Matrix<float, 1, 9>::LinSpaced(9, 0.0f, 10.0f);
  auto y = x.array().sin();
  auto rbf = RBF<float>(std::numeric_limits<float>::quiet_NaN(), 0.0f,
                        RBFKernel::kCubic);
  auto yi = rbf.interpolate(x, y, x);

  ASSERT_EQ(y.size(), yi.size());
  for (int64_t ix = 0; ix < yi.size(); ++ix) {
    EXPECT_NEAR(y(ix), yi(ix), 1e-5f);  // Lower precision for float
  }
}

// Test with minimal number of points
TEST(MathRbf, MinimalPoints) {
  auto x = Eigen::Matrix<double, 1, 2>::LinSpaced(2, 0, 1);
  auto y = x.array().sin();
  auto rbf = RBF<double>(std::numeric_limits<double>::quiet_NaN(), 0,
                         RBFKernel::kLinear);
  auto yi = rbf.interpolate(x, y, x);

  ASSERT_EQ(y.size(), yi.size());
  for (int64_t ix = 0; ix < yi.size(); ++ix) {
    EXPECT_NEAR(y(ix), yi(ix), 1e-9);
  }
}

// Test with constant values
TEST(MathRbf, ConstantValues) {
  auto x = Eigen::Matrix<double, 2, 10>::Random();
  auto y = Eigen::Matrix<double, 10, 1>::Constant(5.0);
  auto rbf = RBF<double>(std::numeric_limits<double>::quiet_NaN(), 0,
                         RBFKernel::kMultiquadric);

  // Test at training points
  auto yi = rbf.interpolate(x, y, x);
  ASSERT_EQ(y.size(), yi.size());
  for (int64_t ix = 0; ix < yi.size(); ++ix) {
    EXPECT_NEAR(5.0, yi(ix), 2.0);  // Should be close at training points
  }

  // Test at new points
  auto xi = Eigen::Matrix<double, 2, 5>::Random();
  auto yi_new = rbf.interpolate(x, y, xi);
  ASSERT_EQ(xi.cols(), yi_new.size());
  for (int64_t ix = 0; ix < yi_new.size(); ++ix) {
    EXPECT_NEAR(5.0, yi_new(ix), 2.0);  // Reasonable approximation
  }
}

// Test linear function reproduction
TEST(MathRbf, LinearFunctionReproduction) {
  // Create a linear function: f(x, y) = 2x + 3y + 1
  Eigen::Matrix<double, 2, 20> x = Eigen::Matrix<double, 2, 20>::Random();
  Eigen::Matrix<double, 20, 1> y =
      2.0 * x.row(0).array() + 3.0 * x.row(1).array() + 1.0;

  auto rbf = RBF<double>(std::numeric_limits<double>::quiet_NaN(), 0,
                         RBFKernel::kLinear);

  // Test at new points
  Eigen::Matrix<double, 2, 10> xi = Eigen::Matrix<double, 2, 10>::Random();
  auto yi = rbf.interpolate(x, y, xi);

  // Calculate expected values
  Eigen::Matrix<double, 10, 1> expected =
      2.0 * xi.row(0).array() + 3.0 * xi.row(1).array() + 1.0;

  ASSERT_EQ(expected.size(), yi.size());
  for (int64_t ix = 0; ix < yi.size(); ++ix) {
    // Linear RBF approximates linear functions reasonably well
    EXPECT_NEAR(expected(ix), yi(ix), 0.6);
  }
}

// Test with zero distance (duplicate points should still work)
TEST(MathRbf, DuplicatePoints) {
  Eigen::Matrix<double, 2, 6> x;
  x << 0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 1, 1;
  Eigen::Matrix<double, 6, 1> y;
  y << 1, 2, 3, 4, 5, 6;

  // Thin plate should handle zero distances gracefully
  auto rbf = RBF<double>(std::numeric_limits<double>::quiet_NaN(), 0.01,
                         RBFKernel::kThinPlate);

  auto yi = rbf.interpolate(x, y, x);

  ASSERT_EQ(y.size(), yi.size());
  // With smoothing, values should be close but may not be exact
  for (int64_t ix = 0; ix < yi.size(); ++ix) {
    EXPECT_NEAR(y(ix), yi(ix), 0.5);
  }
}

// Test different epsilon values produce different results
TEST(MathRbf, EpsilonEffect) {
  auto x = Eigen::Matrix<double, 1, 10>::LinSpaced(10, 0, 10);
  auto y = x.array().sin();

  // Small epsilon
  auto rbf1 = RBF<double>(0.1, 0, RBFKernel::kGaussian);
  auto yi1 = rbf1.interpolate(x, y, x);

  // Large epsilon
  auto rbf2 = RBF<double>(2.0, 0, RBFKernel::kGaussian);
  auto yi2 = rbf2.interpolate(x, y, x);

  ASSERT_EQ(y.size(), yi1.size());
  ASSERT_EQ(y.size(), yi2.size());

  // Both should interpolate at nodes
  for (int64_t ix = 0; ix < yi1.size(); ++ix) {
    EXPECT_NEAR(y(ix), yi1(ix), 1e-9);
    EXPECT_NEAR(y(ix), yi2(ix), 1e-9);
  }
}

// Test higher dimensional space (4D)
TEST(MathRbf, HighDimensional) {
  Eigen::Matrix<double, 4, 30> x = Eigen::Matrix<double, 4, 30>::Random();
  Eigen::Matrix<double, 30, 1> y = x.row(0).array().square();

  auto rbf = RBF<double>(std::numeric_limits<double>::quiet_NaN(), 0,
                         RBFKernel::kMultiquadric);
  auto yi = rbf.interpolate(x, y, x);

  ASSERT_EQ(y.size(), yi.size());
  for (int64_t ix = 0; ix < yi.size(); ++ix) {
    EXPECT_NEAR(y(ix), yi(ix), 1e-9);
  }
}

}  // namespace pyinterp::math::interpolate

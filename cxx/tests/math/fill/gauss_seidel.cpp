// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/fill/gauss_seidel.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace pyinterp::fill {

/// Test the detail::gauss_seidel_core function for homogeneous case (rhs=0)
TEST(GaussSeidelCoreTest, HomogeneousSimpleGrid) {
  // Simple 3x3 grid with interior NaN
  RowMajorMatrix<double> grid(3, 3);
  grid << 1.0, 2.0, 3.0,  //
      4.0, 5.0, 6.0,      //
      7.0, 8.0, 9.0;

  Matrix<bool> mask(3, 3);
  mask.setConstant(false);

  Matrix<double> rhs = Matrix<double>::Zero(3, 3);

  // Set center cell to be updated
  double test_value = 10.0;
  grid(1, 1) = test_value;
  mask(1, 1) = true;

  auto residual =
      detail::gauss_seidel_core<double>(grid, rhs, mask, false, 1.0, 1);

  // After one iteration at (1,1): avg of neighbors
  // (1,2) + (1,0) + (0,1) + (2,1) = 6 + 4 + 2 + 8 = 20, avg = 5
  // residual = 5 - 10 = -5
  double expected = 5.0;
  EXPECT_NEAR(grid(1, 1), expected, 1e-10);
  EXPECT_NEAR(residual, 5.0, 1e-10);
}

/// Test gauss_seidel_core with relaxation parameter
TEST(GaussSeidelCoreTest, WithRelaxation) {
  RowMajorMatrix<double> grid(3, 3);
  grid << 1.0, 2.0, 3.0,  //
      4.0, 5.0, 6.0,      //
      7.0, 8.0, 9.0;

  Matrix<bool> mask(3, 3);
  mask.setConstant(false);
  mask(1, 1) = true;

  Matrix<double> rhs = Matrix<double>::Zero(3, 3);

  double initial = 10.0;
  grid(1, 1) = initial;

  // With relaxation 0.5:
  // residual = 0.5 * (5 - 10) = -2.5
  // new value = 10 - 2.5 = 7.5
  auto residual =
      detail::gauss_seidel_core<double>(grid, rhs, mask, false, 0.5, 1);

  EXPECT_NEAR(grid(1, 1), 7.5, 1e-10);
  EXPECT_NEAR(residual, 2.5, 1e-10);
}

/// Test gauss_seidel_core with RHS
TEST(GaussSeidelCoreTest, WithRHS) {
  RowMajorMatrix<double> grid(3, 3);
  grid << 1.0, 2.0, 3.0,  //
      4.0, 0.0, 6.0,      //
      7.0, 8.0, 9.0;

  Matrix<bool> mask(3, 3);
  mask.setConstant(false);
  mask(1, 1) = true;

  Matrix<double> rhs(3, 3);
  rhs.setConstant(0.0);
  rhs(1, 1) = 4.0;

  auto residual =
      detail::gauss_seidel_core<double>(grid, rhs, mask, false, 1.0, 1);

  // new value = 0.25 * (neighbors + rhs)
  // = 0.25 * (2 + 6 + 4 + 8 + 4) = 0.25 * 24 = 6
  // residual = 6 - 0 = 6
  EXPECT_NEAR(grid(1, 1), 6.0, 1e-10);
  EXPECT_NEAR(residual, 6.0, 1e-10);
}

/// Test periodic boundary condition
TEST(GaussSeidelCoreTest, PeriodicBoundaryX) {
  RowMajorMatrix<double> grid(3, 3);
  grid << 1.0, 2.0, 3.0,  //
      4.0, 0.0, 6.0,      //
      7.0, 8.0, 9.0;

  Matrix<bool> mask(3, 3);
  mask.setConstant(false);
  mask(1, 1) = true;

  Matrix<double> rhs = Matrix<double>::Zero(3, 3);

  // With periodic: (1,1) neighbors are (0,1), (2,1), (1,0), (1,2) = 2, 8, 4, 6
  // avg = 5
  detail::gauss_seidel_core<double>(grid, rhs, mask, true, 1.0, 1);

  EXPECT_NEAR(grid(1, 1), 5.0, 1e-10);
}

/// Test detail::gauss_seidel wrapper for homogeneous case
TEST(GaussSeidelWrapperTest, Homogeneous) {
  RowMajorMatrix<double> grid(3, 3);
  grid << 1.0, 2.0, 3.0,  //
      4.0, 10.0, 6.0,     //
      7.0, 8.0, 9.0;

  Matrix<bool> mask(3, 3);
  mask.setConstant(false);
  mask(1, 1) = true;

  auto residual = detail::gauss_seidel<double>(grid, mask, false, 1.0, 1);

  double expected = 5.0;
  EXPECT_NEAR(grid(1, 1), expected, 1e-10);
  EXPECT_NEAR(residual, 5.0, 1e-10);
}

/// Test main gauss_seidel function with simple config
TEST(GaussSeidelTest, SimpleNoNaN) {
  RowMajorMatrix<double> grid(3, 3);
  grid << 1.0, 2.0, 3.0,  //
      4.0, 5.0, 6.0,      //
      7.0, 8.0, 9.0;

  config::fill::GaussSeidel config;

  auto [iterations, max_residual] = gauss_seidel<double>(grid, config);

  // No NaN, should return immediately
  EXPECT_EQ(iterations, 0);
  EXPECT_EQ(max_residual, 0.0);
}

/// Test main gauss_seidel function with NaN values
TEST(GaussSeidelTest, WithNaN) {
  RowMajorMatrix<double> grid(3, 3);
  const double nan = std::numeric_limits<double>::quiet_NaN();

  grid << 1.0, 2.0, 3.0,  //
      4.0, nan, 6.0,      //
      7.0, 8.0, 9.0;

  config::fill::GaussSeidel config;
  config = config.with_max_iterations(100).with_epsilon(1e-6);

  auto [iterations, max_residual] = gauss_seidel<double>(grid, config);

  // Should have run iterations and converged
  EXPECT_GT(iterations, 0);
  EXPECT_LT(iterations, 100);
  EXPECT_LT(max_residual, config.epsilon());

  // Center should be filled with interpolated value (approximately 5.0)
  EXPECT_FALSE(std::isnan(grid(1, 1)));
  EXPECT_NEAR(grid(1, 1), 5.0, 0.5);
}

/// Test with multiple NaN values
TEST(GaussSeidelTest, MultipleNaNValues) {
  RowMajorMatrix<double> grid(4, 4);
  const double nan = std::numeric_limits<double>::quiet_NaN();

  grid << 1.0, 2.0, 3.0, 4.0,  //
      5.0, nan, nan, 8.0,      //
      9.0, nan, nan, 12.0,     //
      13.0, 14.0, 15.0, 16.0;

  config::fill::GaussSeidel config;
  config = config.with_max_iterations(200).with_epsilon(1e-6);

  auto [iterations, max_residual] = gauss_seidel<double>(grid, config);

  ASSERT_GT(iterations, 0);
  ASSERT_LT(max_residual, config.epsilon());

  // All values should be filled
  for (int i = 0; i < grid.rows(); ++i) {
    for (int j = 0; j < grid.cols(); ++j) {
      EXPECT_FALSE(std::isnan(grid(i, j)))
          << "NaN at (" << i << "," << j << ")";
    }
  }
}

/// Test convergence with zero first guess
TEST(GaussSeidelTest, ZeroFirstGuess) {
  RowMajorMatrix<double> grid(3, 3);
  const double nan = std::numeric_limits<double>::quiet_NaN();

  grid << 1.0, 2.0, 3.0,  //
      4.0, nan, 6.0,      //
      7.0, 8.0, 9.0;

  config::fill::GaussSeidel config;
  config = config.with_first_guess(config::fill::FirstGuess::kZero)
               .with_max_iterations(100)
               .with_epsilon(1e-6);

  auto [iterations, max_residual] = gauss_seidel<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_LT(max_residual, config.epsilon());
  EXPECT_FALSE(std::isnan(grid(1, 1)));
}

/// Test with different relaxation parameters
TEST(GaussSeidelTest, DifferentRelaxation) {
  // Two grids with same NaN pattern, but different relaxation
  RowMajorMatrix<double> grid1(3, 3);
  RowMajorMatrix<double> grid2(3, 3);
  const double nan = std::numeric_limits<double>::quiet_NaN();

  grid1 << 1.0, 2.0, 3.0, 4.0, nan, 6.0, 7.0, 8.0, 9.0;
  grid2 << 1.0, 2.0, 3.0, 4.0, nan, 6.0, 7.0, 8.0, 9.0;

  // Standard relaxation (1.0)
  config::fill::GaussSeidel config1;
  config1 =
      config1.with_relaxation(1.0).with_max_iterations(100).with_epsilon(1e-8);

  // Over-relaxation (1.5)
  config::fill::GaussSeidel config2;
  config2 =
      config2.with_relaxation(1.5).with_max_iterations(100).with_epsilon(1e-8);

  auto [iter1, res1] = gauss_seidel<double>(grid1, config1);
  auto [iter2, res2] = gauss_seidel<double>(grid2, config2);

  ASSERT_GT(iter1, 0);
  ASSERT_GT(iter2, 0);
  ASSERT_LT(res1, config1.epsilon());
  ASSERT_LT(res2, config2.epsilon());

  // Both should converge to approximately the same value
  EXPECT_NEAR(grid1(1, 1), grid2(1, 1), 0.1);
}

/// Test periodic boundary with cyclic grid
TEST(GaussSeidelTest, PeriodicGrid) {
  RowMajorMatrix<double> grid(3, 3);
  const double nan = std::numeric_limits<double>::quiet_NaN();

  grid << 1.0, nan, 1.0,  //
      nan, nan, nan,      //
      1.0, nan, 1.0;

  config::fill::GaussSeidel config;
  config =
      config.with_is_periodic(true).with_max_iterations(200).with_epsilon(1e-6);

  auto [iterations, max_residual] = gauss_seidel<double>(grid, config);

  ASSERT_GT(iterations, 0);
  ASSERT_LT(max_residual, config.epsilon());

  // All values filled
  for (int i = 0; i < grid.rows(); ++i) {
    for (int j = 0; j < grid.cols(); ++j) {
      EXPECT_FALSE(std::isnan(grid(i, j)));
    }
  }
}

/// Test with edge NaN values
TEST(GaussSeidelTest, EdgeNaN) {
  RowMajorMatrix<double> grid(3, 3);
  const double nan = std::numeric_limits<double>::quiet_NaN();

  grid << nan, 2.0, nan,  //
      4.0, 5.0, 6.0,      //
      nan, 8.0, nan;

  config::fill::GaussSeidel config;
  config = config.with_max_iterations(100).with_epsilon(1e-6);

  auto [iterations, max_residual] = gauss_seidel<double>(grid, config);

  ASSERT_GT(iterations, 0);
  ASSERT_LT(max_residual, config.epsilon());

  // Edge values should be filled
  EXPECT_FALSE(std::isnan(grid(0, 0)));
  EXPECT_FALSE(std::isnan(grid(0, 2)));
  EXPECT_FALSE(std::isnan(grid(2, 0)));
  EXPECT_FALSE(std::isnan(grid(2, 2)));
}

/// Test early convergence when epsilon is reached
TEST(GaussSeidelTest, EarlyConvergence) {
  RowMajorMatrix<double> grid(3, 3);
  const double nan = std::numeric_limits<double>::quiet_NaN();

  grid << 1.0, 2.0, 3.0,  //
      4.0, nan, 6.0,      //
      7.0, 8.0, 9.0;

  config::fill::GaussSeidel config;
  config = config.with_max_iterations(1000).with_epsilon(1e-4);

  auto [iterations, max_residual] = gauss_seidel<double>(grid, config);

  // Should converge in fewer iterations than max
  EXPECT_LT(iterations, 1000);
  EXPECT_LT(max_residual, config.epsilon());
}

}  // namespace pyinterp::fill

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/fill/multi_grid.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace pyinterp::fill {

// Test compute_residual
TEST(MultiGridDetailTest, ComputeResidual) {
  // 3x3 grid with simple values
  RowMajorMatrix<double> grid(3, 3);
  grid << 1.0, 2.0, 3.0,  //
      4.0, 5.0, 6.0,      //
      7.0, 8.0, 9.0;

  Matrix<double> rhs = Matrix<double>::Zero(3, 3);
  Matrix<bool> mask(3, 3);
  mask.setConstant(true);

  Matrix<double> residual(3, 3);

  detail::compute_residual<double>(residual, grid, rhs, mask, false);

  // For center point (1,1): Au = 5 - 0.25*(4+6+2+8) = 5 - 5 = 0
  // r = f - Au = 0 - 0 = 0
  EXPECT_DOUBLE_EQ(residual(1, 1), 0.0);
}

TEST(MultiGridDetailTest, ComputeResidualWithRHS) {
  RowMajorMatrix<double> grid(3, 3);
  grid << 1.0, 2.0, 3.0,  //
      4.0, 5.0, 6.0,      //
      7.0, 8.0, 9.0;

  Matrix<double> rhs(3, 3);
  rhs.setConstant(2.0);

  Matrix<bool> mask(3, 3);
  mask.setConstant(true);

  Matrix<double> residual(3, 3);

  detail::compute_residual<double>(residual, grid, rhs, mask, false);

  // For center: Au = 5 - 0.25*(4+6+2+8) = 0
  // r = f - Au = 2 - 0 = 2
  EXPECT_DOUBLE_EQ(residual(1, 1), 2.0);
}

TEST(MultiGridDetailTest, ComputeResidualMasked) {
  RowMajorMatrix<double> grid(3, 3);
  grid << 1.0, 2.0, 3.0,  //
      4.0, 5.0, 6.0,      //
      7.0, 8.0, 9.0;

  Matrix<double> rhs = Matrix<double>::Zero(3, 3);
  Matrix<bool> mask(3, 3);
  mask.setConstant(false);
  mask(1, 1) = true;  // Only center masked

  Matrix<double> residual(3, 3);

  detail::compute_residual<double>(residual, grid, rhs, mask, false);

  // Only center should have residual computed
  EXPECT_DOUBLE_EQ(residual(1, 1), 0.0);
  // Others should be zero (not masked)
  EXPECT_DOUBLE_EQ(residual(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(residual(2, 2), 0.0);
}

// Test restrict_grid
TEST(MultiGridDetailTest, RestrictGrid) {
  RowMajorMatrix<double> fine(4, 4);
  fine << 1.0, 2.0, 3.0, 4.0,  //
      5.0, 6.0, 7.0, 8.0,      //
      9.0, 10.0, 11.0, 12.0,   //
      13.0, 14.0, 15.0, 16.0;

  Matrix<double> coarse(2, 2);

  detail::restrict_grid<double>(coarse, fine, false);

  // coarse(0,0) = 0.25 * (1+2+5+6) = 3.5
  EXPECT_DOUBLE_EQ(coarse(0, 0), 3.5);

  // coarse(0,1) = 0.25 * (3+4+7+8) = 5.5
  EXPECT_DOUBLE_EQ(coarse(0, 1), 5.5);

  // coarse(1,0) = 0.25 * (9+10+13+14) = 11.5
  EXPECT_DOUBLE_EQ(coarse(1, 0), 11.5);

  // coarse(1,1) = 0.25 * (11+12+15+16) = 13.5
  EXPECT_DOUBLE_EQ(coarse(1, 1), 13.5);
}

TEST(MultiGridDetailTest, RestrictGridOddSize) {
  RowMajorMatrix<double> fine(3, 3);
  fine << 1.0, 2.0, 3.0,  //
      4.0, 5.0, 6.0,      //
      7.0, 8.0, 9.0;

  Matrix<double> coarse(2, 2);

  detail::restrict_grid<double>(coarse, fine, false);

  // With reflective boundary, indices are adjusted
  // coarse(0,0) = 0.25 * (fine[0,0] + fine[1,0] + fine[0,1] + fine[1,1])
  EXPECT_DOUBLE_EQ(coarse(0, 0), 0.25 * (1.0 + 4.0 + 2.0 + 5.0));
}

// Test restrict_mask
TEST(MultiGridDetailTest, RestrictMask) {
  Matrix<bool> fine(4, 4);
  fine.setConstant(false);
  fine(0, 0) = true;
  fine(1, 1) = true;

  Matrix<bool> coarse(2, 2);

  detail::restrict_mask(coarse, fine, false);

  // coarse(0,0) should be true (includes fine(0,0) and fine(1,1))
  EXPECT_TRUE(coarse(0, 0));

  // Other coarse cells should be false
  EXPECT_FALSE(coarse(0, 1));
  EXPECT_FALSE(coarse(1, 0));
  EXPECT_FALSE(coarse(1, 1));
}

// Test prolong_grid
TEST(MultiGridDetailTest, ProlongGrid) {
  RowMajorMatrix<double> coarse(2, 2);
  coarse << 1.0, 2.0,  //
      3.0, 4.0;

  Matrix<double> fine = Matrix<double>::Zero(4, 4);

  detail::prolong_grid<double>(fine, coarse, false);

  // Check direct injection at coarse points
  EXPECT_DOUBLE_EQ(fine(0, 0), 1.0);

  // Check interpolated values
  // fine(1,0) = 0.5 * (coarse(0,0) + coarse(1,0)) = 0.5 * (1+3) = 2.0
  EXPECT_DOUBLE_EQ(fine(1, 0), 2.0);

  // fine(0,1) = 0.5 * (coarse(0,0) + coarse(0,1)) = 0.5 * (1+2) = 1.5
  EXPECT_DOUBLE_EQ(fine(0, 1), 1.5);
}

// Test smooth function
TEST(MultiGridDetailTest, Smooth) {
  RowMajorMatrix<double> grid(3, 3);
  grid << 1.0, 2.0, 3.0,  //
      4.0, 10.0, 6.0,     //
      7.0, 8.0, 9.0;

  Matrix<double> rhs = Matrix<double>::Zero(3, 3);
  Matrix<bool> mask(3, 3);
  mask.setConstant(false);
  mask(1, 1) = true;

  detail::smooth<double>(grid, rhs, mask, false, 1.0, 1, 1);

  // Center should move toward average of neighbors
  // avg = 0.25 * (4+6+2+8) = 5
  EXPECT_NEAR(grid(1, 1), 5.0, 0.1);
}

// Test v_cycle base case
TEST(MultiGridDetailTest, VCycleBaseCase) {
  RowMajorMatrix<double> grid(3, 3);
  grid.setZero();

  Matrix<double> rhs = Matrix<double>::Zero(3, 3);
  Matrix<bool> mask(3, 3);
  mask.setConstant(true);

  // Small grid should trigger base case
  detail::v_cycle<double>(grid, rhs, mask, false, 1.0, 2, 2, 10, 1);

  // Grid should remain near zero (solving zero RHS)
  for (int i = 0; i < grid.rows(); ++i) {
    for (int j = 0; j < grid.cols(); ++j) {
      EXPECT_NEAR(grid(i, j), 0.0, 1e-3);
    }
  }
}

// Test multigrid with no NaN
TEST(MultiGridTest, NoNaN) {
  RowMajorMatrix<double> grid(5, 5);
  grid.setConstant(1.0);

  config::fill::Multigrid config;

  EigenDRef<RowMajorMatrix<double>> grid_ref(grid);
  auto [iterations, max_residual] = multigrid<double>(grid_ref, config);

  EXPECT_EQ(iterations, 0);
  EXPECT_DOUBLE_EQ(max_residual, 0.0);
}

// Test multigrid with single NaN
TEST(MultiGridTest, SingleNaN) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(5, 5);
  grid.setConstant(1.0);
  grid(2, 2) = nan;

  config::fill::Multigrid config;
  config = config.with_max_iterations(20).with_epsilon(1e-4);

  auto [iterations, max_residual] = multigrid<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_FALSE(std::isnan(grid(2, 2)));
  // Should be close to 1.0 (average of surroundings)
  EXPECT_NEAR(grid(2, 2), 1.0, 0.1);
}

// Test multigrid with multiple NaN
TEST(MultiGridTest, MultipleNaN) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(7, 7);

  grid << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  //
      1.0, nan, nan, nan, nan, nan, 1.0,      //
      1.0, nan, nan, nan, nan, nan, 1.0,      //
      1.0, nan, nan, nan, nan, nan, 1.0,      //
      1.0, nan, nan, nan, nan, nan, 1.0,      //
      1.0, nan, nan, nan, nan, nan, 1.0,      //
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

  config::fill::Multigrid config;
  config = config.with_max_iterations(50).with_epsilon(1e-6);

  auto [iterations, max_residual] = multigrid<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_LT(max_residual, config.epsilon());

  // All NaN should be filled
  for (int i = 0; i < grid.rows(); ++i) {
    for (int j = 0; j < grid.cols(); ++j) {
      EXPECT_FALSE(std::isnan(grid(i, j)))
          << "NaN at (" << i << "," << j << ")";
    }
  }

  // Interior values should be close to 1.0
  EXPECT_NEAR(grid(3, 3), 1.0, 0.2);
}

// Test with zero first guess
TEST(MultiGridTest, ZeroFirstGuess) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(5, 5);
  grid.setConstant(2.0);
  grid(2, 2) = nan;

  config::fill::Multigrid config;
  config = config.with_first_guess(config::fill::FirstGuess::kZero)
               .with_max_iterations(20)
               .with_epsilon(1e-4);

  auto [iterations, max_residual] = multigrid<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_LT(max_residual, config.epsilon());
  EXPECT_FALSE(std::isnan(grid(2, 2)));
  EXPECT_NEAR(grid(2, 2), 2.0, 0.2);
}

// Test with periodic boundary
TEST(MultiGridTest, PeriodicBoundary) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(5, 5);
  grid.setConstant(1.0);

  // Set NaN at edges
  grid(0, 2) = nan;
  grid(4, 2) = nan;

  config::fill::Multigrid config;
  config =
      config.with_is_periodic(true).with_max_iterations(30).with_epsilon(1e-5);

  auto [iterations, max_residual] = multigrid<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_LT(max_residual, config.epsilon());

  // Both edge NaN should be filled
  EXPECT_FALSE(std::isnan(grid(0, 2)));
  EXPECT_FALSE(std::isnan(grid(4, 2)));
}

// Test different pre/post smoothing
TEST(MultiGridTest, DifferentSmoothing) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(6, 6);
  grid.setConstant(5.0);
  grid(3, 3) = nan;

  config::fill::Multigrid config;
  config = config.with_pre_smooth(3)
               .with_post_smooth(3)
               .with_max_iterations(15)
               .with_epsilon(1e-4);

  auto [iterations, max_residual] = multigrid<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_LT(max_residual, config.epsilon());

  EXPECT_FALSE(std::isnan(grid(3, 3)));
  EXPECT_NEAR(grid(3, 3), 5.0, 0.2);
}

// Test convergence behavior with different epsilon values
TEST(MultiGridTest, ConvergenceEpsilon) {
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // Create a more complex scenario with multiple NaN values
  // that requires multiple iterations to converge
  RowMajorMatrix<double> grid(8, 8);
  grid.setConstant(3.0);

  // Add multiple NaN values in a pattern
  for (int i = 2; i < 6; ++i) {
    for (int j = 2; j < 6; ++j) {
      grid(i, j) = nan;
    }
  }

  // Strict epsilon
  config::fill::Multigrid config1;
  config1 = config1.with_max_iterations(100).with_epsilon(1e-12);
  auto [iter1, res1] = multigrid<double>(grid, config1);

  // Create fresh grid for second test
  RowMajorMatrix<double> grid2(8, 8);
  grid2.setConstant(3.0);
  for (int i = 2; i < 6; ++i) {
    for (int j = 2; j < 6; ++j) {
      grid2(i, j) = nan;
    }
  }

  // Loose epsilon
  config::fill::Multigrid config2;
  config2 = config2.with_max_iterations(100).with_epsilon(1e-3);
  auto [iter2, res2] = multigrid<double>(grid2, config2);

  // Both should converge and meet their respective tolerances
  EXPECT_GT(iter1, 0);
  EXPECT_GT(iter2, 0);
  EXPECT_LE(res1, config1.epsilon());
  EXPECT_LE(res2, config2.epsilon());

  // All NaN values should be filled in both grids
  for (int i = 0; i < grid.rows(); ++i) {
    for (int j = 0; j < grid.cols(); ++j) {
      EXPECT_FALSE(std::isnan(grid(i, j)))
          << "grid1 NaN at (" << i << "," << j << ")";
      EXPECT_FALSE(std::isnan(grid2(i, j)))
          << "grid2 NaN at (" << i << "," << j << ")";
    }
  }

  // Both should converge to approximately the same solution (3.0)
  EXPECT_NEAR(grid(3, 3), 3.0, 0.2);
  EXPECT_NEAR(grid2(3, 3), 3.0, 0.2);
}

// Test large grid
TEST(MultiGridTest, LargeGrid) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(20, 20);
  grid.setConstant(1.0);

  // Create a cross pattern of NaN
  for (int i = 5; i < 15; ++i) {
    grid(i, 10) = nan;
    grid(10, i) = nan;
  }

  config::fill::Multigrid config;
  config = config.with_max_iterations(30).with_epsilon(1e-5);

  auto [iterations, max_residual] = multigrid<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_LT(max_residual, config.epsilon());

  // All values should be filled
  for (int i = 0; i < grid.rows(); ++i) {
    for (int j = 0; j < grid.cols(); ++j) {
      EXPECT_FALSE(std::isnan(grid(i, j)));
    }
  }
}

// Test edge case: small grid
TEST(MultiGridTest, SmallGrid) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(3, 3);
  grid << 1.0, 2.0, 3.0,  //
      4.0, nan, 6.0,      //
      7.0, 8.0, 9.0;

  config::fill::Multigrid config;
  config = config.with_max_iterations(20).with_epsilon(1e-4);

  auto [iterations, max_residual] = multigrid<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_LT(max_residual, config.epsilon());
  EXPECT_FALSE(std::isnan(grid(1, 1)));
  EXPECT_NEAR(grid(1, 1), 5.0, 0.5);
}

// Test multi-threading
TEST(MultiGridTest, MultiThreaded) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(10, 10);
  grid.setConstant(2.0);

  for (int i = 3; i < 7; ++i) {
    for (int j = 3; j < 7; ++j) {
      grid(i, j) = nan;
    }
  }

  config::fill::Multigrid config;
  config =
      config.with_max_iterations(30).with_epsilon(1e-5).with_num_threads(4);

  auto [iterations, max_residual] = multigrid<double>(grid, config);
  EXPECT_GT(iterations, 0);
  EXPECT_LT(max_residual, config.epsilon());

  // All NaN should be filled
  for (int i = 0; i < grid.rows(); ++i) {
    for (int j = 0; j < grid.cols(); ++j) {
      EXPECT_FALSE(std::isnan(grid(i, j)));
    }
  }
}

}  // namespace pyinterp::fill

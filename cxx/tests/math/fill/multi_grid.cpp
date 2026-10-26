// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/fill/multi_grid.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>

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
  // A constant field is preserved everywhere, borders included.
  RowMajorMatrix<double> constant = RowMajorMatrix<double>::Constant(5, 5, 7.0);
  Matrix<double> coarse(3, 3);
  detail::restrict_grid<double>(coarse, constant, false);
  EXPECT_TRUE(coarse.isApproxToConstant(7.0));

  // A linear field is preserved at interior points.
  RowMajorMatrix<double> linear(5, 5);
  for (int64_t ix = 0; ix < 5; ++ix) {
    for (int64_t iy = 0; iy < 5; ++iy) {
      linear(ix, iy) = 10.0 * static_cast<double>(ix) + static_cast<double>(iy);
    }
  }
  detail::restrict_grid<double>(coarse, linear, false);
  EXPECT_DOUBLE_EQ(coarse(1, 1), 22.0);
}

TEST(MultiGridDetailTest, RestrictGridWeights) {
  // A spike on a fine point spreads with the full-weighting stencil: 4/16 to
  // the coarse point it coincides with, 2/16 to the coarse points it lies
  // halfway between.
  RowMajorMatrix<double> fine = RowMajorMatrix<double>::Zero(7, 7);
  Matrix<double> coarse(4, 4);

  fine(2, 2) = 16.0;
  detail::restrict_grid<double>(coarse, fine, false);
  EXPECT_DOUBLE_EQ(coarse(1, 1), 4.0);
  EXPECT_DOUBLE_EQ(coarse(0, 1), 0.0);

  fine.setZero();
  fine(3, 2) = 16.0;
  detail::restrict_grid<double>(coarse, fine, false);
  EXPECT_DOUBLE_EQ(coarse(1, 1), 2.0);
  EXPECT_DOUBLE_EQ(coarse(2, 1), 2.0);
  EXPECT_DOUBLE_EQ(coarse(1, 0), 0.0);
}

// Test restrict_mask
TEST(MultiGridDetailTest, RestrictMask) {
  Matrix<bool> fine(5, 5);
  fine.setConstant(false);
  fine(1, 1) = true;  // Between coarse points: not injected
  fine(2, 2) = true;  // Coincides with coarse (1, 1)
  fine(4, 0) = true;  // Coincides with coarse (2, 0)

  Matrix<bool> coarse(3, 3);

  detail::restrict_mask(coarse, fine);

  EXPECT_FALSE(coarse(0, 0));
  EXPECT_TRUE(coarse(1, 1));
  EXPECT_TRUE(coarse(2, 0));
  EXPECT_EQ(coarse.count(), 2);
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

// The number of V-cycles must not grow with the grid size: a few percent of
// scattered gaps plus a large hole must converge in a handful of cycles.
TEST(MultiGridTest, ConvergenceIndependentOfSize) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const int64_t nx : {128, 256, 512}) {
    const int64_t ny = nx / 2;
    RowMajorMatrix<double> grid(nx, ny);
    auto gen = std::mt19937(42);
    auto uniform = std::uniform_real_distribution<>(0.0, 1.0);
    for (int64_t ix = 0; ix < nx; ++ix) {
      for (int64_t iy = 0; iy < ny; ++iy) {
        grid(ix, iy) = uniform(gen) < 0.1 ? nan
                                          : std::sin(2.0 * std::numbers::pi *
                                                     static_cast<double>(ix) /
                                                     static_cast<double>(nx));
      }
    }
    grid.block(nx / 3, ny / 3, nx / 10, ny / 10).setConstant(nan);

    config::fill::Multigrid config;
    config =
        config.with_max_iterations(100).with_epsilon(1e-4).with_is_periodic(
            true);

    auto [iterations, max_residual] = multigrid<double>(grid, config);

    EXPECT_LT(max_residual, config.epsilon()) << "nx=" << nx;
    EXPECT_LE(iterations, 20) << "nx=" << nx;
    EXPECT_FALSE(grid.hasNaN()) << "nx=" << nx;
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

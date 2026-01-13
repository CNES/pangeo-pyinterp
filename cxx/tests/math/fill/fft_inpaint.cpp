// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/fill/fft_inpaint.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace pyinterp::fill {

// Test fft_inpaint with no NaN values
TEST(FFTInpaintTest, NoNaN) {
  RowMajorMatrix<double> grid(5, 5);
  grid.setConstant(1.0);

  config::fill::FFTInpaint config;

  auto [iterations, max_residual] = fft_inpaint<double>(grid, config);

  EXPECT_EQ(iterations, 0);
  EXPECT_DOUBLE_EQ(max_residual, 0.0);

  // Grid should remain unchanged
  for (int i = 0; i < grid.rows(); ++i) {
    for (int j = 0; j < grid.cols(); ++j) {
      EXPECT_DOUBLE_EQ(grid(i, j), 1.0);
    }
  }
}

// Test fft_inpaint with single NaN value (non-periodic)
TEST(FFTInpaintTest, SingleNaNNonPeriodic) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(5, 5);
  grid.setConstant(1.0);
  grid(2, 2) = nan;

  config::fill::FFTInpaint config;
  config = config.with_is_periodic(false).with_max_iterations(100).with_epsilon(
      1e-6);

  auto [iterations, max_residual] = fft_inpaint<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_LT(iterations, 100);
  EXPECT_LT(max_residual, config.epsilon());

  // Center should be filled close to 1.0
  EXPECT_FALSE(std::isnan(grid(2, 2)));
  EXPECT_NEAR(grid(2, 2), 1.0, 0.1);
}

// Test fft_inpaint with single NaN value (periodic)
TEST(FFTInpaintTest, SingleNaNPeriodic) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(5, 5);
  grid.setConstant(2.0);
  grid(2, 2) = nan;

  config::fill::FFTInpaint config;
  config =
      config.with_is_periodic(true).with_max_iterations(100).with_epsilon(1e-6);

  auto [iterations, max_residual] = fft_inpaint<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_FALSE(std::isnan(grid(2, 2)));
  EXPECT_NEAR(grid(2, 2), 2.0, 0.1);
}

// Test fft_inpaint with multiple NaN values
TEST(FFTInpaintTest, MultipleNaN) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(7, 7);

  grid << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  //
      1.0, nan, nan, nan, nan, nan, 1.0,      //
      1.0, nan, nan, nan, nan, nan, 1.0,      //
      1.0, nan, nan, nan, nan, nan, 1.0,      //
      1.0, nan, nan, nan, nan, nan, 1.0,      //
      1.0, nan, nan, nan, nan, nan, 1.0,      //
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

  config::fill::FFTInpaint config;
  config = config.with_is_periodic(false).with_max_iterations(200).with_epsilon(
      1e-6);

  auto [iterations, max_residual] = fft_inpaint<double>(grid, config);

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
TEST(FFTInpaintTest, ZeroFirstGuess) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(5, 5);
  grid.setConstant(3.0);
  grid(2, 2) = nan;

  config::fill::FFTInpaint config;
  config = config.with_first_guess(config::fill::FirstGuess::kZero)
               .with_is_periodic(false)
               .with_max_iterations(500)
               .with_epsilon(1e-6);

  auto [iterations, max_residual] = fft_inpaint<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_LT(max_residual, config.epsilon());
  EXPECT_FALSE(std::isnan(grid(2, 2)));
  EXPECT_NEAR(grid(2, 2), 3.0, 0.2);
}

// Test with zonal average first guess
TEST(FFTInpaintTest, ZonalAverageFirstGuess) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(5, 5);
  grid.setConstant(4.0);
  grid(2, 2) = nan;

  config::fill::FFTInpaint config;
  config = config.with_first_guess(config::fill::FirstGuess::kZonalAverage)
               .with_is_periodic(false)
               .with_max_iterations(100)
               .with_epsilon(1e-6);

  auto [iterations, max_residual] = fft_inpaint<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_LT(max_residual, config.epsilon());
  EXPECT_FALSE(std::isnan(grid(2, 2)));
  EXPECT_NEAR(grid(2, 2), 4.0, 0.2);
}

// Test different sigma values
TEST(FFTInpaintTest, DifferentSigma) {
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // Small sigma
  RowMajorMatrix<double> grid1(6, 6);
  grid1.setConstant(1.0);
  grid1(3, 3) = nan;

  config::fill::FFTInpaint config1;
  config1 = config1.with_sigma(5.0)
                .with_is_periodic(false)
                .with_max_iterations(100)
                .with_epsilon(1e-6);

  auto [iter1, res1] = fft_inpaint<double>(grid1, config1);
  EXPECT_GT(iter1, 0);
  EXPECT_LT(res1, config1.epsilon());
  EXPECT_FALSE(std::isnan(grid1(3, 3)));

  // Large sigma
  RowMajorMatrix<double> grid2(6, 6);
  grid2.setConstant(1.0);
  grid2(3, 3) = nan;

  config::fill::FFTInpaint config2;
  config2 = config2.with_sigma(20.0)
                .with_is_periodic(false)
                .with_max_iterations(100)
                .with_epsilon(1e-6);

  auto [iter2, res2] = fft_inpaint<double>(grid2, config2);
  EXPECT_GT(iter2, 0);
  EXPECT_LT(res2, config2.epsilon());
  EXPECT_FALSE(std::isnan(grid2(3, 3)));

  // Both should converge to approximately 1.0
  EXPECT_NEAR(grid1(3, 3), 1.0, 0.1);
  EXPECT_NEAR(grid2(3, 3), 1.0, 0.1);
}

// Test convergence behavior with different epsilon values
TEST(FFTInpaintTest, ConvergenceEpsilon) {
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // Create a more complex scenario with multiple NaN values
  RowMajorMatrix<double> grid1(8, 8);
  grid1.setConstant(2.0);
  for (int i = 2; i < 6; ++i) {
    for (int j = 2; j < 6; ++j) {
      grid1(i, j) = nan;
    }
  }

  // Strict epsilon
  config::fill::FFTInpaint config1;
  config1 =
      config1.with_is_periodic(false).with_max_iterations(200).with_epsilon(
          1e-8);

  auto [iter1, res1] = fft_inpaint<double>(grid1, config1);

  // Create fresh grid for second test
  RowMajorMatrix<double> grid2(8, 8);
  grid2.setConstant(2.0);
  for (int i = 2; i < 6; ++i) {
    for (int j = 2; j < 6; ++j) {
      grid2(i, j) = nan;
    }
  }

  // Loose epsilon
  config::fill::FFTInpaint config2;
  config2 =
      config2.with_is_periodic(false).with_max_iterations(200).with_epsilon(
          1e-3);

  auto [iter2, res2] = fft_inpaint<double>(grid2, config2);

  // Both should converge and meet their respective tolerances
  EXPECT_GT(iter1, 0);
  EXPECT_GT(iter2, 0);
  EXPECT_LE(res1, config1.epsilon());
  EXPECT_LE(res2, config2.epsilon());

  // All NaN values should be filled in both grids
  for (int i = 0; i < grid1.rows(); ++i) {
    for (int j = 0; j < grid1.cols(); ++j) {
      EXPECT_FALSE(std::isnan(grid1(i, j)))
          << "grid1 NaN at (" << i << "," << j << ")";
      EXPECT_FALSE(std::isnan(grid2(i, j)))
          << "grid2 NaN at (" << i << "," << j << ")";
    }
  }

  // Both should converge to approximately 2.0
  EXPECT_NEAR(grid1(3, 3), 2.0, 0.2);
  EXPECT_NEAR(grid2(3, 3), 2.0, 0.2);
}

// Test periodic vs non-periodic
TEST(FFTInpaintTest, PeriodicVsNonPeriodic) {
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // Non-periodic
  RowMajorMatrix<double> grid1(8, 8);
  grid1.setConstant(1.0);
  grid1(4, 4) = nan;

  config::fill::FFTInpaint config1;
  config1 =
      config1.with_is_periodic(false).with_max_iterations(100).with_epsilon(
          1e-6);

  auto [iter1, res1] = fft_inpaint<double>(grid1, config1);
  EXPECT_GT(iter1, 0);
  EXPECT_LT(res1, config1.epsilon());
  EXPECT_FALSE(std::isnan(grid1(4, 4)));

  // Periodic
  RowMajorMatrix<double> grid2(8, 8);
  grid2.setConstant(1.0);
  grid2(4, 4) = nan;

  config::fill::FFTInpaint config2;
  config2 =
      config2.with_is_periodic(true).with_max_iterations(100).with_epsilon(
          1e-6);

  auto [iter2, res2] = fft_inpaint<double>(grid2, config2);
  EXPECT_GT(iter2, 0);
  EXPECT_LT(res2, config2.epsilon());
  EXPECT_FALSE(std::isnan(grid2(4, 4)));

  // Both should fill to approximately 1.0
  EXPECT_NEAR(grid1(4, 4), 1.0, 0.1);
  EXPECT_NEAR(grid2(4, 4), 1.0, 0.1);
}

// Test edge NaN values
TEST(FFTInpaintTest, EdgeNaN) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(5, 5);
  grid.setConstant(1.0);

  // Set edge values to NaN
  grid(0, 0) = nan;
  grid(0, 4) = nan;
  grid(4, 0) = nan;
  grid(4, 4) = nan;

  config::fill::FFTInpaint config;
  config = config.with_is_periodic(false).with_max_iterations(100).with_epsilon(
      1e-6);

  auto [iterations, max_residual] = fft_inpaint<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_LT(max_residual, config.epsilon());

  // All edge NaN should be filled
  EXPECT_FALSE(std::isnan(grid(0, 0)));
  EXPECT_FALSE(std::isnan(grid(0, 4)));
  EXPECT_FALSE(std::isnan(grid(4, 0)));
  EXPECT_FALSE(std::isnan(grid(4, 4)));
}

// Test large grid
TEST(FFTInpaintTest, LargeGrid) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(20, 20);
  grid.setConstant(5.0);

  // Create a cross pattern of NaN
  for (int i = 5; i < 15; ++i) {
    grid(i, 10) = nan;
    grid(10, i) = nan;
  }

  config::fill::FFTInpaint config;
  config = config.with_is_periodic(false).with_max_iterations(200).with_epsilon(
      1e-6);

  auto [iterations, max_residual] = fft_inpaint<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_LT(max_residual, config.epsilon());

  // All NaN should be filled
  for (int i = 0; i < grid.rows(); ++i) {
    for (int j = 0; j < grid.cols(); ++j) {
      EXPECT_FALSE(std::isnan(grid(i, j)));
    }
  }
}

// Test small grid
TEST(FFTInpaintTest, SmallGrid) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(3, 3);
  grid << 1.0, 2.0, 3.0,  //
      4.0, nan, 6.0,      //
      7.0, 8.0, 9.0;

  config::fill::FFTInpaint config;
  config = config.with_is_periodic(false).with_max_iterations(100).with_epsilon(
      1e-6);

  auto [iterations, max_residual] = fft_inpaint<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_LT(max_residual, config.epsilon());
  EXPECT_FALSE(std::isnan(grid(1, 1)));
  // Should be close to average of neighbors
  EXPECT_NEAR(grid(1, 1), 5.0, 1.0);
}

// Test multi-threading
TEST(FFTInpaintTest, MultiThreaded) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(16, 16);
  grid.setConstant(3.0);

  // Create a block of NaN
  for (int i = 6; i < 10; ++i) {
    for (int j = 6; j < 10; ++j) {
      grid(i, j) = nan;
    }
  }

  config::fill::FFTInpaint config;
  config = config.with_is_periodic(false)
               .with_max_iterations(150)
               .with_epsilon(1e-6)
               .with_num_threads(4);

  auto [iterations, max_residual] = fft_inpaint<double>(grid, config);

  EXPECT_GT(iterations, 0);
  EXPECT_LT(max_residual, config.epsilon());

  // All NaN should be filled
  for (int i = 0; i < grid.rows(); ++i) {
    for (int j = 0; j < grid.cols(); ++j) {
      EXPECT_FALSE(std::isnan(grid(i, j)));
    }
  }
}

// Test max iterations behavior
TEST(FFTInpaintTest, MaxIterationsReached) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> grid(8, 8);
  grid.setConstant(1.0);

  for (int i = 2; i < 6; ++i) {
    for (int j = 2; j < 6; ++j) {
      grid(i, j) = nan;
    }
  }

  config::fill::FFTInpaint config;
  config = config.with_is_periodic(false)
               .with_max_iterations(5)  // Limited iteration count
               .with_epsilon(1e-10);    // Very strict epsilon

  auto [iterations, max_residual] = fft_inpaint<double>(grid, config);

  // FFT inpaint is very efficient - may converge before max_iterations
  // The key is that it should not exceed max_iterations
  EXPECT_LE(iterations, 5);
  EXPECT_GT(iterations, 0);

  // Values should still be filled (not NaN)
  for (int i = 0; i < grid.rows(); ++i) {
    for (int j = 0; j < grid.cols(); ++j) {
      EXPECT_FALSE(std::isnan(grid(i, j)));
    }
  }

  // If we didn't converge, residual should be non-zero
  // If we did converge, residual should be below epsilon
  if (iterations == 5) {
    // Hit max iterations without converging
    EXPECT_GT(max_residual, 0.0);
  } else {
    // Converged before max iterations
    EXPECT_LE(max_residual, config.epsilon());
  }
}

// Test with float type
TEST(FFTInpaintTest, FloatType) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  RowMajorMatrix<float> grid(5, 5);
  grid.setConstant(2.0f);
  grid(2, 2) = nan;

  config::fill::FFTInpaint config;
  config = config.with_is_periodic(false).with_max_iterations(100).with_epsilon(
      1e-6);

  auto [iterations, max_residual] = fft_inpaint<float>(grid, config);

  EXPECT_GT(iterations, 0u);
  EXPECT_LT(max_residual, static_cast<float>(config.epsilon()));
  EXPECT_FALSE(std::isnan(grid(2, 2)));
  EXPECT_NEAR(grid(2, 2), 2.0f, 0.2f);
}

}  // namespace pyinterp::fill

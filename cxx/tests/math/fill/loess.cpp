// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/fill/loess.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace pyinterp::fill {

// Test tri-cube weight function
TEST(LoessDetailTest, TricubeWeight) {
  // At distance 0, weight should be 1
  EXPECT_DOUBLE_EQ(detail::tricube_weight(0.0), 1.0);

  // At distance 1, weight should be 0
  EXPECT_DOUBLE_EQ(detail::tricube_weight(1.0), 0.0);

  // Beyond distance 1, weight should be 0
  EXPECT_DOUBLE_EQ(detail::tricube_weight(1.5), 0.0);
  EXPECT_DOUBLE_EQ(detail::tricube_weight(2.0), 0.0);

  // At distance 0.5: (1 - 0.5^3)^3 = (1 - 0.125)^3 = 0.875^3 ≈ 0.669921875
  EXPECT_NEAR(detail::tricube_weight(0.5), 0.669921875, 1e-9);

  // Weight function should be continuous
  EXPECT_GT(detail::tricube_weight(0.99), 0.0);
  EXPECT_LT(detail::tricube_weight(0.99), detail::tricube_weight(0.5));
}

// Test should_process logic
TEST(LoessDetailTest, ShouldProcess) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double value = 5.0;

  // kAll: process everything
  EXPECT_TRUE(
      detail::should_process(value, config::fill::LoessValueType::kAll));
  EXPECT_TRUE(detail::should_process(nan, config::fill::LoessValueType::kAll));

  // kDefined: only process non-NaN values
  EXPECT_TRUE(
      detail::should_process(value, config::fill::LoessValueType::kDefined));
  EXPECT_FALSE(
      detail::should_process(nan, config::fill::LoessValueType::kDefined));

  // kUndefined: only process NaN values
  EXPECT_FALSE(
      detail::should_process(value, config::fill::LoessValueType::kUndefined));
  EXPECT_TRUE(
      detail::should_process(nan, config::fill::LoessValueType::kUndefined));
}

// Test compute_zonal_average
TEST(LoessDetailTest, ComputeZonalAverage) {
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // All valid values
  RowMajorMatrix<double> data1(2, 2);
  data1 << 1.0, 2.0, 3.0, 4.0;
  EXPECT_DOUBLE_EQ(detail::compute_zonal_average(data1), 2.5);

  // Mixed valid and NaN
  RowMajorMatrix<double> data2(2, 2);
  data2 << 1.0, nan, 3.0, 4.0;
  EXPECT_NEAR(detail::compute_zonal_average(data2), (1.0 + 3.0 + 4.0) / 3.0,
              1e-10);

  // All NaN
  RowMajorMatrix<double> data3(2, 2);
  data3 << nan, nan, nan, nan;
  EXPECT_DOUBLE_EQ(detail::compute_zonal_average(data3), 0.0);
}

// Test apply_first_guess
TEST(LoessDetailTest, ApplyFirstGuessZero) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> data(2, 2);
  data << 1.0, nan, 3.0, nan;

  detail::apply_first_guess(data, config::fill::FirstGuess::kZero);

  EXPECT_DOUBLE_EQ(data(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(data(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(data(1, 0), 3.0);
  EXPECT_DOUBLE_EQ(data(1, 1), 0.0);
}

TEST(LoessDetailTest, ApplyFirstGuessZonalAverage) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> data(2, 2);
  data << 2.0, nan, 4.0, nan;

  detail::apply_first_guess(data, config::fill::FirstGuess::kZonalAverage);

  double avg = (2.0 + 4.0) / 2.0;
  EXPECT_DOUBLE_EQ(data(0, 0), 2.0);
  EXPECT_DOUBLE_EQ(data(0, 1), avg);
  EXPECT_DOUBLE_EQ(data(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(data(1, 1), avg);
}

// Test compute_max_difference
TEST(LoessDetailTest, ComputeMaxDifference) {
  RowMajorMatrix<double> current(2, 2);
  current << 1.0, 2.0, 3.0, 4.0;

  RowMajorMatrix<double> previous(2, 2);
  previous << 1.5, 2.0, 3.0, 3.0;

  double max_diff = detail::compute_max_difference(current, previous);
  EXPECT_DOUBLE_EQ(max_diff, 1.0);  // max(|1.0-1.5|, |2.0-2.0|, |3.0-3.0|,
                                    // |4.0-3.0|)
}

// Test loess with simple uniform grid
TEST(LoessTest, UniformGrid) {
  RowMajorMatrix<double> data(3, 3);
  data << 1.0, 1.0, 1.0,  //
      1.0, 1.0, 1.0,      //
      1.0, 1.0, 1.0;

  config::fill::Loess config;
  config = config.with_nx(1).with_ny(1);

  auto result = loess<double>(data, config);

  // All values should remain 1.0
  for (int i = 0; i < result.rows(); ++i) {
    for (int j = 0; j < result.cols(); ++j) {
      EXPECT_DOUBLE_EQ(result(i, j), 1.0);
    }
  }
}

// Test loess fills NaN values
TEST(LoessTest, FillsNaN) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> data(3, 3);
  data << 1.0, 2.0, 3.0,  //
      4.0, nan, 6.0,      //
      7.0, 8.0, 9.0;

  config::fill::Loess config;
  config = config.with_nx(1).with_ny(1).with_max_iterations(10);

  auto result = loess<double>(data, config);

  // Center should be filled with weighted average of neighbors
  EXPECT_FALSE(std::isnan(result(1, 1)));
  // Should be close to 5.0 (average of neighbors)
  EXPECT_NEAR(result(1, 1), 5.0, 1.0);
}

// Test loess with multiple NaN values and iterations
TEST(LoessTest, MultipleNaNIterative) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> data(4, 4);
  data << 1.0, 2.0, 3.0, 4.0,  //
      5.0, nan, nan, 8.0,      //
      9.0, nan, nan, 12.0,     //
      13.0, 14.0, 15.0, 16.0;

  config::fill::Loess config;
  config =
      config.with_nx(2).with_ny(2).with_max_iterations(50).with_epsilon(1e-6);

  auto result = loess<double>(data, config);

  // All NaN values should be filled
  for (int i = 0; i < result.rows(); ++i) {
    for (int j = 0; j < result.cols(); ++j) {
      EXPECT_FALSE(std::isnan(result(i, j)))
          << "NaN at (" << i << "," << j << ")";
    }
  }
}

// Test value_type = kDefined (smooth only defined values)
TEST(LoessTest, SmoothDefinedOnly) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> data(3, 3);
  data << 1.0, 2.0, 3.0,  //
      4.0, nan, 6.0,      //
      7.0, 8.0, 9.0;

  config::fill::Loess config;
  config = config.with_nx(1)
               .with_ny(1)
               .with_value_type(config::fill::LoessValueType::kDefined)
               .with_max_iterations(1);

  auto result = loess<double>(data, config);

  // NaN should remain NaN with kDefined
  EXPECT_TRUE(std::isnan(result(1, 1)));
}

// Test value_type = kAll (smooth everything)
TEST(LoessTest, SmoothAll) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> data(3, 3);
  data << 1.0, 2.0, 3.0,  //
      4.0, 5.0, 6.0,      //
      7.0, 8.0, 9.0;

  // Add NaN in center
  data(1, 1) = nan;

  config::fill::Loess config;
  config = config.with_nx(1)
               .with_ny(1)
               .with_value_type(config::fill::LoessValueType::kAll)
               .with_max_iterations(1);

  auto result = loess<double>(data, config);

  // With kAll and single pass, NaN stays but defined values are smoothed
  // The actual behavior depends on loess implementation details
  // kAll with single iteration still processes all points
  for (int i = 0; i < result.rows(); ++i) {
    for (int j = 0; j < result.cols(); ++j) {
      if (i == 1 && j == 1) {
        // Center might still be NaN with single pass kAll
        continue;
      }
      EXPECT_FALSE(std::isnan(result(i, j)));
    }
  }
}

// Test periodic boundary (is_periodic=true)
TEST(LoessTest, PeriodicBoundary) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> data(3, 3);
  data << 1.0, nan, 1.0,  //
      nan, nan, nan,      //
      1.0, nan, 1.0;

  config::fill::Loess config;
  config = config.with_nx(1)
               .with_ny(1)
               .with_is_periodic(true)
               .with_max_iterations(20)
               .with_epsilon(1e-4);

  auto result = loess<double>(data, config);

  // All values should be filled
  for (int i = 0; i < result.rows(); ++i) {
    for (int j = 0; j < result.cols(); ++j) {
      EXPECT_FALSE(std::isnan(result(i, j)));
    }
  }
}

// Test different window sizes
TEST(LoessTest, DifferentWindowSizes) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> data(5, 5);
  data.setConstant(1.0);
  data(2, 2) = nan;

  // Small window
  config::fill::Loess config1;
  config1 = config1.with_nx(1).with_ny(1).with_max_iterations(10);
  auto result1 = loess<double>(data, config1);
  EXPECT_FALSE(std::isnan(result1(2, 2)));

  // Large window
  config::fill::Loess config2;
  config2 = config2.with_nx(2).with_ny(2).with_max_iterations(10);
  auto result2 = loess<double>(data, config2);
  EXPECT_FALSE(std::isnan(result2(2, 2)));

  // Both should fill with approximately 1.0
  EXPECT_NEAR(result1(2, 2), 1.0, 0.1);
  EXPECT_NEAR(result2(2, 2), 1.0, 0.1);
}

// Test convergence behavior
TEST(LoessTest, ConvergenceEpsilon) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> data(3, 3);
  data << 1.0, 2.0, 3.0,  //
      4.0, nan, 6.0,      //
      7.0, 8.0, 9.0;

  // Strict epsilon
  config::fill::Loess config1;
  config1 =
      config1.with_nx(1).with_ny(1).with_max_iterations(100).with_epsilon(1e-8);
  auto result1 = loess<double>(data, config1);

  // Loose epsilon
  config::fill::Loess config2;
  config2 =
      config2.with_nx(1).with_ny(1).with_max_iterations(100).with_epsilon(1e-2);
  auto result2 = loess<double>(data, config2);

  // Both should fill the NaN, values may differ slightly
  EXPECT_FALSE(std::isnan(result1(1, 1)));
  EXPECT_FALSE(std::isnan(result2(1, 1)));
}

// Test edge cases: single value
TEST(LoessTest, SingleValue) {
  RowMajorMatrix<double> data(1, 1);
  data << 5.0;

  config::fill::Loess config;
  config = config.with_nx(1).with_ny(1);

  auto result = loess<double>(data, config);
  EXPECT_DOUBLE_EQ(result(0, 0), 5.0);
}

// Test edge cases: all NaN
TEST(LoessTest, AllNaN) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> data(3, 3);
  data.setConstant(nan);

  config::fill::Loess config;
  config = config.with_nx(1).with_ny(1).with_max_iterations(10);

  auto result = loess<double>(data, config);

  // When all values are NaN, first guess fills with zonal average (0.0)
  // Then iterations keep them at 0.0 since all neighbors are 0.0
  for (int i = 0; i < result.rows(); ++i) {
    for (int j = 0; j < result.cols(); ++j) {
      EXPECT_DOUBLE_EQ(result(i, j), 0.0);
    }
  }
}

// Test with zero first guess
TEST(LoessTest, ZeroFirstGuess) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> data(3, 3);
  data << 1.0, 2.0, 3.0,  //
      4.0, nan, 6.0,      //
      7.0, 8.0, 9.0;

  config::fill::Loess config;
  config = config.with_nx(1)
               .with_ny(1)
               .with_first_guess(config::fill::FirstGuess::kZero)
               .with_max_iterations(10);

  auto result = loess<double>(data, config);

  EXPECT_FALSE(std::isnan(result(1, 1)));
}

// Test multi-threading
TEST(LoessTest, MultiThreaded) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  RowMajorMatrix<double> data(10, 10);
  data.setConstant(1.0);
  data(5, 5) = nan;
  data(3, 3) = nan;
  data(7, 7) = nan;

  config::fill::Loess config;
  config =
      config.with_nx(1).with_ny(1).with_max_iterations(10).with_num_threads(4);

  auto result = loess<double>(data, config);

  // All NaN should be filled
  EXPECT_FALSE(std::isnan(result(5, 5)));
  EXPECT_FALSE(std::isnan(result(3, 3)));
  EXPECT_FALSE(std::isnan(result(7, 7)));
}

}  // namespace pyinterp::fill

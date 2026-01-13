// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/math/descriptive_statistics.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace pyinterp::math {

template <typename T>
auto default_tol() -> T {
  // Mixed absolute/relative tolerance scaled by machine epsilon
  return static_cast<T>(100) * std::numeric_limits<T>::epsilon();
}

template <typename T>
auto approx_equal(T a, T b, T tol = default_tol<T>())
    -> ::testing::AssertionResult {
  if (std::isnan(a) && std::isnan(b)) {
    return ::testing::AssertionSuccess();
  }
  if (std::isinf(a) || std::isinf(b)) {
    if (a == b) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << "Inf mismatch: a=" << a << " b=" << b;
  }
  T diff = std::abs(a - b);
  T scale = std::max<T>(static_cast<T>(1), std::max(std::abs(a), std::abs(b)));
  if (diff <= tol * scale) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << "Not approx equal: a=" << a << " b=" << b << " diff=" << diff
         << " tol=" << tol << " scale=" << scale;
}

template <typename T>
void expect_near(T val, T exp, T tol = default_tol<T>()) {
  EXPECT_TRUE(approx_equal(val, exp, tol));
}

template <typename T>
struct Stats : ::testing::Test {};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Stats, TestTypes);

TYPED_TEST(Stats, EmptyStatsNaN) {
  using T = TypeParam;
  DescriptiveStatistics<T> stats;

  EXPECT_TRUE(std::isnan(stats.mean()));
  EXPECT_TRUE(std::isnan(stats.min()));
  EXPECT_TRUE(std::isnan(stats.max()));

  // variance/std: empty -> NaN regardless of ddof
  EXPECT_TRUE(std::isnan(stats.variance(0)));
  EXPECT_TRUE(std::isnan(stats.variance(1)));
  EXPECT_TRUE(std::isnan(stats.std(0)));
  EXPECT_TRUE(std::isnan(stats.std(1)));

  EXPECT_EQ(stats.count(), 0u);
  expect_near(stats.sum_of_weights(), static_cast<T>(0));
  expect_near(stats.sum(), static_cast<T>(0));
}

TYPED_TEST(Stats, SingleValue) {
  using T = TypeParam;
  DescriptiveStatistics<T> stats;
  const T v = static_cast<T>(3.5);
  stats(v);

  EXPECT_EQ(stats.count(), 1u);
  expect_near(stats.sum_of_weights(), static_cast<T>(1));
  expect_near(stats.sum(), v);
  expect_near(stats.mean(), v);
  expect_near(stats.min(), v);
  expect_near(stats.max(), v);

  expect_near(stats.variance(0), static_cast<T>(0));
  expect_near(stats.std(0), static_cast<T>(0));
  EXPECT_TRUE(std::isnan(stats.variance(1)));
  EXPECT_TRUE(std::isnan(stats.std(1)));

  // mom2 == 0 => skewness and kurtosis are NaN
  EXPECT_TRUE(std::isnan(stats.skewness()));
  EXPECT_TRUE(std::isnan(stats.kurtosis()));
}

TYPED_TEST(Stats, MultiUnweightedSequence) {
  using T = TypeParam;
  DescriptiveStatistics<T> stats;

  std::vector<T> data{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3),
                      static_cast<T>(4)};
  for (auto v : data) {
    stats(v);
  }

  EXPECT_EQ(stats.count(), 4u);
  expect_near(stats.sum_of_weights(), static_cast<T>(4));
  expect_near(stats.sum(), static_cast<T>(10));
  expect_near(stats.mean(), static_cast<T>(2.5));
  expect_near(stats.min(), static_cast<T>(1));
  expect_near(stats.max(), static_cast<T>(4));

  // Known exact values for dataset {1,2,3,4}
  // M2 = sum (x - mean)^2 = 5
  // variance(pop) = 5/4 = 1.25 ; variance(sample) = 5/3
  expect_near(stats.variance(0), static_cast<T>(1.25));
  expect_near(stats.std(0), static_cast<T>(std::sqrt(1.25)));
  expect_near(stats.variance(1), static_cast<T>(5.0 / 3.0));
  expect_near(stats.std(1), static_cast<T>(std::sqrt(5.0 / 3.0)));

  // Skewness 0 for symmetric dataset
  expect_near(stats.skewness(), static_cast<T>(0));

  // Excess kurtosis = n * m4 / m2^2 - 3 = 4 * 10.25 / 25 - 3 = -1.36
  expect_near(stats.kurtosis(), static_cast<T>(-1.36), static_cast<T>(1e-5));

  // Direct access to accumulators
  const auto& acc = stats.accumulators();
  expect_near(static_cast<T>(acc.mom2), static_cast<T>(5));
  expect_near(static_cast<T>(acc.mom3), static_cast<T>(0));
  expect_near(static_cast<T>(acc.mom4), static_cast<T>(10.25));
}

TYPED_TEST(Stats, WeightedPush) {
  using T = TypeParam;
  DescriptiveStatistics<T> stats;
  stats(static_cast<T>(1), static_cast<T>(2));  // x=1, w=2
  stats(static_cast<T>(3), static_cast<T>(1));  // x=3, w=1

  expect_near(stats.sum_of_weights(), static_cast<T>(3));
  expect_near(stats.sum(), static_cast<T>(5));  // 2*1 + 1*3
  expect_near(stats.mean(), static_cast<T>(5.0 / 3.0));
  expect_near(stats.min(), static_cast<T>(1));
  expect_near(stats.max(), static_cast<T>(3));

  // M2 = 2*(1-5/3)^2 + 1*(3-5/3)^2 = 8/3
  // Var(pop) = (8/3)/3 = 8/9 ; Var(sample ddof=1) = (8/3)/2 = 4/3
  expect_near(stats.variance(0), static_cast<T>(8.0 / 9.0));
  expect_near(stats.std(0), static_cast<T>(std::sqrt(8.0 / 9.0)));
  expect_near(stats.variance(1), static_cast<T>(4.0 / 3.0));
  expect_near(stats.std(1), static_cast<T>(std::sqrt(4.0 / 3.0)));

  // Skewness (standard) ≈ 0.70710678; Excess kurtosis ≈ -1.5
  expect_near(stats.skewness(), static_cast<T>(0.7071067811865476),
              static_cast<T>(5e-6));
  expect_near(stats.kurtosis(), static_cast<T>(-1.5), static_cast<T>(5e-6));
}

TYPED_TEST(Stats, WeightedUnweightedMixed) {
  using T = TypeParam;
  DescriptiveStatistics<T> stats;
  stats(static_cast<T>(2));                     // w=1
  stats(static_cast<T>(4), static_cast<T>(3));  // w=3

  expect_near(stats.sum_of_weights(), static_cast<T>(4));
  expect_near(stats.sum(), static_cast<T>(14));  // 1*2 + 3*4
  expect_near(stats.mean(), static_cast<T>(3.5));
  expect_near(stats.min(), static_cast<T>(2));
  expect_near(stats.max(), static_cast<T>(4));

  // Compute expected M2: 1*(2-3.5)^2 + 3*(4-3.5)^2 = 1*2.25 + 3*0.25 = 3.0
  expect_near(stats.variance(0), static_cast<T>(3.0 / 4.0));
  expect_near(stats.std(0), static_cast<T>(std::sqrt(3.0 / 4.0)));
  expect_near(stats.variance(1), static_cast<T>(3.0 / 3.0));
  expect_near(stats.std(1), static_cast<T>(1.0));
}

TYPED_TEST(Stats, CombineAccumulatorsEqualSequential) {
  using T = TypeParam;
  std::vector<T> data{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3),
                      static_cast<T>(4), static_cast<T>(5), static_cast<T>(6),
                      static_cast<T>(7), static_cast<T>(8)};
  DescriptiveStatistics<T> seq;
  for (auto v : data) {
    seq(v);
  }

  DescriptiveStatistics<T> a, b;
  for (size_t i = 0; i < data.size(); ++i) {
    (i < data.size() / 2 ? a : b)(data[i]);
  }
  auto combined = a + b;

  expect_near(combined.sum_of_weights(), seq.sum_of_weights());
  EXPECT_EQ(combined.count(), seq.count());
  expect_near(combined.sum(), seq.sum());
  expect_near(combined.mean(), seq.mean());
  expect_near(combined.min(), seq.min());
  expect_near(combined.max(), seq.max());
  expect_near(combined.variance(0), seq.variance(0));
  expect_near(combined.variance(1), seq.variance(1));
  expect_near(combined.std(0), seq.std(0));
  expect_near(combined.std(1), seq.std(1));
  expect_near(combined.skewness(), seq.skewness(), static_cast<T>(1e-6));
  expect_near(combined.kurtosis(), seq.kurtosis(), static_cast<T>(1e-6));
}

TYPED_TEST(Stats, CombineWithEmpty) {
  using T = TypeParam;
  DescriptiveStatistics<T> stats;
  for (int i = 0; i < 5; ++i) {
    stats(static_cast<T>(i + 1));
  }

  DescriptiveStatistics<T> empty;

  auto left = empty + stats;
  auto right = stats + empty;

  expect_near(left.mean(), stats.mean());
  expect_near(left.variance(0), stats.variance(0));
  expect_near(right.mean(), stats.mean());
  expect_near(right.variance(1), stats.variance(1));

  // In-place += with empty should not change stats
  DescriptiveStatistics<T> copy = stats;
  copy += empty;
  expect_near(copy.mean(), stats.mean());
  expect_near(copy.variance(0), stats.variance(0));
}

TYPED_TEST(Stats, OrderIndependenceApprox) {
  using T = TypeParam;
  std::vector<T> data(100);
  std::iota(data.begin(), data.end(), static_cast<T>(1));

  DescriptiveStatistics<T> stats1;
  for (auto v : data) {
    stats1(v);
  }

  std::vector<T> shuffled = data;
  std::mt19937 rng(42);
  std::shuffle(shuffled.begin(), shuffled.end(), rng);

  DescriptiveStatistics<T> stats2;
  for (auto v : shuffled) {
    stats2(v);
  }

  expect_near(stats1.mean(), stats2.mean());
  expect_near(stats1.variance(0), stats2.variance(0));
  expect_near(stats1.skewness(), stats2.skewness(), static_cast<T>(1e-6));
  expect_near(stats1.kurtosis(), stats2.kurtosis(), static_cast<T>(1e-6));
  expect_near(stats1.sum(), stats2.sum());
  EXPECT_EQ(stats1.count(), stats2.count());
}

TYPED_TEST(Stats, LargeOffsetStability) {
  using T = TypeParam;
  const T offset = static_cast<T>(1e9);
  std::vector<T> data{offset + static_cast<T>(1), offset + static_cast<T>(2),
                      offset + static_cast<T>(3), offset + static_cast<T>(4)};

  DescriptiveStatistics<T> stats;
  for (auto v : data) {
    stats(v);
  }
  if constexpr (std::is_same_v<T, float>) {
    // Spacing of float near 1e9 is ~128; deltas of 1..4 collapse.
    // Just assert zero variance due to quantization.
    expect_near(stats.variance(0), static_cast<T>(0));
    return;
  }
  expect_near(stats.mean(), offset + static_cast<T>(2.5),
              static_cast<T>(1e-12));
  expect_near(stats.variance(0), static_cast<T>(1.25), static_cast<T>(1e-12));
  expect_near(stats.skewness(), static_cast<T>(0), static_cast<T>(1e-9));
}

TYPED_TEST(Stats, ClearResets) {
  using T = TypeParam;
  DescriptiveStatistics<T> stats;
  for (int i = 0; i < 10; ++i) {
    stats(static_cast<T>(i));
  }

  stats.clear();

  EXPECT_EQ(stats.count(), 0u);
  expect_near(stats.sum_of_weights(), static_cast<T>(0));
  expect_near(stats.sum(), static_cast<T>(0));
  EXPECT_TRUE(std::isnan(stats.mean()));
  EXPECT_TRUE(std::isnan(stats.min()));
  EXPECT_TRUE(std::isnan(stats.max()));
  EXPECT_TRUE(std::isnan(stats.variance(0)));
  EXPECT_TRUE(std::isnan(stats.std(0)));
}

}  // namespace pyinterp::math

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/math/tdigest.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyinterp::math {

constexpr auto POINTS = 10000;

// Helper function to calculate exact quantile from sorted data
auto quantile(const std::vector<double>& x, double q) -> double {
  if (x.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const auto idx = (x.size() - 1) * q;
  const auto lo = static_cast<size_t>(std::floor(idx));
  const auto hi = static_cast<size_t>(std::ceil(idx));
  return (x[lo] + x[hi]) * 0.5;
}

// Test basic add operation
TEST(MathTDigest, Add) {
  auto digest = TDigest<double>();

  // Empty digest
  EXPECT_EQ(digest.count(), 0);
  EXPECT_EQ(digest.size(), 0);
  EXPECT_TRUE(std::isnan(digest.min()));
  EXPECT_TRUE(std::isnan(digest.max()));
  EXPECT_TRUE(std::isnan(digest.mean()));

  // Add single value
  digest.add(10.0);
  EXPECT_EQ(digest.count(), 1);
  EXPECT_EQ(digest.min(), 10.0);
  EXPECT_EQ(digest.max(), 10.0);

  // Add more values
  digest.add(20.0);
  digest.add(15.0);
  EXPECT_EQ(digest.count(), 3);
  EXPECT_EQ(digest.min(), 10.0);
  EXPECT_EQ(digest.max(), 20.0);
}

// Test add with weights
TEST(MathTDigest, AddWeighted) {
  auto digest = TDigest<double>();

  digest.add(10.0, 2.0);
  EXPECT_EQ(digest.count(), 2);
  EXPECT_EQ(digest.sum_of_weights(), 2.0);

  digest.add(20.0, 3.0);
  EXPECT_EQ(digest.count(), 5);
  EXPECT_EQ(digest.sum_of_weights(), 5.0);

  // Test zero or negative weight (should be ignored)
  digest.add(30.0, 0.0);
  EXPECT_EQ(digest.count(), 5);

  digest.add(40.0, -1.0);
  EXPECT_EQ(digest.count(), 5);
}

// Test operator() for adding values
TEST(MathTDigest, OperatorCall) {
  auto digest = TDigest<double>();

  digest(10.0);
  EXPECT_EQ(digest.count(), 1);

  digest(20.0, 2.0);
  EXPECT_EQ(digest.count(), 3);
}

// Test compression parameter validation
TEST(MathTDigest, CompressionValidation) {
  // Zero compression should throw
  EXPECT_THROW(TDigest<double>(0), std::invalid_argument);

  // Valid compression should work
  EXPECT_NO_THROW(TDigest<double>(1));
  EXPECT_NO_THROW(TDigest<double>(100));
}

// Test set_compression validation
TEST(MathTDigest, SetCompressionValidation) {
  auto digest = TDigest<double>(100);

  // Zero compression should throw
  EXPECT_THROW(digest.set_compression(0), std::invalid_argument);

  // Valid compression should work
  EXPECT_NO_THROW(digest.set_compression(50));
  EXPECT_NO_THROW(digest.set_compression(200));
}

// Test compression parameter
TEST(MathTDigest, Compression) {
  auto digest1 = TDigest<double>(50);
  auto digest2 = TDigest<double>(200);

  auto rd = std::random_device();
  auto gen = std::mt19937(rd());
  auto uniform = std::uniform_real_distribution<>(0.0, 100.0);

  for (auto ix = 0; ix < 1000; ++ix) {
    auto value = uniform(gen);
    digest1.add(value);
    digest2.add(value);
  }

  // Both digests should compress the data
  EXPECT_LT(digest1.size(), 1000);
  EXPECT_LT(digest2.size(), 1000);
}

// Test set_compression
TEST(MathTDigest, SetCompression) {
  auto digest = TDigest<double>(100);

  for (auto ix = 0; ix < 500; ++ix) {
    digest.add(static_cast<double>(ix));
  }

  auto size_before = digest.size();
  digest.set_compression(50);
  auto size_after = digest.size();

  EXPECT_LE(size_after, size_before);
  EXPECT_LE(size_after, 50);
}

// Test clear
TEST(MathTDigest, Clear) {
  auto digest = TDigest<double>();

  digest.add(10.0);
  digest.add(20.0);
  digest.add(30.0);

  EXPECT_GT(digest.count(), 0);
  EXPECT_FALSE(std::isnan(digest.min()));

  digest.clear();

  EXPECT_EQ(digest.count(), 0);
  EXPECT_EQ(digest.size(), 0);
  EXPECT_TRUE(std::isnan(digest.min()));
  EXPECT_TRUE(std::isnan(digest.max()));
  EXPECT_TRUE(std::isnan(digest.mean()));
}

// Test quantile calculation
TEST(MathTDigest, Quantile) {
  auto digest = TDigest<double>();
  auto values = std::vector<double>();

  auto rd = std::random_device();
  auto gen = std::mt19937(rd());
  auto normal = std::normal_distribution<>(0.0, 1.0);

  for (auto ix = 0; ix < POINTS; ++ix) {
    auto value = normal(gen);
    digest.add(value);
    values.push_back(value);
  }

  std::sort(values.begin(), values.end());

  // Test various quantiles
  for (auto q : {0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99}) {
    auto expected = quantile(values, q);
    auto actual = digest.quantile(q);
    auto error = std::abs(expected - actual);
    // T-digest should be accurate within a few percent
    EXPECT_LT(error, 0.1) << "Quantile " << q << " error too large";
  }
}

// Test boundary quantiles
TEST(MathTDigest, QuantileBoundaries) {
  auto digest = TDigest<double>();

  for (auto ix = 1; ix <= 100; ++ix) {
    digest.add(static_cast<double>(ix));
  }

  // q = 0 should return min
  EXPECT_DOUBLE_EQ(digest.quantile(0.0), digest.min());

  // q = 1 should return max
  EXPECT_DOUBLE_EQ(digest.quantile(1.0), digest.max());
}

// Test quantile with empty digest
TEST(MathTDigest, QuantileEmpty) {
  auto digest = TDigest<double>();
  EXPECT_TRUE(std::isnan(digest.quantile(0.5)));
}

// Test quantile out of bounds
TEST(MathTDigest, QuantileOutOfBounds) {
  auto digest = TDigest<double>();
  digest.add(10.0);

  EXPECT_THROW(static_cast<void>(digest.quantile(-0.1)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(digest.quantile(1.1)), std::invalid_argument);
}

// Test single value
TEST(MathTDigest, SingleValue) {
  auto digest = TDigest<double>();
  digest.add(42.0);

  EXPECT_EQ(digest.count(), 1);
  EXPECT_EQ(digest.min(), 42.0);
  EXPECT_EQ(digest.max(), 42.0);
  EXPECT_EQ(digest.mean(), 42.0);
  EXPECT_EQ(digest.quantile(0.0), 42.0);
  EXPECT_EQ(digest.quantile(0.5), 42.0);
  EXPECT_EQ(digest.quantile(1.0), 42.0);
}

// Test mean calculation
TEST(MathTDigest, Mean) {
  auto digest = TDigest<double>();

  digest.add(10.0);
  digest.add(20.0);
  digest.add(30.0);

  auto expected_mean = (10.0 + 20.0 + 30.0) / 3.0;
  EXPECT_NEAR(digest.mean(), expected_mean, 1e-9);
}

// Test weighted mean
TEST(MathTDigest, WeightedMean) {
  auto digest = TDigest<double>();

  digest.add(10.0, 1.0);
  digest.add(20.0, 2.0);
  digest.add(30.0, 3.0);

  auto expected_mean = (10.0 * 1.0 + 20.0 * 2.0 + 30.0 * 3.0) / 6.0;
  EXPECT_NEAR(digest.mean(), expected_mean, 1e-9);
}

// Test min/max tracking
TEST(MathTDigest, MinMax) {
  auto digest = TDigest<double>();
  auto min_val = std::numeric_limits<double>::max();
  auto max_val = std::numeric_limits<double>::lowest();

  auto rd = std::random_device();
  auto gen = std::mt19937(rd());
  auto uniform = std::uniform_real_distribution<>(-100.0, 100.0);

  for (auto ix = 0; ix < 1000; ++ix) {
    auto value = uniform(gen);
    digest.add(value);
    min_val = std::min(min_val, value);
    max_val = std::max(max_val, value);
  }

  EXPECT_DOUBLE_EQ(digest.min(), min_val);
  EXPECT_DOUBLE_EQ(digest.max(), max_val);
}

// Test merging two digests
TEST(MathTDigest, Merge) {
  auto digest1 = TDigest<double>();
  auto digest2 = TDigest<double>();
  auto values = std::vector<double>();

  auto rd = std::random_device();
  auto gen = std::mt19937(rd());
  auto normal = std::normal_distribution<>(0.0, 1.0);

  for (auto ix = 0; ix < POINTS / 2; ++ix) {
    auto value = normal(gen);
    digest1.add(value);
    values.push_back(value);
  }

  for (auto ix = 0; ix < POINTS / 2; ++ix) {
    auto value = normal(gen);
    digest2.add(value);
    values.push_back(value);
  }

  std::sort(values.begin(), values.end());

  // Merge digest2 into digest1
  digest1 += digest2;

  EXPECT_EQ(digest1.count(), POINTS);

  // Check quantile accuracy after merge
  auto expected = quantile(values, 0.5);
  auto actual = digest1.quantile(0.5);
  EXPECT_NEAR(actual, expected, 0.1);

  expected = quantile(values, 0.95);
  actual = digest1.quantile(0.95);
  EXPECT_NEAR(actual, expected, 0.15);
}

// Test merge operator+
TEST(MathTDigest, MergeOperatorPlus) {
  auto digest1 = TDigest<double>();
  auto digest2 = TDigest<double>();

  for (auto ix = 0; ix < 100; ++ix) {
    digest1.add(static_cast<double>(ix));
  }

  for (auto ix = 100; ix < 200; ++ix) {
    digest2.add(static_cast<double>(ix));
  }

  auto merged = digest1 + digest2;

  EXPECT_EQ(merged.count(), 200);
  EXPECT_DOUBLE_EQ(merged.min(), 0.0);
  EXPECT_DOUBLE_EQ(merged.max(), 199.0);
}

// Test merging empty digest
TEST(MathTDigest, MergeEmpty) {
  auto digest1 = TDigest<double>();
  auto digest2 = TDigest<double>();

  digest1.add(10.0);
  digest1.add(20.0);

  auto count_before = digest1.count();
  digest1 += digest2;

  EXPECT_EQ(digest1.count(), count_before);
}

// Test accuracy at distribution tails
TEST(MathTDigest, TailAccuracy) {
  auto digest = TDigest<double>(200);  // Higher compression for tails
  auto values = std::vector<double>();

  auto rd = std::random_device();
  auto gen = std::mt19937(rd());
  auto normal = std::normal_distribution<>(0.0, 1.0);

  for (auto ix = 0; ix < POINTS; ++ix) {
    auto value = normal(gen);
    digest.add(value);
    values.push_back(value);
  }

  std::sort(values.begin(), values.end());

  // T-digest should be especially accurate at tails
  for (auto q : {0.001, 0.01, 0.99, 0.999}) {
    auto expected = quantile(values, q);
    auto actual = digest.quantile(q);
    auto error = std::abs(expected - actual);
    // Extreme tails (0.001, 0.999) may have larger errors
    auto tolerance = (q < 0.01 || q > 0.99) ? 1.5 : 0.2;
    EXPECT_LT(error, tolerance) << "Tail quantile " << q << " error too large";
  }
}

// Test with uniform distribution
TEST(MathTDigest, UniformDistribution) {
  auto digest = TDigest<double>();
  auto values = std::vector<double>();

  auto rd = std::random_device();
  auto gen = std::mt19937(rd());
  auto uniform = std::uniform_real_distribution<>(0.0, 100.0);

  for (auto ix = 0; ix < POINTS; ++ix) {
    auto value = uniform(gen);
    digest.add(value);
    values.push_back(value);
  }

  std::sort(values.begin(), values.end());

  // Check median
  auto expected = quantile(values, 0.5);
  auto actual = digest.quantile(0.5);
  EXPECT_NEAR(actual, expected, 1.0);

  // Mean should be near 50 for uniform(0, 100)
  EXPECT_NEAR(digest.mean(), 50.0, 5.0);
}

// Test with exponential distribution
TEST(MathTDigest, ExponentialDistribution) {
  auto digest = TDigest<double>();
  auto values = std::vector<double>();

  auto rd = std::random_device();
  auto gen = std::mt19937(rd());
  auto exponential = std::exponential_distribution<>(1.0);

  for (auto ix = 0; ix < POINTS; ++ix) {
    auto value = exponential(gen);
    digest.add(value);
    values.push_back(value);
  }

  std::sort(values.begin(), values.end());

  // Check lower quantiles (exponential is skewed)
  for (auto q : {0.1, 0.25, 0.5}) {
    auto expected = quantile(values, q);
    auto actual = digest.quantile(q);
    EXPECT_NEAR(actual, expected, 0.15);
  }
}

// Test serialization
TEST(MathTDigest, Serialization) {
  auto digest = TDigest<double>(100);

  // Empty digest
  auto state_reader = serialization::Reader(digest.pack());
  auto restored = TDigest<double>::unpack(state_reader);
  EXPECT_EQ(restored.count(), 0);
  EXPECT_EQ(restored.size(), 0);

  // Add values
  for (auto ix = 0; ix < 1000; ++ix) {
    digest.add(static_cast<double>(ix));
  }

  state_reader = serialization::Reader(digest.pack());
  restored = TDigest<double>::unpack(state_reader);

  EXPECT_EQ(restored.count(), digest.count());
  EXPECT_EQ(restored.size(), digest.size());
  EXPECT_DOUBLE_EQ(restored.min(), digest.min());
  EXPECT_DOUBLE_EQ(restored.max(), digest.max());
  EXPECT_NEAR(restored.mean(), digest.mean(), 1e-9);

  // Check quantiles match
  for (auto q : {0.1, 0.25, 0.5, 0.75, 0.9}) {
    EXPECT_NEAR(restored.quantile(q), digest.quantile(q), 1e-9);
  }

  // Check centroids
  EXPECT_EQ(restored.centroids().size(), digest.centroids().size());
  for (size_t ix = 0; ix < digest.centroids().size(); ++ix) {
    EXPECT_DOUBLE_EQ(restored.centroids()[ix].mean,
                     digest.centroids()[ix].mean);
    EXPECT_DOUBLE_EQ(restored.centroids()[ix].weight,
                     digest.centroids()[ix].weight);
  }
}

// Test serialization error handling
TEST(MathTDigest, SerializationInvalid) {
  // Empty data
  auto empty_data = std::vector<std::byte>{};
  auto empty_reader = serialization::Reader(std::move(empty_data));
  EXPECT_THROW(static_cast<void>(TDigest<double>::unpack(empty_reader)),
               std::invalid_argument);

  // Invalid version
  auto invalid_version = std::vector<std::byte>{
      std::byte{0xFF}, std::byte{0xFF}, std::byte{0xFF}, std::byte{0xFF},
      std::byte{0},    std::byte{0},    std::byte{0},    std::byte{0}};
  auto version_reader = serialization::Reader(std::move(invalid_version));
  EXPECT_THROW(static_cast<void>(TDigest<double>::unpack(version_reader)),
               std::invalid_argument);
}

// Test with constant values
TEST(MathTDigest, ConstantValues) {
  auto digest = TDigest<double>();

  for (auto ix = 0; ix < 100; ++ix) {
    digest.add(42.0);
  }

  EXPECT_EQ(digest.count(), 100);
  EXPECT_EQ(digest.min(), 42.0);
  EXPECT_EQ(digest.max(), 42.0);
  EXPECT_EQ(digest.mean(), 42.0);

  for (auto q : {0.0, 0.25, 0.5, 0.75, 1.0}) {
    EXPECT_EQ(digest.quantile(q), 42.0);
  }
}

// Test with duplicate values
TEST(MathTDigest, DuplicateValues) {
  auto digest = TDigest<double>();

  // Add values with duplicates
  for (auto value : {1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0}) {
    digest.add(value);
  }

  EXPECT_EQ(digest.count(), 8);
  EXPECT_EQ(digest.min(), 1.0);
  EXPECT_EQ(digest.max(), 5.0);

  // Median should be near 3.0 (middle values)
  auto median = digest.quantile(0.5);
  EXPECT_GT(median, 2.5);
  EXPECT_LT(median, 3.5);
}

// Test with bimodal distribution
TEST(MathTDigest, BimodalDistribution) {
  auto digest = TDigest<double>();

  auto rd = std::random_device();
  auto gen = std::mt19937(rd());
  auto normal1 = std::normal_distribution<>(-5.0, 1.0);
  auto normal2 = std::normal_distribution<>(5.0, 1.0);

  for (auto ix = 0; ix < POINTS / 2; ++ix) {
    digest.add(normal1(gen));
    digest.add(normal2(gen));
  }

  // Mean should be near 0 for balanced bimodal
  EXPECT_NEAR(digest.mean(), 0.0, 1.0);

  // Check that we can estimate quantiles
  EXPECT_LT(digest.quantile(0.25), 0.0);  // Lower mode
  EXPECT_GT(digest.quantile(0.75), 0.0);  // Upper mode
}

// Test large dataset
TEST(MathTDigest, LargeDataset) {
  auto digest = TDigest<double>(100);

  auto rd = std::random_device();
  auto gen = std::mt19937(rd());
  auto normal = std::normal_distribution<>(0.0, 1.0);

  const auto large_n = 100000;
  for (auto ix = 0; ix < large_n; ++ix) {
    digest.add(normal(gen));
  }

  EXPECT_EQ(digest.count(), large_n);
  // Digest should successfully handle large dataset
  EXPECT_GT(digest.size(), 0);

  // Basic sanity checks
  EXPECT_TRUE(std::isfinite(digest.min()));
  EXPECT_TRUE(std::isfinite(digest.max()));
  EXPECT_NEAR(digest.mean(), 0.0, 0.1);

  // Quantiles should work
  auto median = digest.quantile(0.5);
  EXPECT_TRUE(std::isfinite(median));
  EXPECT_NEAR(median, 0.0, 0.2);  // Normal distribution centered at 0
}

// Test float type
TEST(MathTDigest, FloatType) {
  auto digest = TDigest<float>(100);

  for (auto ix = 0; ix < 1000; ++ix) {
    digest.add(static_cast<float>(ix));
  }

  EXPECT_EQ(digest.count(), 1000);
  EXPECT_FLOAT_EQ(digest.min(), 0.0f);
  EXPECT_FLOAT_EQ(digest.max(), 999.0f);
  EXPECT_NEAR(digest.mean(), 499.5f, 1.0f);
  EXPECT_NEAR(digest.quantile(0.5f), 499.5f, 10.0f);
}

// Test weighted quantiles
TEST(MathTDigest, WeightedQuantiles) {
  auto digest = TDigest<double>();

  // Add values with different weights
  digest.add(1.0, 1.0);
  digest.add(2.0, 2.0);
  digest.add(3.0, 3.0);
  digest.add(4.0, 4.0);

  // Total weight = 10, median weight = 5
  // Values: 1(1), 2(2), 3(3), 4(4)
  // Cumulative: 1, 3, 6, 10
  // Median should be between 2 and 3, closer to 3
  auto median = digest.quantile(0.5);
  EXPECT_GT(median, 2.0);
  EXPECT_LT(median, 4.0);
}

// Test centroids accessor
TEST(MathTDigest, CentroidsAccessor) {
  auto digest = TDigest<double>(10);

  for (auto ix = 0; ix < 100; ++ix) {
    digest.add(static_cast<double>(ix));
  }

  const auto& centroids = digest.centroids();
  EXPECT_GT(centroids.size(), 0);
  EXPECT_LT(centroids.size(),
            100);  // Should compress the 100 values significantly

  // Centroids should be sorted by mean
  for (size_t ix = 1; ix < centroids.size(); ++ix) {
    EXPECT_GE(centroids[ix].mean, centroids[ix - 1].mean);
  }

  // All weights should be positive
  for (const auto& centroid : centroids) {
    EXPECT_GT(centroid.weight, 0.0);
  }
}

// Test sum_of_weights
TEST(MathTDigest, SumOfWeights) {
  auto digest = TDigest<double>();

  EXPECT_EQ(digest.sum_of_weights(), 0.0);

  digest.add(10.0, 2.5);
  EXPECT_EQ(digest.sum_of_weights(), 2.5);

  digest.add(20.0, 3.5);
  EXPECT_EQ(digest.sum_of_weights(), 6.0);

  digest.add(30.0);
  EXPECT_EQ(digest.sum_of_weights(), 7.0);
}

// Test centroid structure comparison
TEST(MathTDigest, CentroidComparison) {
  auto c1 = Centroid<double>{.mean = 10.0, .weight = 1.0};
  auto c2 = Centroid<double>{.mean = 20.0, .weight = 1.0};
  auto c3 = Centroid<double>{.mean = 10.0, .weight = 2.0};

  EXPECT_TRUE(c1.mean < c2.mean);
  EXPECT_FALSE(c2.mean < c1.mean);
  EXPECT_FALSE(c1.mean < c3.mean);  // Same mean
  EXPECT_FALSE(c3.mean < c1.mean);  // Same mean
}

}  // namespace pyinterp::math

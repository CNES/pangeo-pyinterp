// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/p_square_quantile.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <fstream>
#include <random>

#include "pyinterp/detail/math/descriptive_statistics.hpp"
#include "pyinterp/detail/math/streaming_histogram.hpp"

namespace math = pyinterp::detail::math;

constexpr auto POINTS = 1000;

auto quantile(const std::vector<double> &x, double q) {
  const auto ix = (x.size() - 1) * q;
  const auto lo = floor(ix);
  const auto hi = ceil(ix);

  return (x[lo] + x[hi]) * 0.5;
}

TEST(math_streaming_histogram, push) {
  auto instance = math::StreamingHistogram<double>(3, false);

  instance(10);
  const auto &bins = instance.bins();
  ASSERT_EQ(bins.size(), 1);
  EXPECT_EQ(bins[0].value, 10);
  EXPECT_EQ(bins[0].weight, 1);

  instance(13);
  ASSERT_EQ(bins.size(), 2);
  EXPECT_EQ(bins[0].value, 10);
  EXPECT_EQ(bins[0].weight, 1);
  EXPECT_EQ(bins[1].value, 13);
  EXPECT_EQ(bins[1].weight, 1);

  instance(3);
  ASSERT_EQ(bins.size(), 3);
  EXPECT_EQ(bins[0].value, 3);
  EXPECT_EQ(bins[0].weight, 1);
  EXPECT_EQ(bins[1].value, 10);
  EXPECT_EQ(bins[1].weight, 1);
  EXPECT_EQ(bins[2].value, 13);
  EXPECT_EQ(bins[2].weight, 1);

  instance(13);
  ASSERT_EQ(bins.size(), 3);
  EXPECT_EQ(bins[0].value, 3);
  EXPECT_EQ(bins[0].weight, 1);
  EXPECT_EQ(bins[1].value, 10);
  EXPECT_EQ(bins[1].weight, 1);
  EXPECT_EQ(bins[2].value, 13);
  EXPECT_EQ(bins[2].weight, 2);

  instance(3);
  ASSERT_EQ(bins.size(), 3);
  EXPECT_EQ(bins[0].value, 3);
  EXPECT_EQ(bins[0].weight, 2);
  EXPECT_EQ(bins[1].value, 10);
  EXPECT_EQ(bins[1].weight, 1);
  EXPECT_EQ(bins[2].value, 13);
  EXPECT_EQ(bins[2].weight, 2);

  instance(10);
  ASSERT_EQ(bins.size(), 3);
  EXPECT_EQ(bins[0].value, 3);
  EXPECT_EQ(bins[0].weight, 2);
  EXPECT_EQ(bins[1].value, 10);
  EXPECT_EQ(bins[1].weight, 2);
  EXPECT_EQ(bins[2].value, 13);
  EXPECT_EQ(bins[2].weight, 2);

  instance(11);
  ASSERT_EQ(bins.size(), 3);
  EXPECT_EQ(bins[0].value, 3);
  EXPECT_EQ(bins[0].weight, 2);
  EXPECT_NEAR(bins[1].value, 10 + 1.0 / 3.0, 1e-9);
  EXPECT_EQ(bins[1].weight, 3);
  EXPECT_EQ(bins[2].value, 13);
  EXPECT_EQ(bins[2].weight, 2);
}

TEST(math_streaming_histogram, sum_of_weights) {
  auto instance = math::StreamingHistogram<double>(3, false);
  EXPECT_EQ(instance.count(), 0);
  EXPECT_EQ(instance.size(), 0);
  EXPECT_EQ(instance.sum_of_weights(), 0);

  instance(0, 4);
  EXPECT_EQ(instance.count(), 1);
  EXPECT_EQ(instance.size(), 1);
  EXPECT_EQ(instance.sum_of_weights(), 4);

  instance(1, 3);
  EXPECT_EQ(instance.count(), 2);
  EXPECT_EQ(instance.size(), 2);
  EXPECT_EQ(instance.sum_of_weights(), 7);

  instance(2, 5);
  EXPECT_EQ(instance.count(), 3);
  EXPECT_EQ(instance.size(), 3);
  EXPECT_EQ(instance.sum_of_weights(), 12);
}

TEST(math_streaming_histogram, bounds) {
  auto rd = std::random_device();
  auto gen = std::mt19937(rd());
  auto normal = std::normal_distribution<>();
  auto instance = math::StreamingHistogram<double>();
  auto min = std::numeric_limits<double>::max();
  auto max = std::numeric_limits<double>::min();

  for (auto ix = 0; ix < POINTS; ++ix) {
    auto value = normal(gen);
    instance(value);
    min = std::min(min, value);
    max = std::max(max, value);
  }
  EXPECT_NEAR(min, instance.min(), 1e-6);
  EXPECT_NEAR(max, instance.max(), 1e-6);
}

TEST(math_streaming_histogram, quantile) {
  auto instance = math::StreamingHistogram<double>(3, false);
  instance(1, 4);
  instance(5, 3);
  instance(10, 5);

  auto expected = instance.quantile(0.5);
  EXPECT_NEAR(expected, 5.625, 1e-9);
}

TEST(math_streaming_histogram, quantile_not_enough_elements) {
  auto instance = math::StreamingHistogram<double>(10, false);
  for (const auto &item : std::vector<double>({31, 56, 40, 39, 82, 17})) {
    instance(item);
  }

  auto expected = instance.quantile(0.5);
  EXPECT_NEAR(expected, 39.5, 1e-9);
}

TEST(math_streaming_histogram, quantile_on_left) {
  auto instance = math::StreamingHistogram<double>(6, false);
  for (const auto &item : std::vector<double>(
           {3.075, 1.3, 1.35, 1.225, 1.375, 1.4, 2.05, 7.6325, 5.875, 3.495})) {
    instance(item);
  }

  auto expected = instance.quantile(0.01);
  auto exact = 1.23175;
  EXPECT_NEAR(expected, exact, exact * 0.01);

  expected = instance.quantile(0.05);
  exact = 1.25875;
  EXPECT_NEAR(expected, exact, exact * 0.05);

  expected = instance.quantile(0.25);
  exact = 1.35625;
  EXPECT_NEAR(expected, exact, exact * 0.05);
}

TEST(math_streaming_histogram, quantile_on_right) {
  auto instance = math::StreamingHistogram<double>(6, false);
  for (const auto &item :
       std::vector<double>({3.075, 2.05, 25.1325, 5.875, 3.495, 50., 50.05,
                            50.2, 50.1, 50.025})) {
    instance(item);
  }

  auto expected = instance.quantile(0.99);
  auto exact = 50.191;
  EXPECT_NEAR(expected, exact, exact * 0.01);

  expected = instance.quantile(0.85);
  exact = 50.0825;
  EXPECT_NEAR(expected, exact, exact * 0.01);
}

TEST(math_streaming_histogram, stats) {
  auto rd = std::random_device();
  auto gen = std::mt19937(rd());
  auto normal = std::normal_distribution<>();
  auto acc = math::DescriptiveStatistics<double>();
  auto instance = math::StreamingHistogram<double>(40, false);
  auto values = std::vector<double>();

  for (auto ix = 0; ix < POINTS; ++ix) {
    auto value = normal(gen);
    instance(value);
    acc(value);
    values.push_back(value);
  }

  std::sort(values.begin(), values.end());

  auto expected = instance.quantile(0.5);
  auto exact = quantile(values, 0.5);
  ASSERT_NEAR(std::abs(expected - exact), 0, 0.2);

  expected = instance.quantile(0.8);
  exact = quantile(values, 0.8);
  ASSERT_NEAR(std::abs(expected - exact), 0, 0.2);

  EXPECT_EQ(acc.count(), instance.count());
  EXPECT_EQ(acc.min(), instance.min());
  EXPECT_EQ(acc.max(), instance.max());
  EXPECT_EQ(acc.sum_of_weights(), instance.sum_of_weights());
  acc.clear();
  for (const auto &item : instance.bins()) {
    acc(item.value, item.weight);
  }
  EXPECT_NEAR(acc.mean(), instance.mean(), 1e-6);
  EXPECT_NEAR(acc.variance(), instance.variance(), 1e-6);
  EXPECT_NEAR(acc.skewness(), instance.skewness(), 1e-6);
  EXPECT_NEAR(acc.kurtosis(), instance.kurtosis(), 1e-6);
}

TEST(math_streaming_histogram, merge) {
  auto rd = std::random_device();
  auto gen = std::mt19937(rd());
  auto normal = std::normal_distribution<>();
  auto acc = math::DescriptiveStatistics<double>();
  auto instance1 = math::StreamingHistogram<double>(40, false);
  auto instance2 = math::StreamingHistogram<double>(40, false);
  auto values = std::vector<double>();

  for (auto ix = 0; ix < POINTS / 2; ++ix) {
    auto value = normal(gen);
    instance1(value);
    acc(value);
    values.push_back(value);
  }

  for (auto ix = 0; ix < POINTS / 2; ++ix) {
    auto value = normal(gen);
    instance2(value);
    acc(value);
    values.push_back(value);
  }

  std::sort(values.begin(), values.end());
  instance1 += instance2;

  auto expected = instance1.quantile(0.5);
  auto exact = quantile(values, 0.5);
  ASSERT_NEAR(std::abs(expected - exact), 0, 0.2);

  expected = instance1.quantile(0.8);
  exact = quantile(values, 0.8);
  ASSERT_NEAR(std::abs(expected - exact), 0, 0.2);

  EXPECT_EQ(acc.count(), instance1.count());
  EXPECT_NEAR(acc.min(), instance1.min(), 1e-6);
  EXPECT_NEAR(acc.max(), instance1.max(), 1e-6);
  EXPECT_EQ(acc.sum_of_weights(), instance1.sum_of_weights());
  acc.clear();
  for (const auto &item : instance1.bins()) {
    acc(item.value, item.weight);
  }
  EXPECT_NEAR(acc.mean(), instance1.mean(), 1e-6);
  EXPECT_NEAR(acc.variance(), instance1.variance(), 1e-6);
}

TEST(math_streaming_histogram, quantile_out_of_bounds) {
  auto instance = math::StreamingHistogram<double>(6, false);
  EXPECT_TRUE(std::isnan(instance.quantile(-0.2)));

  for (const auto &item : std::vector<double>({1, 2, 3, 4, 5, 6, 6.1, 6.2})) {
    instance(item);
  }

  EXPECT_THROW(static_cast<void>(instance.quantile(-0.2)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(instance.quantile(10)), std::invalid_argument);
}

TEST(math_streaming_histogram, serialization) {
  auto instance = math::StreamingHistogram<double>(6, false);

  auto dump = static_cast<std::string>(instance);
  auto instance2 = math::StreamingHistogram<double>(dump);
  ASSERT_EQ(instance.count(), instance2.count());
  ASSERT_EQ(instance.sum_of_weights(), instance2.sum_of_weights());
  ASSERT_EQ(instance.bins().size(), instance2.bins().size());

  for (const auto &item :
       std::vector<double>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})) {
    instance(item);
  }

  dump = static_cast<std::string>(instance);
  instance2 = math::StreamingHistogram<double>(dump);
  ASSERT_EQ(instance.count(), instance2.count());
  ASSERT_EQ(instance.sum_of_weights(), instance2.sum_of_weights());
  ASSERT_EQ(instance.size(), instance2.size());
  for (size_t ix = 0; ix < instance.size(); ++ix) {
    ASSERT_EQ(instance.bins()[ix].value, instance2.bins()[ix].value);
  }

  ASSERT_THROW(math::StreamingHistogram<double>("AZERTYUIOP"),
               std::invalid_argument);
}

TEST(math_streaming_histogram, weighted) {
  static double x[20] = {0.00402322, 0.19509434, 0.6425439,  0.66463742,
                         0.76523411, 0.91985221, 0.82729929, 0.21502902,
                         0.48254104, 0.97854649, 0.61394511, 0.00583773,
                         0.06630172, 0.57173946, 0.5881294,  0.30185368,
                         0.18126563, 0.84524097, 0.13754961, 0.17343529};
  static double w[20] = {0.45463566, 0.46341234, 0.2072285,  0.02272363,
                         0.76796619, 0.01987153, 0.43634701, 0.1369698,
                         0.65012667, 0.18825124, 0.96310554, 0.31995482,
                         0.28808939, 0.69961506, 0.97369255, 0.98436659,
                         0.05230501, 0.8073624,  0.40509977, 0.6325752};
  auto acc = math::DescriptiveStatistics<double>();
  auto instance = math::StreamingHistogram<double>(20, false);

  for (auto ix = 0; ix < 20; ++ix) {
    acc(x[ix], w[ix]);
    instance(x[ix], w[ix]);
  }

  EXPECT_EQ(instance.count(), acc.count());
  EXPECT_EQ(instance.sum_of_weights(), acc.sum_of_weights());
  EXPECT_DOUBLE_EQ(instance.mean(), acc.mean());
  EXPECT_NEAR(instance.variance(), acc.variance(), 1e-12);
  EXPECT_NEAR(instance.quantile(0.5), 0.5716742560345885, 1e-12);

  instance = math::StreamingHistogram<double>(10, false);

  for (auto ix = 0; ix < 20; ++ix) {
    instance(x[ix], w[ix]);
  }

  EXPECT_EQ(instance.count(), acc.count());
  EXPECT_TRUE(instance.count() > instance.size());
  EXPECT_NEAR(instance.sum_of_weights(), acc.sum_of_weights(), 1e-6);
  EXPECT_NEAR(instance.mean(), acc.mean(), 1e-6);
  EXPECT_NEAR(instance.variance(), acc.variance(), 1e-3);
  EXPECT_NEAR(instance.quantile(0.5), 0.5716742560345885, 1e-1);
}

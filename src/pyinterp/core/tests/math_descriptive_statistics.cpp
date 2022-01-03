// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/weighted_kurtosis.hpp>
#include <boost/accumulators/statistics/weighted_mean.hpp>
#include <boost/accumulators/statistics/weighted_median.hpp>
#include <boost/accumulators/statistics/weighted_skewness.hpp>
#include <boost/accumulators/statistics/weighted_sum.hpp>
#include <boost/accumulators/statistics/weighted_variance.hpp>

#include "pyinterp/detail/math/descriptive_statistics.hpp"

namespace math = pyinterp::detail::math;

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

using Accumulators = boost::accumulators::accumulator_set<
    double,
    boost::accumulators::stats<
        boost::accumulators::tag::count, boost::accumulators::tag::max,
        boost::accumulators::tag::min, boost::accumulators::tag::sum_of_weights,
        boost::accumulators::tag::weighted_kurtosis,
        boost::accumulators::tag::weighted_mean,
        // boost::accumulators::tag::weighted_median(
        //     boost::accumulators::with_p_square_quantile),
        boost::accumulators::tag::weighted_skewness,
        boost::accumulators::tag::weighted_sum,
        boost::accumulators::tag::weighted_variance(boost::accumulators::lazy)>,
    double>;

TEST(math_descriptive_statistics, univariate) {
  auto boost_acc = Accumulators();
  auto acc = math::DescriptiveStatistics<double>();

  for (auto ix = 0; ix < 20; ++ix) {
    boost_acc(x[ix], boost::accumulators::weight = 1);
    acc(x[ix]);
  }

  EXPECT_EQ(boost::accumulators::count(boost_acc), acc.count());
  EXPECT_DOUBLE_EQ(boost::accumulators::min(boost_acc), acc.min());
  EXPECT_DOUBLE_EQ(boost::accumulators::max(boost_acc), acc.max());
  EXPECT_DOUBLE_EQ(boost::accumulators::weighted_mean(boost_acc), acc.mean());
  EXPECT_NEAR(boost::accumulators::weighted_variance(boost_acc), acc.variance(),
              1e-12);
  EXPECT_NEAR(boost::accumulators::weighted_kurtosis(boost_acc), acc.kurtosis(),
              1e-12);
  EXPECT_NEAR(boost::accumulators::weighted_skewness(boost_acc), acc.skewness(),
              1e-12);
  EXPECT_DOUBLE_EQ(boost::accumulators::weighted_sum(boost_acc), acc.sum());
  EXPECT_DOUBLE_EQ(boost::accumulators::sum_of_weights(boost_acc),
                   acc.sum_of_weights());

  auto copy = math::DescriptiveStatistics<double>(
      static_cast<math::Accumulators<double>>(acc));
  EXPECT_EQ(boost::accumulators::count(boost_acc), copy.count());
  EXPECT_DOUBLE_EQ(boost::accumulators::min(boost_acc), copy.min());
  EXPECT_DOUBLE_EQ(boost::accumulators::max(boost_acc), copy.max());
  EXPECT_DOUBLE_EQ(boost::accumulators::weighted_mean(boost_acc), copy.mean());
  EXPECT_NEAR(boost::accumulators::weighted_variance(boost_acc),
              copy.variance(), 1e-12);
  EXPECT_NEAR(boost::accumulators::weighted_kurtosis(boost_acc),
              copy.kurtosis(), 1e-12);
  EXPECT_NEAR(boost::accumulators::weighted_skewness(boost_acc),
              copy.skewness(), 1e-12);
  EXPECT_DOUBLE_EQ(boost::accumulators::weighted_sum(boost_acc), copy.sum());
  EXPECT_DOUBLE_EQ(boost::accumulators::sum_of_weights(boost_acc),
                   copy.sum_of_weights());
}

TEST(math_descriptive_statistics, weighted) {
  auto boost_acc = Accumulators();
  auto acc = math::DescriptiveStatistics<double>();
  auto min = std::numeric_limits<double>::max();
  auto max = std::numeric_limits<double>::min();

  for (auto ix = 0; ix < 20; ++ix) {
    boost_acc(x[ix], boost::accumulators::weight = w[ix]);
    acc(x[ix], w[ix]);
    auto value = x[ix] * w[ix];
    min = std::min(min, value);
    max = std::max(max, value);
  }

  EXPECT_EQ(boost::accumulators::count(boost_acc), acc.count());
  EXPECT_DOUBLE_EQ(min, acc.min());
  EXPECT_DOUBLE_EQ(max, acc.max());
  EXPECT_DOUBLE_EQ(boost::accumulators::weighted_mean(boost_acc), acc.mean());
  EXPECT_NEAR(boost::accumulators::weighted_variance(boost_acc), acc.variance(),
              1e-12);
  EXPECT_NEAR(boost::accumulators::weighted_kurtosis(boost_acc), acc.kurtosis(),
              1e-12);
  EXPECT_NEAR(boost::accumulators::weighted_skewness(boost_acc), acc.skewness(),
              1e-12);
  EXPECT_DOUBLE_EQ(boost::accumulators::weighted_sum(boost_acc), acc.sum());
  EXPECT_DOUBLE_EQ(boost::accumulators::sum_of_weights(boost_acc),
                   acc.sum_of_weights());
}

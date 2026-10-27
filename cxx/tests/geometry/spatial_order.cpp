// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geometry/spatial_order.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace pyinterp::geometry {

using Point3D =
    boost::geometry::model::point<double, 3, boost::geometry::cs::cartesian>;

// The order is a permutation of the indexes, undefined coordinates included.
TEST(SpatialOrder, Permutation) {
  auto gen = std::mt19937(42);
  auto uniform = std::uniform_real_distribution<>(-1.0, 1.0);
  auto points = std::vector<Point3D>();
  for (auto ix = 0; ix < 1000; ++ix) {
    points.emplace_back(uniform(gen), uniform(gen), uniform(gen));
  }
  points[10] = Point3D(std::numeric_limits<double>::quiet_NaN(), 0, 0);

  auto order = spatial_order(points);
  std::ranges::sort(order);
  auto expected = std::vector<int64_t>(points.size());
  std::ranges::iota(expected, int64_t{0});
  EXPECT_EQ(order, expected);

  EXPECT_TRUE(spatial_order(std::vector<Point3D>{}).empty());
  EXPECT_EQ(spatial_order(std::vector<Point3D>{Point3D(1, 2, 3)}),
            std::vector<int64_t>{0});
}

// Consecutive points of the order are much closer than in random order.
TEST(SpatialOrder, Locality) {
  auto gen = std::mt19937(42);
  auto uniform = std::uniform_real_distribution<>(0.0, 1.0);
  auto points = std::vector<Point3D>();
  for (auto ix = 0; ix < 10'000; ++ix) {
    points.emplace_back(uniform(gen), uniform(gen), uniform(gen));
  }
  const auto path_length = [&points](const std::vector<int64_t>& order) {
    double result = 0;
    for (size_t ix = 1; ix < order.size(); ++ix) {
      result +=
          boost::geometry::distance(points[order[ix - 1]], points[order[ix]]);
    }
    return result;
  };
  auto identity = std::vector<int64_t>(points.size());
  std::ranges::iota(identity, int64_t{0});
  EXPECT_LT(path_length(spatial_order(points)), path_length(identity) / 10);
}

}  // namespace pyinterp::geometry

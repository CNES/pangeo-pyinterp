// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geometry/rtree.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/geometry.hpp>
#include <cstddef>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/rtree_strategy.hpp"

namespace pyinterp::geometry {

using geographic::Point;
using Box = boost::geometry::model::box<Point>;

// Boxes crossing the antimeridian are stored with a maximum longitude greater
// than 180°. The distance from a point located west of such a box must be the
// distance to its western edge (GitHub issue #38).
TEST(GeographicPointBoxDistance, AntimeridianCrossingBox) {
  const auto strategy = detail::GeographicPointBoxDistance<>();

  // Reference: box that does not cross the antimeridian, point 0.03° west.
  const auto expected =
      strategy.apply(Point{4.97, 43.52}, Box{{5, 43}, {100, 45}});
  EXPECT_NEAR(
      expected,
      boost::geometry::distance(Point{4.97, 43.52}, Box{{5, 43}, {100, 45}}),
      1e-6);

  // Longitude span greater than 180°
  EXPECT_NEAR(strategy.apply(Point{4.97, 43.52}, Box{{5, 43}, {185, 45}}),
              expected, 1e-6);
  // Narrow box crossing the antimeridian, point west and east of the box
  EXPECT_NEAR(strategy.apply(Point{169.97, 43.52}, Box{{170, 43}, {190, 45}}),
              expected, 1e-6);
  EXPECT_NEAR(strategy.apply(Point{-169.97, 43.52}, Box{{170, 43}, {190, 45}}),
              expected, 1e-6);
  // Point longitude expressed in [0, 360[
  EXPECT_NEAR(strategy.apply(Point{259.97, 43.52}, Box{{-100, 43}, {85, 45}}),
              expected, 1e-6);
  // Points inside the longitude band of the box
  EXPECT_EQ(strategy.apply(Point{100, 44}, Box{{5, 43}, {185, 45}}), 0);
  EXPECT_EQ(strategy.apply(Point{350, 44}, Box{{-20, 43}, {0, 45}}), 0);
  EXPECT_NEAR(strategy.apply(Point{-176, 46}, Box{{5, 43}, {185, 45}}),
              boost::geometry::distance(Point{-176, 46}, Point{-176, 45}),
              1e-6);
}

// The nearest neighbors must not depend on the layout of the tree when the
// indexed longitudes span more than 180° (GitHub issue #38).
TEST(GeographicRTree, NearestNeighborsWithWideLongitudeSpan) {
  // Regular 0.05° lattice over [5, 6[ x [43, 44[, plus one distant node.
  std::vector<std::pair<Point, double>> points;
  for (int iy = 0; iy < 20; ++iy) {
    for (int ix = 0; ix < 20; ++ix) {
      points.emplace_back(Point{5.0 + (ix * 0.05), 43.0 + (iy * 0.05)},
                          static_cast<double>(points.size()));
    }
  }
  points.emplace_back(Point{-175.0, 45.0}, static_cast<double>(points.size()));

  std::mt19937 generator(42);
  std::uniform_real_distribution<double> lon(4.5, 6.5);
  std::uniform_real_distribution<double> lat(42.5, 44.5);
  std::vector<Point> queries{{4.97, 43.52}};
  for (int ix = 0; ix < 100; ++ix) {
    queries.emplace_back(lon(generator), lat(generator));
  }

  constexpr uint32_t k = 3;
  for (int trial = 0; trial < 10; ++trial) {
    std::shuffle(points.begin(), points.end(), generator);

    RTree<Point, double> tree;
    if (trial % 2 == 0) {
      for (const auto& item : points) {
        tree.insert(item);
      }
    } else {
      tree.packing(points);
    }

    for (const auto& query : queries) {
      std::vector<double> expected;
      expected.reserve(points.size());
      for (const auto& item : points) {
        expected.push_back(boost::geometry::distance(query, item.first));
      }
      std::ranges::sort(expected);

      const auto result =
          tree.query(query, k, std::numeric_limits<double>::max());
      ASSERT_EQ(result.size(), k);
      for (size_t ix = 0; ix < k; ++ix) {
        EXPECT_NEAR(result[ix].first, expected[ix], 1e-6)
            << "query (" << query.lon() << ", " << query.lat() << "), trial "
            << trial;
      }
    }
  }
}

}  // namespace pyinterp::geometry

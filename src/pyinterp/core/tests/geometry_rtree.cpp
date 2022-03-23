// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include "pyinterp/detail/geometry/rtree.hpp"

namespace geometry = pyinterp::detail::geometry;
namespace math = pyinterp::detail::math;

using RTree = geometry::RTree<geometry::Point2D<double>, int64_t>;

TEST(geometry_rtree, constructor) {
  auto rtree = RTree();
  EXPECT_TRUE(rtree.empty());
  rtree.insert(std::make_pair(geometry::Point2D<double>(2, 3), 0));
  rtree.insert(std::make_pair(geometry::Point2D<double>(5, 4), 1));
  rtree.insert(std::make_pair(geometry::Point2D<double>(9, 6), 2));
  rtree.insert(std::make_pair(geometry::Point2D<double>(4, 7), 3));
  rtree.insert(std::make_pair(geometry::Point2D<double>(8, 1), 4));
  rtree.insert(std::make_pair(geometry::Point2D<double>(7, 2), 5));
  ASSERT_FALSE(rtree.empty());
  EXPECT_EQ(rtree.size(), 6);
  auto bounds = rtree.bounds();
  ASSERT_TRUE(bounds);
  auto min_corner = bounds->min_corner();
  EXPECT_EQ(boost::geometry::get<0>(min_corner), 2);
  EXPECT_EQ(boost::geometry::get<1>(min_corner), 1);
  auto max_corner = bounds->max_corner();
  EXPECT_EQ(boost::geometry::get<0>(max_corner), 9);
  EXPECT_EQ(boost::geometry::get<1>(max_corner), 7);
  rtree.clear();
  EXPECT_TRUE(rtree.empty());
}

static auto get_coordinates() -> std::vector<RTree::value_t> {
  // https://en.wikipedia.org/wiki/K-d_tree#/media/File:Kdtree_2d.svg
  return {{geometry::Point2D<double>(2, 3), 0},
          {geometry::Point2D<double>(5, 4), 1},
          {geometry::Point2D<double>(9, 6), 2},
          {geometry::Point2D<double>(4, 7), 3},
          {geometry::Point2D<double>(8, 1), 4},
          {geometry::Point2D<double>(7, 2), 5}};
}

TEST(geometry_rtree, query) {
  auto rtree = RTree();
  rtree.packing(get_coordinates());
  auto nearest = rtree.query({3, 4}, 1);
  ASSERT_EQ(nearest.size(), 1);
  EXPECT_EQ(nearest[0].second, 0);
  nearest = rtree.query({3, 4}, 3);
  ASSERT_EQ(nearest.size(), 3);
  EXPECT_EQ(nearest[0].second, 0);
  EXPECT_EQ(nearest[1].second, 1);
  EXPECT_EQ(nearest[2].second, 3);

  nearest = rtree.query_ball({4, 4}, 1);
  ASSERT_EQ(nearest.size(), 1);
  EXPECT_EQ(nearest[0].second, 1);

  nearest = rtree.query_ball({4, 4}, 3);
  ASSERT_EQ(nearest.size(), 3);
  EXPECT_EQ(nearest[0].second, 0);
  EXPECT_EQ(nearest[1].second, 1);
  EXPECT_EQ(nearest[2].second, 3);

  nearest = rtree.query_within({4, 4}, 3);
  EXPECT_EQ(nearest.size(), 3);
  nearest = rtree.query_within({0, 0}, 3);
  EXPECT_EQ(nearest.size(), 0);
  nearest = rtree.query_within({2, 4}, 3);
  EXPECT_EQ(nearest.size(), 3);
  nearest = rtree.query_within({2, 3}, 3);
  EXPECT_EQ(nearest.size(), 3);
}

TEST(geometry_rtree, inverse_distance_weighting) {
  auto rtree = RTree();
  rtree.packing(get_coordinates());

  auto idw = rtree.inverse_distance_weighting({4, 6}, 2, 4, 2, true);
  EXPECT_EQ(idw.first, 3);
  EXPECT_EQ(idw.second, 1);

  idw = rtree.inverse_distance_weighting({4, 4}, 3, 4, 2, false);
  EXPECT_EQ(idw.first, 1);
  EXPECT_EQ(idw.second, 3);

  idw = rtree.inverse_distance_weighting({4, 4}, 0.1, 4, 2, false);
  EXPECT_EQ(idw.second, 0);

  idw = rtree.inverse_distance_weighting({0, 0}, 3, 4, 2, true);
  EXPECT_EQ(idw.second, 0);
}

TEST(geometry_rtree, nearest) {
  auto rtree = RTree();
  rtree.packing(get_coordinates());

  auto nearest = rtree.nearest({4, 4}, 3, 4);
  auto points = std::get<0>(nearest);
  ASSERT_EQ(points.cols(), 3);
  ASSERT_EQ(points.rows(), 2);
  EXPECT_EQ(points(0, 0), 5);
  EXPECT_EQ(points(1, 0), 4);
  EXPECT_EQ(points(0, 1), 2);
  EXPECT_EQ(points(1, 1), 3);
  EXPECT_EQ(points(0, 2), 4);
  EXPECT_EQ(points(1, 2), 7);

  auto values = std::get<1>(nearest);
  ASSERT_EQ(values.size(), 3);
  EXPECT_EQ(values(0), 1);
  EXPECT_EQ(values(1), 0);
  EXPECT_EQ(values(2), 3);

  nearest = rtree.nearest({4, 4}, 0.1, 4);
  points = std::get<0>(nearest);
  ASSERT_EQ(points.cols(), 0);
  ASSERT_EQ(points.rows(), 2);

  values = std::get<1>(nearest);
  ASSERT_EQ(values.size(), 0);

  nearest = rtree.nearest({2, 8}, 5, 4);
  points = std::get<0>(nearest);
  ASSERT_EQ(points.cols(), 3);
  ASSERT_EQ(points.rows(), 2);
  EXPECT_EQ(points(0, 0), 4);
  EXPECT_EQ(points(1, 0), 7);
  EXPECT_EQ(points(0, 1), 2);
  EXPECT_EQ(points(1, 1), 3);
  EXPECT_EQ(points(0, 2), 5);
  EXPECT_EQ(points(1, 2), 4);

  values = std::get<1>(nearest);
  ASSERT_EQ(values.size(), 3);
  EXPECT_EQ(values(0), 3);
  EXPECT_EQ(values(1), 0);
  EXPECT_EQ(values(2), 1);
}

TEST(geometry_rtree, nearest_within) {
  auto rtree = RTree();
  rtree.packing(get_coordinates());

  auto nearest = rtree.nearest_within({4, 4}, 3, 4);
  auto points = std::get<0>(nearest);
  ASSERT_EQ(points.cols(), 3);
  ASSERT_EQ(points.rows(), 2);
  EXPECT_EQ(points(0, 0), 5);
  EXPECT_EQ(points(1, 0), 4);
  EXPECT_EQ(points(0, 1), 2);
  EXPECT_EQ(points(1, 1), 3);
  EXPECT_EQ(points(0, 2), 4);
  EXPECT_EQ(points(1, 2), 7);

  auto values = std::get<1>(nearest);
  ASSERT_EQ(values.size(), 3);
  EXPECT_EQ(values(0), 1);
  EXPECT_EQ(values(1), 0);
  EXPECT_EQ(values(2), 3);

  nearest = rtree.nearest_within({4, 4}, 0.1, 4);
  points = std::get<0>(nearest);
  ASSERT_EQ(points.cols(), 0);
  ASSERT_EQ(points.rows(), 2);

  values = std::get<1>(nearest);
  ASSERT_EQ(values.size(), 0);

  nearest = rtree.nearest_within({2, 8}, 5, 4);
  points = std::get<0>(nearest);
  ASSERT_EQ(points.cols(), 0);
  ASSERT_EQ(points.rows(), 2);

  values = std::get<1>(nearest);
  ASSERT_EQ(values.size(), 0);
}

TEST(geometry_rtree, radial_basis_function) {
  auto rtree = RTree();
  rtree.packing(get_coordinates());
  using PromotionType = RTree::promotion_t;
  auto rbf =
      math::RBF<PromotionType>(std::numeric_limits<PromotionType>::quiet_NaN(),
                               0, math::RadialBasisFunction::Multiquadric);
  rtree.radial_basis_function({4, 4}, rbf, 4, 4, false);
}

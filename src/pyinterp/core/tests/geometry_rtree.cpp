// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>
#include "pyinterp/detail/geometry/rtree.hpp"

namespace geometry = pyinterp::detail::geometry;

using RTree = geometry::RTree<double, int64_t, 2>;

TEST(geometry_rtree, constructor) {
  auto rtree = RTree();
  EXPECT_TRUE(rtree.empty());
  rtree.insert(std::make_pair(geometry::PointND<double, 2>(2, 3), 0));
  rtree.insert(std::make_pair(geometry::PointND<double, 2>(5, 4), 1));
  rtree.insert(std::make_pair(geometry::PointND<double, 2>(9, 6), 2));
  rtree.insert(std::make_pair(geometry::PointND<double, 2>(4, 7), 3));
  rtree.insert(std::make_pair(geometry::PointND<double, 2>(8, 1), 4));
  rtree.insert(std::make_pair(geometry::PointND<double, 2>(7, 2), 5));
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

TEST(geometry_rtree, query) {
  auto rtree = RTree();
  // https://en.wikipedia.org/wiki/K-d_tree#/media/File:Kdtree_2d.svg
  auto coordinates =
      std::vector<RTree::value_t>{{geometry::PointND<double, 2>(2, 3), 0},
                                  {geometry::PointND<double, 2>(5, 4), 1},
                                  {geometry::PointND<double, 2>(9, 6), 2},
                                  {geometry::PointND<double, 2>(4, 7), 3},
                                  {geometry::PointND<double, 2>(8, 1), 4},
                                  {geometry::PointND<double, 2>(7, 2), 5}};
  rtree.packing(coordinates);
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

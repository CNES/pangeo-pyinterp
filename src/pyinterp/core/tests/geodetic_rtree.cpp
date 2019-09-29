// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>
#include "pyinterp/detail/geodetic/rtree.hpp"
#include "pyinterp/detail/broadcast.hpp"

namespace geodetic = pyinterp::detail::geodetic;
namespace geometry = pyinterp::detail::geometry;
namespace detail = pyinterp::detail;

struct RTree : public geodetic::RTree<double, int64_t> {
  using geodetic::RTree<double, int64_t>::RTree;

  /// The tree is created using packing algorithm (The old data is erased before
  /// construction.)
  void packing(const std::vector<geometry::EquatorialPoint3D<double>> &points,
               const std::vector<int64_t> &values) {
    detail::check_container_size("point", points, "values", values);
    std::vector<typename geometry::RTree<double, int64_t, 3>::value_t> items;
    for (size_t ix = 0; ix < points.size(); ++ix) {
      items.emplace_back(
          std::make_pair(coordinates_.lla_to_ecef(points[ix]), values[ix]));
    }
    geometry::RTree<double, int64_t, 3>::packing(items);
  }
};

TEST(geodetic, rtree) {
  std::vector<geometry::EquatorialPoint3D<double>> points;
  std::vector<int64_t> indexes;

  int64_t iz = 0;
  int64_t where;
  for (int64_t ix = -180; ix < 180; ix += 10) {
    for (int64_t iy = -90; iy < 90; iy += 10) {
      if (iy == 0 && ix == 0) {
        where = iz;
      }
      points.emplace_back(geometry::EquatorialPoint3D<double>{ix, iy});
      indexes.emplace_back(iz++);
    }
  }

  auto rtree = RTree(geodetic::System());
  EXPECT_TRUE(rtree.empty());
  rtree.packing(points, indexes);
  EXPECT_EQ(rtree.size(), iz);

  auto bounds = rtree.equatorial_bounds();
  ASSERT_TRUE(bounds);
  EXPECT_EQ(boost::geometry::get<0>(bounds->min_corner()), -180);
  EXPECT_EQ(boost::geometry::get<1>(bounds->min_corner()), -90);
  EXPECT_NEAR(boost::geometry::get<2>(bounds->min_corner()), 0, 1e-8);
  EXPECT_EQ(boost::geometry::get<0>(bounds->max_corner()), 180);
  EXPECT_EQ(boost::geometry::get<1>(bounds->max_corner()), 80);
  EXPECT_NEAR(boost::geometry::get<2>(bounds->max_corner()), 0, 1e-8);

  auto nearest = rtree.query({0, 0}, 1);
  ASSERT_TRUE(nearest.size());
  EXPECT_EQ(nearest[0].first, 0);
  EXPECT_EQ(nearest[0].second, where);

  nearest = rtree.query_ball({0, 0}, 1);
  ASSERT_TRUE(nearest.size());
  EXPECT_EQ(nearest[0].first, 0);
  EXPECT_EQ(nearest[0].second, where);

  nearest = rtree.query_ball({5, 5}, 784000);
  ASSERT_EQ(nearest.size(), 2);
  EXPECT_NEAR(nearest[0].first, 783655.977319, 1e-5);
  EXPECT_EQ(nearest[0].second, 334);
  EXPECT_NEAR(nearest[1].first, 783655.977319, 1e-5);
  EXPECT_EQ(nearest[1].second, 352);

  nearest = rtree.query_within({0, 0}, 1);
  ASSERT_TRUE(nearest.size());
  EXPECT_EQ(nearest[0].first, 0);
  EXPECT_EQ(nearest[0].second, where);

  nearest = rtree.query_within({0, 90}, 1);
  EXPECT_EQ(nearest.size(), 0);

  rtree.clear();
  EXPECT_EQ(rtree.size(), 0);
  EXPECT_TRUE(rtree.empty());
}

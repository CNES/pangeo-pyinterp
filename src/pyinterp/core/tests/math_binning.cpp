// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>
#include <boost/geometry.hpp>
#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/detail/math/binning.hpp"

namespace math = pyinterp::detail::math;
namespace geometry = pyinterp::detail::geometry;

TEST(math_binning, binning) {
  auto p = geometry::Point2D<double>{2, 2};
  auto p0 = geometry::Point2D<double>{1, 1};
  auto p1 = geometry::Point2D<double>{3, 5};

  auto weights = math::binning<geometry::Point2D, double>(p, p0, p1);
  EXPECT_EQ(std::get<0>(weights), 3.0 / 8.0);
  EXPECT_EQ(std::get<1>(weights), 1.0 / 8.0);
  EXPECT_EQ(std::get<2>(weights), 1.0 / 8.0);
  EXPECT_EQ(std::get<3>(weights), 3.0 / 8.0);

  p = geometry::Point2D<double>{4, 3};
  p0 = geometry::Point2D<double>{3, 1};
  p1 = geometry::Point2D<double>{5, 5};

  weights = math::binning<geometry::Point2D, double>(p, p0, p1);
  EXPECT_EQ(std::get<0>(weights), 2.0 / 8.0);
  EXPECT_EQ(std::get<1>(weights), 2.0 / 8.0);
  EXPECT_EQ(std::get<2>(weights), 2.0 / 8.0);
  EXPECT_EQ(std::get<3>(weights), 2.0 / 8.0);

  p = geometry::Point2D<double>{4, 4};
  p0 = geometry::Point2D<double>{1, 1};
  p1 = geometry::Point2D<double>{5, 5};

  weights = math::binning<geometry::Point2D, double>(p, p0, p1);
  EXPECT_EQ(std::get<0>(weights), 1.0 / 16.0);
  EXPECT_EQ(std::get<1>(weights), 3.0 / 16.0);
  EXPECT_EQ(std::get<2>(weights), 9.0 / 16.0);
  EXPECT_EQ(std::get<3>(weights), 3.0 / 16.0);
}

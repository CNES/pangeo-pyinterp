// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include <boost/geometry.hpp>

#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/detail/math/binning.hpp"

namespace math = pyinterp::detail::math;
namespace geometry = pyinterp::detail::geometry;

TEST(math_binning, binning_cartesian) {
  auto strategy = boost::geometry::strategy::area::cartesian<>();
  auto p = geometry::Point2D<double>{2, 2};
  auto p0 = geometry::Point2D<double>{1, 1};
  auto p1 = geometry::Point2D<double>{3, 5};

  auto weights =
      math::binning_2d<geometry::Point2D,
                       boost::geometry::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 3.0 / 8.0);
  EXPECT_EQ(std::get<1>(weights), 1.0 / 8.0);
  EXPECT_EQ(std::get<2>(weights), 1.0 / 8.0);
  EXPECT_EQ(std::get<3>(weights), 3.0 / 8.0);

  p = geometry::Point2D<double>{4, 3};
  p0 = geometry::Point2D<double>{3, 1};
  p1 = geometry::Point2D<double>{5, 5};

  weights =
      math::binning_2d<geometry::Point2D,
                       boost::geometry::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 2.0 / 8.0);
  EXPECT_EQ(std::get<1>(weights), 2.0 / 8.0);
  EXPECT_EQ(std::get<2>(weights), 2.0 / 8.0);
  EXPECT_EQ(std::get<3>(weights), 2.0 / 8.0);

  p = geometry::Point2D<double>{4, 4};
  p0 = geometry::Point2D<double>{1, 1};
  p1 = geometry::Point2D<double>{5, 5};

  weights =
      math::binning_2d<geometry::Point2D,
                       boost::geometry::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 1.0 / 16.0);
  EXPECT_EQ(std::get<1>(weights), 3.0 / 16.0);
  EXPECT_EQ(std::get<2>(weights), 9.0 / 16.0);
  EXPECT_EQ(std::get<3>(weights), 3.0 / 16.0);

  p = geometry::Point2D<double>{1, 2};
  p0 = geometry::Point2D<double>{1, 1};
  p1 = geometry::Point2D<double>{3, 5};

  weights =
      math::binning_2d<geometry::Point2D,
                       boost::geometry::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 3.0 / 4.0);
  EXPECT_EQ(std::get<1>(weights), 1.0 / 4.0);
  EXPECT_EQ(std::get<2>(weights), 0.0);
  EXPECT_EQ(std::get<3>(weights), 0.0);

  p = geometry::Point2D<double>{1, 1};
  p0 = geometry::Point2D<double>{1, 1};
  p1 = geometry::Point2D<double>{3, 5};

  weights =
      math::binning_2d<geometry::Point2D,
                       boost::geometry::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 1.0);
  EXPECT_EQ(std::get<1>(weights), 0.0);
  EXPECT_EQ(std::get<2>(weights), 0.0);
  EXPECT_EQ(std::get<3>(weights), 0.0);

  p = geometry::Point2D<double>{1, 5};
  p0 = geometry::Point2D<double>{1, 1};
  p1 = geometry::Point2D<double>{3, 5};

  weights =
      math::binning_2d<geometry::Point2D,
                       boost::geometry::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 0.0);
  EXPECT_EQ(std::get<1>(weights), 1.0);
  EXPECT_EQ(std::get<2>(weights), 0.0);
  EXPECT_EQ(std::get<3>(weights), 0.0);

  p = geometry::Point2D<double>{3, 2};
  p0 = geometry::Point2D<double>{1, 1};
  p1 = geometry::Point2D<double>{3, 5};

  weights =
      math::binning_2d<geometry::Point2D,
                       boost::geometry::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 0.0);
  EXPECT_EQ(std::get<1>(weights), 0.0);
  EXPECT_EQ(std::get<2>(weights), 1.0 / 4.0);
  EXPECT_EQ(std::get<3>(weights), 3.0 / 4.0);

  p = geometry::Point2D<double>{1.5, 1};
  p0 = geometry::Point2D<double>{1, 1};
  p1 = geometry::Point2D<double>{3, 5};

  weights =
      math::binning_2d<geometry::Point2D,
                       boost::geometry::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);

  EXPECT_EQ(std::get<0>(weights), 1.5 / 2.0);
  EXPECT_EQ(std::get<1>(weights), 0.0);
  EXPECT_EQ(std::get<2>(weights), 0);
  EXPECT_EQ(std::get<3>(weights), 0.5 / 2.0);

  p = geometry::Point2D<double>{1.5, 5};
  p0 = geometry::Point2D<double>{1, 1};
  p1 = geometry::Point2D<double>{3, 5};

  weights =
      math::binning_2d<geometry::Point2D,
                       boost::geometry::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 0);
  EXPECT_EQ(std::get<1>(weights), 1.5 / 2.0);
  EXPECT_EQ(std::get<2>(weights), 0.5 / 2.0);
  EXPECT_EQ(std::get<3>(weights), 0);
}

TEST(math_binning, binning_spheroid) {
  auto wgs84 = boost::geometry::srs::spheroid(6378137.0, 6356752.3142451793);
  auto strategy = boost::geometry::strategy::area::geographic<>(wgs84);
  auto p = geometry::GeographicPoint2D<double>{2, 2};
  auto p0 = geometry::GeographicPoint2D<double>{1, 1};
  auto p1 = geometry::GeographicPoint2D<double>{3, 5};
  auto weights =
      math::binning_2d<geometry::GeographicPoint2D,
                       boost::geometry::strategy::area::geographic<>, double>(
          p, p0, p1, strategy);

  EXPECT_NEAR(std::get<0>(weights), 0.37482309, 1e-6);
  EXPECT_NEAR(std::get<1>(weights), 0.12513892, 1e-6);
  EXPECT_NEAR(std::get<2>(weights), 0.12513892, 1e-6);
  EXPECT_NEAR(std::get<3>(weights), 0.37482309, 1e-6);

  p = geometry::GeographicPoint2D<double>{31.800000000000000711,
                                          45.695000000000000284};
  p0 = geometry::GeographicPoint2D<double>{31.700000000000081002,
                                           45.695000000000000284};
  p1 = geometry::GeographicPoint2D<double>{31.800000000000082423,
                                           45.600000000000079581};
  weights =
      math::binning_2d<geometry::GeographicPoint2D,
                       boost::geometry::strategy::area::geographic<>, double>(
          p, p0, p1, strategy);

  EXPECT_NEAR(std::get<0>(weights), 0, 1e-6);
  EXPECT_NEAR(std::get<1>(weights), 0, 1e-6);
  EXPECT_NEAR(std::get<2>(weights), 0, 1e-6);
  EXPECT_NEAR(std::get<3>(weights), 0.9999999999, 1e-6);
}

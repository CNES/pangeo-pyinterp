// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/math/interpolate/bilinear_weights.hpp"

#include <gtest/gtest.h>

#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/srs/spheroid.hpp>
#include <boost/geometry/strategy/cartesian/area.hpp>
#include <boost/geometry/strategy/geographic/area.hpp>

namespace bg = boost::geometry;
namespace pyinterp::math::interpolate {

template <typename T>
using CartesianPoint = bg::model::point<T, 2, bg::cs::cartesian>;

template <typename T>
using GeographicPoint = bg::model::point<T, 2, bg::cs::geographic<bg::degree>>;

TEST(LinearBinning, Cartesian) {
  auto strategy = bg::strategy::area::cartesian<>();
  auto p = CartesianPoint<double>{2, 2};
  auto p0 = CartesianPoint<double>{1, 1};
  auto p1 = CartesianPoint<double>{3, 5};

  auto weights =
      bilinear_weights<CartesianPoint, bg::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 3.0 / 8.0);
  EXPECT_EQ(std::get<1>(weights), 1.0 / 8.0);
  EXPECT_EQ(std::get<2>(weights), 1.0 / 8.0);
  EXPECT_EQ(std::get<3>(weights), 3.0 / 8.0);

  p = CartesianPoint<double>{4, 3};
  p0 = CartesianPoint<double>{3, 1};
  p1 = CartesianPoint<double>{5, 5};

  weights =
      bilinear_weights<CartesianPoint, bg::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 2.0 / 8.0);
  EXPECT_EQ(std::get<1>(weights), 2.0 / 8.0);
  EXPECT_EQ(std::get<2>(weights), 2.0 / 8.0);
  EXPECT_EQ(std::get<3>(weights), 2.0 / 8.0);

  p = CartesianPoint<double>{4, 4};
  p0 = CartesianPoint<double>{1, 1};
  p1 = CartesianPoint<double>{5, 5};

  weights =
      bilinear_weights<CartesianPoint, bg::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 1.0 / 16.0);
  EXPECT_EQ(std::get<1>(weights), 3.0 / 16.0);
  EXPECT_EQ(std::get<2>(weights), 9.0 / 16.0);
  EXPECT_EQ(std::get<3>(weights), 3.0 / 16.0);

  p = CartesianPoint<double>{1, 2};
  p0 = CartesianPoint<double>{1, 1};
  p1 = CartesianPoint<double>{3, 5};

  weights =
      bilinear_weights<CartesianPoint, bg::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 3.0 / 4.0);
  EXPECT_EQ(std::get<1>(weights), 1.0 / 4.0);
  EXPECT_EQ(std::get<2>(weights), 0.0);
  EXPECT_EQ(std::get<3>(weights), 0.0);

  p = CartesianPoint<double>{1, 1};
  p0 = CartesianPoint<double>{1, 1};
  p1 = CartesianPoint<double>{3, 5};

  weights =
      bilinear_weights<CartesianPoint, bg::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 1.0);
  EXPECT_EQ(std::get<1>(weights), 0.0);
  EXPECT_EQ(std::get<2>(weights), 0.0);
  EXPECT_EQ(std::get<3>(weights), 0.0);

  p = CartesianPoint<double>{1, 5};
  p0 = CartesianPoint<double>{1, 1};
  p1 = CartesianPoint<double>{3, 5};

  weights =
      bilinear_weights<CartesianPoint, bg::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 0.0);
  EXPECT_EQ(std::get<1>(weights), 1.0);
  EXPECT_EQ(std::get<2>(weights), 0.0);
  EXPECT_EQ(std::get<3>(weights), 0.0);

  p = CartesianPoint<double>{3, 2};
  p0 = CartesianPoint<double>{1, 1};
  p1 = CartesianPoint<double>{3, 5};

  weights =
      bilinear_weights<CartesianPoint, bg::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 0.0);
  EXPECT_EQ(std::get<1>(weights), 0.0);
  EXPECT_EQ(std::get<2>(weights), 1.0 / 4.0);
  EXPECT_EQ(std::get<3>(weights), 3.0 / 4.0);

  p = CartesianPoint<double>{1.5, 1};
  p0 = CartesianPoint<double>{1, 1};
  p1 = CartesianPoint<double>{3, 5};

  weights =
      bilinear_weights<CartesianPoint, bg::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);

  EXPECT_EQ(std::get<0>(weights), 1.5 / 2.0);
  EXPECT_EQ(std::get<1>(weights), 0.0);
  EXPECT_EQ(std::get<2>(weights), 0);
  EXPECT_EQ(std::get<3>(weights), 0.5 / 2.0);

  p = CartesianPoint<double>{1.5, 5};
  p0 = CartesianPoint<double>{1, 1};
  p1 = CartesianPoint<double>{3, 5};

  weights =
      bilinear_weights<CartesianPoint, bg::strategy::area::cartesian<>, double>(
          p, p0, p1, strategy);
  EXPECT_EQ(std::get<0>(weights), 0);
  EXPECT_EQ(std::get<1>(weights), 1.5 / 2.0);
  EXPECT_EQ(std::get<2>(weights), 0.5 / 2.0);
  EXPECT_EQ(std::get<3>(weights), 0);
}

TEST(LinearBinning, Spheroid) {
  auto wgs84 = bg::srs::spheroid(6378137.0, 6356752.3142451793);
  auto strategy = bg::strategy::area::geographic<>(wgs84);
  auto p = GeographicPoint<double>{2, 2};
  auto p0 = GeographicPoint<double>{1, 1};
  auto p1 = GeographicPoint<double>{3, 5};
  auto weights =
      bilinear_weights<GeographicPoint, bg::strategy::area::geographic<>,
                       double>(p, p0, p1, strategy);

  EXPECT_NEAR(std::get<0>(weights), 0.37482309, 1e-6);
  EXPECT_NEAR(std::get<1>(weights), 0.12513892, 1e-6);
  EXPECT_NEAR(std::get<2>(weights), 0.12513892, 1e-6);
  EXPECT_NEAR(std::get<3>(weights), 0.37482309, 1e-6);

  p = GeographicPoint<double>{31.800000000000000711, 45.695000000000000284};
  p0 = GeographicPoint<double>{31.700000000000081002, 45.695000000000000284};
  p1 = GeographicPoint<double>{31.800000000000082423, 45.600000000000079581};
  weights = bilinear_weights<GeographicPoint, bg::strategy::area::geographic<>,
                             double>(p, p0, p1, strategy);

  EXPECT_NEAR(std::get<0>(weights), 0, 1e-6);
  EXPECT_NEAR(std::get<1>(weights), 0, 1e-6);
  EXPECT_NEAR(std::get<2>(weights), 0, 1e-6);
  EXPECT_NEAR(std::get<3>(weights), 0.9999999999, 1e-6);
}

}  // namespace pyinterp::math::interpolate

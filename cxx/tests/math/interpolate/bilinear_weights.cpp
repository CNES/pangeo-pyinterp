// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/math/interpolate/bilinear_weights.hpp"

#include <gtest/gtest.h>

#include <array>
#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/srs/spheroid.hpp>
#include <boost/geometry/strategy/cartesian/area.hpp>
#include <boost/geometry/strategy/geographic/area.hpp>
#include <random>

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

// A point on a corner of the cell gets all the weight. One sub-polygon then
// covers the whole cell, and its area, rounded differently from the area of
// the cell, used to be discarded as too large: the point got no weight at all.
TEST(LinearBinning, CornerGetsAllTheWeight) {
  auto strategy = bg::strategy::area::cartesian<>();
  // Cells whose bounds are not exactly representable, as on a grid with a
  // step of 0.1
  for (auto ix = 0; ix < 1000; ++ix) {
    const auto x0 = -180.0 + 0.1 * ix;
    const auto y0 = -90.0 + 0.1 * (ix % 1800);
    const auto x1 = -180.0 + 0.1 * (ix + 1);
    const auto y1 = -90.0 + 0.1 * (ix % 1800 + 1);
    // Corners in the order of the weights: p00, p01, p11, p10
    const auto corners = std::array<std::array<double, 2>, 4>{
        {{x0, y0}, {x0, y1}, {x1, y1}, {x1, y0}}};
    for (size_t jx = 0; jx < 4; ++jx) {
      const auto [x, y] = corners[jx];
      const auto polygons =
          bilinear_weights<CartesianPoint, bg::strategy::area::cartesian<>,
                           double>(CartesianPoint<double>{x, y},
                                   CartesianPoint<double>{x0, y0},
                                   CartesianPoint<double>{x1, y1}, strategy);
      const auto closed_form = cartesian_bilinear_weights(x, y, x0, y0, x1, y1);
      for (size_t kx = 0; kx < 4; ++kx) {
        const auto expected = kx == jx ? 1.0 : 0.0;
        EXPECT_NEAR(polygons[kx], expected, 1e-9) << ix << " " << jx;
        EXPECT_NEAR(closed_form[kx], expected, 1e-9) << ix << " " << jx;
      }
    }
  }
}

// The closed form must match the polygon areas of the Cartesian strategy,
// including points on the edges and corners of the cell.
TEST(LinearBinning, CartesianClosedForm) {
  auto strategy = bg::strategy::area::cartesian<>();
  auto gen = std::mt19937(42);
  auto uniform = std::uniform_real_distribution<>(0.0, 1.0);

  for (auto ix = 0; ix < 10'000; ++ix) {
    const auto x0 = 360.0 * uniform(gen) - 180.0;
    const auto y0 = 180.0 * uniform(gen) - 90.0;
    const auto dx = 0.01 + 2.0 * uniform(gen);
    const auto dy = 0.01 + 2.0 * uniform(gen);
    // One point in four snapped to an edge or a corner
    auto fx = uniform(gen);
    auto fy = uniform(gen);
    if (ix % 4 == 0) {
      fx = static_cast<double>(ix % 3) / 2.0;
      fy = static_cast<double>((ix / 3) % 3) / 2.0;
    }
    const auto x = x0 + fx * dx;
    const auto y = y0 + fy * dy;

    const auto expected =
        bilinear_weights<CartesianPoint, bg::strategy::area::cartesian<>,
                         double>(
            CartesianPoint<double>{x, y}, CartesianPoint<double>{x0, y0},
            CartesianPoint<double>{x0 + dx, y0 + dy}, strategy);
    const auto actual =
        cartesian_bilinear_weights(x, y, x0, y0, x0 + dx, y0 + dy);
    // The shoelace formula of the polygon areas loses about 1e-12 to
    // cancellation with coordinates near 180; the closed form works on
    // differences and is the more accurate of the two.
    for (size_t jx = 0; jx < 4; ++jx) {
      EXPECT_NEAR(actual[jx], expected[jx], 1e-9) << ix << " " << jx;
    }
  }
}

}  // namespace pyinterp::math::interpolate

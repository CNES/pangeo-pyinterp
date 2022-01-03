// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include <boost/geometry.hpp>

#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/detail/math/bivariate.hpp"

namespace math = pyinterp::detail::math;
namespace geometry = pyinterp::detail::geometry;

TEST(math_bivariate, bilinear) {
  /// https://en.wikipedia.org/wiki/Bilinear_interpolation
  auto interpolator = math::Bilinear<geometry::Point2D, double>();

  EXPECT_DOUBLE_EQ(interpolator.evaluate(geometry::Point2D<double>{14.5, 20.2},
                                         geometry::Point2D<double>{14.0, 21.0},
                                         geometry::Point2D<double>{15.0, 20.0},
                                         162.0, 91.0, 95.0, 210.0),
                   146.1);
  EXPECT_DOUBLE_EQ(interpolator.evaluate(geometry::Point2D<double>{14.5, 20.0},
                                         geometry::Point2D<double>{14.0, 21.0},
                                         geometry::Point2D<double>{15.0, 20.0},
                                         162.0, 91.0, 95.0, 210.0),
                   150.5);
  EXPECT_DOUBLE_EQ(interpolator.evaluate(geometry::Point2D<double>{14.5, 21.0},
                                         geometry::Point2D<double>{14.0, 21.0},
                                         geometry::Point2D<double>{15.0, 20.0},
                                         162.0, 91.0, 95.0, 210.0),
                   128.5);
}

TEST(math_bivariate, nearest) {
  auto interpolator = math::Nearest<geometry::Point2D, double>();

  EXPECT_DOUBLE_EQ(interpolator.evaluate(geometry::Point2D<double>{14.4, 20.2},
                                         geometry::Point2D<double>{14.0, 21.0},
                                         geometry::Point2D<double>{15.0, 20.0},
                                         162.0, 91.0, 95.0, 210.0),
                   91);
  EXPECT_DOUBLE_EQ(interpolator.evaluate(geometry::Point2D<double>{14.5, 20.0},
                                         geometry::Point2D<double>{14.0, 21.0},
                                         geometry::Point2D<double>{15.0, 20.0},
                                         162.0, 91.0, 95.0, 210.0),
                   91);
  EXPECT_DOUBLE_EQ(interpolator.evaluate(geometry::Point2D<double>{14.9, 20.0},
                                         geometry::Point2D<double>{14.0, 21.0},
                                         geometry::Point2D<double>{15.0, 20.0},
                                         162.0, 91.0, 95.0, 210.0),
                   210);
  EXPECT_DOUBLE_EQ(interpolator.evaluate(geometry::Point2D<double>{14.1, 20.9},
                                         geometry::Point2D<double>{14.0, 21.0},
                                         geometry::Point2D<double>{15.0, 20.0},
                                         162.0, 91.0, 95.0, 210.0),
                   162);
  EXPECT_DOUBLE_EQ(interpolator.evaluate(geometry::Point2D<double>{14.9, 21.0},
                                         geometry::Point2D<double>{14.0, 21.0},
                                         geometry::Point2D<double>{15.0, 20.0},
                                         162.0, 91.0, 95.0, 210.0),
                   95);
}

TEST(math_bivariate, idw) {
  auto interpolator =
      math::InverseDistanceWeighting<geometry::Point2D, double>();

  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(geometry::Point2D<double>{0, 0},
                            geometry::Point2D<double>{0, 0},
                            geometry::Point2D<double>{1, 1}, 0, 1, 2, 3),
      0);
  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(geometry::Point2D<double>{0, 1},
                            geometry::Point2D<double>{0, 0},
                            geometry::Point2D<double>{1, 1}, 0, 1, 2, 3),
      1);
  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(geometry::Point2D<double>{1, 0},
                            geometry::Point2D<double>{0, 0},
                            geometry::Point2D<double>{1, 1}, 0, 1, 2, 3),
      2);
  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(geometry::Point2D<double>{1, 1},
                            geometry::Point2D<double>{0, 0},
                            geometry::Point2D<double>{1, 1}, 0, 1, 2, 3),
      3);
  // 1.5 = 6d / 4d where d is the distance between the point P(0.5, 0.5) and all
  // other points
  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(geometry::Point2D<double>{0.5, 0.5},
                            geometry::Point2D<double>{0, 0},
                            geometry::Point2D<double>{1, 1}, 0, 1, 2, 3),
      1.5);
}

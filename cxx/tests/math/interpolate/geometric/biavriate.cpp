// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <gtest/gtest.h>

#include <boost/geometry.hpp>
#include <boost/geometry/core/access.hpp>
#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/point.hpp>

#include "pyinterp/math/interpolate/geometric/bivariate.hpp"

namespace pyinterp::math::interpolate::geometric {

// Point alias compatible with header (template< class >)
template <class T>
using Point2D =
    boost::geometry::model::point<T, 2, boost::geometry::cs::cartesian>;

TEST(GeometryBivariate, BilinearBasic) {
  Bilinear<Point2D, double> bilinear;
  // Legacy examples
  EXPECT_DOUBLE_EQ(bilinear.evaluate(
                       Point2D<double>{14.5, 20.2}, Point2D<double>{14.0, 21.0},
                       Point2D<double>{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
                   146.1);
  EXPECT_DOUBLE_EQ(bilinear.evaluate(
                       Point2D<double>{14.5, 20.0}, Point2D<double>{14.0, 21.0},
                       Point2D<double>{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
                   150.5);
  EXPECT_DOUBLE_EQ(bilinear.evaluate(
                       Point2D<double>{14.5, 21.0}, Point2D<double>{14.0, 21.0},
                       Point2D<double>{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
                   128.5);
}

TEST(GeometryBivariate, BilinearCorners) {
  Bilinear<Point2D, double> bilinear;
  auto p0 = Point2D<double>{0.0, 0.0};
  auto p1 = Point2D<double>{1.0, 1.0};
  double q00 = 10, q01 = 20, q10 = 30, q11 = 40;
  EXPECT_DOUBLE_EQ(
      bilinear.evaluate(Point2D<double>{0.0, 0.0}, p0, p1, q00, q01, q10, q11),
      q00);
  EXPECT_DOUBLE_EQ(
      bilinear.evaluate(Point2D<double>{0.0, 1.0}, p0, p1, q00, q01, q10, q11),
      q01);
  EXPECT_DOUBLE_EQ(
      bilinear.evaluate(Point2D<double>{1.0, 0.0}, p0, p1, q00, q01, q10, q11),
      q10);
  EXPECT_DOUBLE_EQ(
      bilinear.evaluate(Point2D<double>{1.0, 1.0}, p0, p1, q00, q01, q10, q11),
      q11);
}

TEST(GeometryBivariate, BilinearCenterAverage) {
  Bilinear<Point2D, double> bilinear;
  auto p0 = Point2D<double>{0.0, 0.0};
  auto p1 = Point2D<double>{2.0, 2.0};
  double q00 = 1, q01 = 2, q10 = 3, q11 = 6;
  // Center (1,1) -> t = u = 0.5
  double expected = 0.25 * q00 + 0.25 * q10 + 0.25 * q01 + 0.25 * q11;
  EXPECT_DOUBLE_EQ(
      bilinear.evaluate(Point2D<double>{1.0, 1.0}, p0, p1, q00, q01, q10, q11),
      expected);
}

TEST(GeometryBivariate, NearestBasic) {
  Nearest<Point2D, double> nearest;
  EXPECT_DOUBLE_EQ(
      nearest.evaluate(Point2D<double>{14.4, 20.2}, Point2D<double>{14.0, 21.0},
                       Point2D<double>{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
      91);
  EXPECT_DOUBLE_EQ(
      nearest.evaluate(Point2D<double>{14.5, 20.0}, Point2D<double>{14.0, 21.0},
                       Point2D<double>{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
      91);
  EXPECT_DOUBLE_EQ(
      nearest.evaluate(Point2D<double>{14.9, 20.0}, Point2D<double>{14.0, 21.0},
                       Point2D<double>{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
      210);
  EXPECT_DOUBLE_EQ(
      nearest.evaluate(Point2D<double>{14.1, 20.9}, Point2D<double>{14.0, 21.0},
                       Point2D<double>{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
      162);
  EXPECT_DOUBLE_EQ(
      nearest.evaluate(Point2D<double>{14.9, 21.0}, Point2D<double>{14.0, 21.0},
                       Point2D<double>{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
      95);
}

TEST(GeometryBivariate, NearestCenterTie) {
  Nearest<Point2D, double> nearest;
  auto p0 = Point2D<double>{0.0, 0.0};
  auto p1 = Point2D<double>{1.0, 1.0};
  // Equal distance to all four corners
  double val =
      nearest.evaluate(Point2D<double>{0.5, 0.5}, p0, p1, 10, 20, 30, 40);
  // Implementation order returns first corner value (10)
  EXPECT_DOUBLE_EQ(val, 10);
}

TEST(GeometryBivariate, IdwBasicExactCorners) {
  InverseDistanceWeighting<Point2D, double> idw;
  auto p0 = Point2D<double>{0.0, 0.0};
  auto p1 = Point2D<double>{1.0, 1.0};
  EXPECT_DOUBLE_EQ(idw.evaluate(Point2D<double>{0.0, 0.0}, p0, p1, 0, 1, 2, 3),
                   0);
  EXPECT_DOUBLE_EQ(idw.evaluate(Point2D<double>{0.0, 1.0}, p0, p1, 0, 1, 2, 3),
                   1);
  EXPECT_DOUBLE_EQ(idw.evaluate(Point2D<double>{1.0, 0.0}, p0, p1, 0, 1, 2, 3),
                   2);
  EXPECT_DOUBLE_EQ(idw.evaluate(Point2D<double>{1.0, 1.0}, p0, p1, 0, 1, 2, 3),
                   3);
}

TEST(GeometryBivariate, IdwCenterExp2) {
  InverseDistanceWeighting<Point2D, double> idw(2);
  auto p0 = Point2D<double>{0.0, 0.0};
  auto p1 = Point2D<double>{1.0, 1.0};
  // Center point -> symmetric weights -> average of four values (0+1+2+3)/4=1.5
  EXPECT_DOUBLE_EQ(idw.evaluate(Point2D<double>{0.5, 0.5}, p0, p1, 0, 1, 2, 3),
                   1.5);
}

TEST(GeometryBivariate, IdwEqualValuesInvariant) {
  InverseDistanceWeighting<Point2D, double> idw(3);
  auto p0 = Point2D<double>{0.0, 0.0};
  auto p1 = Point2D<double>{2.0, 2.0};
  double result = idw.evaluate(Point2D<double>{1.3, 0.7}, p0, p1, 5, 5, 5, 5);
  EXPECT_DOUBLE_EQ(result, 5.0);
}

TEST(GeometryBivariate, IdwExponentInfluence) {
  auto p0 = Point2D<double>{0.0, 0.0};
  auto p1 = Point2D<double>{1.0, 1.0};
  InverseDistanceWeighting<Point2D, double> idw1(1);
  InverseDistanceWeighting<Point2D, double> idw8(8);
  auto query = Point2D<double>{0.05, 0.05};  // very close to q00
  double r1 = idw1.evaluate(query, p0, p1, 100.0, 0.0, 0.0, 0.0);
  double r8 = idw8.evaluate(query, p0, p1, 100.0, 0.0, 0.0, 0.0);
  EXPECT_GT(r8, r1);             // higher exponent concentrates weight
  EXPECT_NEAR(r8, 100.0, 1e-6);  // Very close to nearest value
}

TEST(GeometryBivariate, IdwEpsilonCornerSelection) {
  // Query exactly matches a corner -> immediate return corner value
  InverseDistanceWeighting<Point2D, double> idw;
  auto p0 = Point2D<double>{10.0, 20.0};
  auto p1 = Point2D<double>{11.0, 21.0};
  EXPECT_DOUBLE_EQ(
      idw.evaluate(Point2D<double>{10.0, 20.0}, p0, p1, 7, 8, 9, 10), 7);
}

TEST(GeometryBivariate, Factory) {
  auto bilinear_ptr =
      make_interpolator<Point2D, double>(InterpolationMethod::kBilinear);
  auto idw_ptr = make_interpolator<Point2D, double>(
      InterpolationMethod::kInverseDistanceWeighting, 3);
  auto nearest_ptr =
      make_interpolator<Point2D, double>(InterpolationMethod::kNearest);

  auto p0 = Point2D<double>{0.0, 0.0};
  auto p1 = Point2D<double>{1.0, 1.0};
  auto q00 = 0.0;
  auto q01 = 1.0;
  auto q10 = 2.0;
  auto q11 = 3.0;

  // Bilinear center
  EXPECT_DOUBLE_EQ(bilinear_ptr->evaluate(Point2D<double>{0.5, 0.5}, p0, p1,
                                          q00, q01, q10, q11),
                   1.5);

  // IDW corner
  EXPECT_DOUBLE_EQ(
      idw_ptr->evaluate(Point2D<double>{0.0, 1.0}, p0, p1, q00, q01, q10, q11),
      1.0);

  // Nearest center tie -> q00
  EXPECT_DOUBLE_EQ(nearest_ptr->evaluate(Point2D<double>{0.5, 0.5}, p0, p1, q00,
                                         q01, q10, q11),
                   0.0);
}

TEST(GeometryBivariate, MakePointHelper) {
  auto p = make_point<Point2D, double>(3.14, 2.72);
  EXPECT_DOUBLE_EQ(boost::geometry::get<0>(p), 3.14);
  EXPECT_DOUBLE_EQ(boost::geometry::get<1>(p), 2.72);
}

}  // namespace pyinterp::math::interpolate::geometric

#include "pyinterp/detail/math/bivariate.hpp"
#include <boost/geometry.hpp>
#include <gtest/gtest.h>

namespace math = pyinterp::detail::math;

TEST(math_bivariate, bilinear) {
  /// https://en.wikipedia.org/wiki/Bilinear_interpolation
  using Point =
      boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian>;
  auto interpolator = math::Bilinear<Point, double>();

  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(Point{14.5, 20.2}, Point{14.0, 21.0},
                            Point{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
      146.1);
  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(Point{14.5, 20.0}, Point{14.0, 21.0},
                            Point{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
      150.5);
  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(Point{14.5, 21.0}, Point{14.0, 21.0},
                            Point{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
      128.5);
}

TEST(math_bivariate, nearest) {
  using Point =
      boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian>;
  auto interpolator = math::Nearest<Point, double>();

  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(Point{14.4, 20.2}, Point{14.0, 21.0},
                            Point{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
      91);
  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(Point{14.5, 20.0}, Point{14.0, 21.0},
                            Point{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
      91);
  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(Point{14.9, 20.0}, Point{14.0, 21.0},
                            Point{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
      210);
  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(Point{14.1, 20.9}, Point{14.0, 21.0},
                            Point{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
      162);
  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(Point{14.9, 21.0}, Point{14.0, 21.0},
                            Point{15.0, 20.0}, 162.0, 91.0, 95.0, 210.0),
      95);
}

TEST(math_bivariate, idw) {
  using Point =
      boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian>;
  auto interpolator = math::InverseDistanceWeigthing<Point, double>();

  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(Point{0, 0}, Point{0, 0}, Point{1, 1}, 0, 1, 2, 3),
      0);
  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(Point{0, 1}, Point{0, 0}, Point{1, 1}, 0, 1, 2, 3),
      1);
  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(Point{1, 0}, Point{0, 0}, Point{1, 1}, 0, 1, 2, 3),
      2);
  EXPECT_DOUBLE_EQ(
      interpolator.evaluate(Point{1, 1}, Point{0, 0}, Point{1, 1}, 0, 1, 2, 3),
      3);
  // 1.5 = 6d / 4d where d is the distance between the point P(0.5, 0.5) and all
  // other points
  EXPECT_DOUBLE_EQ(interpolator.evaluate(Point{0.5, 0.5}, Point{0, 0},
                                         Point{1, 1}, 0, 1, 2, 3),
                   1.5);
}

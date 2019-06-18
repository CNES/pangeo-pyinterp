#include "pyinterp/detail/math/trivariate.hpp"
#include <boost/geometry.hpp>
#include <gtest/gtest.h>

namespace math = pyinterp::detail::math;

TEST(math_trivariate, trivariate) {
  using Point =
      boost::geometry::model::point<double, 3, boost::geometry::cs::cartesian>;
  auto bilinear = math::Bilinear<Point, double>();
  auto trilinear = math::Trivariate<Point, double>();

  EXPECT_DOUBLE_EQ(
      bilinear.evaluate(Point{14.5, 20.2}, Point{14.0, 21.0}, Point{15.0, 20.0},
                        162.0, 91.0, 95.0, 210.0),
      146.1);
  EXPECT_DOUBLE_EQ(
      bilinear.evaluate(Point{14.5, 20.2}, Point{14.0, 21.0}, Point{15.0, 20.0},
                        262.0, 191.0, 195.0, 310.0),
      246.1);
  EXPECT_DOUBLE_EQ(
      trilinear.evaluate(Point{14.5, 20.2, 0.5}, Point{14.0, 21.0, 0},
                         Point{15.0, 20.0, 1}, 162.0, 91.0, 95.0, 210.0, 262.0,
                         191.0, 195.0, 310.0, &bilinear),
      (146.1 + 246.1) * 0.5);
}
#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/detail/math/trivariate.hpp"
#include <boost/geometry.hpp>
#include <gtest/gtest.h>

namespace math = pyinterp::detail::math;
namespace geometry = pyinterp::detail::geometry;

TEST(math_trivariate, trivariate) {
  auto bilinear = math::Bilinear<geometry::Point3D, double>();
  auto trilinear = math::Trivariate<geometry::Point3D, double>();

  EXPECT_DOUBLE_EQ(bilinear.evaluate(geometry::Point3D<double>{14.5, 20.2},
                                     geometry::Point3D<double>{14.0, 21.0},
                                     geometry::Point3D<double>{15.0, 20.0},
                                     162.0, 91.0, 95.0, 210.0),
                   146.1);
  EXPECT_DOUBLE_EQ(bilinear.evaluate(geometry::Point3D<double>{14.5, 20.2},
                                     geometry::Point3D<double>{14.0, 21.0},
                                     geometry::Point3D<double>{15.0, 20.0},
                                     262.0, 191.0, 195.0, 310.0),
                   246.1);
  EXPECT_DOUBLE_EQ(
      trilinear.evaluate(geometry::Point3D<double>{14.5, 20.2, 0.5},
                         geometry::Point3D<double>{14.0, 21.0, 0},
                         geometry::Point3D<double>{15.0, 20.0, 1}, 162.0, 91.0,
                         95.0, 210.0, 262.0, 191.0, 195.0, 310.0, &bilinear),
      (146.1 + 246.1) * 0.5);
}
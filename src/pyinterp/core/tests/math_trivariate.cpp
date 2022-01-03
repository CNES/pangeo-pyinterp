// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include <boost/geometry.hpp>

#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/detail/math/trivariate.hpp"

namespace math = pyinterp::detail::math;
namespace geometry = pyinterp::detail::geometry;

TEST(math_trivariate, trivariate) {
  auto bilinear = math::Bilinear<geometry::Point3D, double>();

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
  auto interpolated = math::trivariate<geometry::Point3D, double>(
      geometry::Point3D<double>{14.5, 20.2, 0.5},
      geometry::Point3D<double>{14.0, 21.0, 0},
      geometry::Point3D<double>{15.0, 20.0, 1}, 162.0, 91.0, 95.0, 210.0, 262.0,
      191.0, 195.0, 310.0, &bilinear);
  EXPECT_DOUBLE_EQ(interpolated, (146.1 + 246.1) * 0.5);

  interpolated = math::trivariate<geometry::Point3D, double>(
      geometry::Point3D<double>{14.5, 20.2, 0.4},
      geometry::Point3D<double>{14.0, 21.0, 0},
      geometry::Point3D<double>{15.0, 20.0, 1}, 162.0, 91.0, 95.0, 210.0, 262.0,
      191.0, 195.0, 310.0, &bilinear, &math::nearest<double, double>);
  EXPECT_DOUBLE_EQ(interpolated, 146.1);

  interpolated = math::trivariate<geometry::Point3D, double>(
      geometry::Point3D<double>{14.5, 20.2, 0.6},
      geometry::Point3D<double>{14.0, 21.0, 0},
      geometry::Point3D<double>{15.0, 20.0, 1}, 162.0, 91.0, 95.0, 210.0, 262.0,
      191.0, 195.0, 310.0, &bilinear, &math::nearest<double, double>);
  EXPECT_DOUBLE_EQ(interpolated, 246.1);

  interpolated = math::trivariate<geometry::Point3D, double>(
      geometry::Point3D<double>{14.5, 20.2, 0.6},
      geometry::Point3D<double>{14.0, 21.0, 0},
      geometry::Point3D<double>{15.0, 20.0, 1}, 162.0, 91.0, 95.0, 210.0, 262.0,
      191.0, 195.0, 310.0, &bilinear, &math::linear<double, double>);
  EXPECT_DOUBLE_EQ(interpolated, 206.1);
}

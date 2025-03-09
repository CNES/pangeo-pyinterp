// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include "pyinterp/detail/math/kriging.hpp"

namespace math = pyinterp::detail::math;

TEST(MathLoess, Loess) {
  Eigen::MatrixXd coordinates(3, 4);
  Eigen::VectorXd values(4);
  Eigen::VectorXd query(3);

  coordinates(0, 0) = 0;
  coordinates(1, 0) = 0;
  coordinates(2, 0) = 0;
  coordinates(0, 1) = 1;
  coordinates(1, 1) = 1;
  coordinates(2, 1) = 1;
  coordinates(0, 2) = 2;
  coordinates(1, 2) = 1;
  coordinates(2, 2) = 2;
  coordinates(0, 3) = 3;
  coordinates(1, 3) = 0;
  coordinates(2, 3) = 1;
  values << 0, 1, 2, 1;
  query << 1.5, 0.5, 1;

  auto interpolator = math::Kriging<double>(1, 0.25, math::kMatern_52);

  auto x = interpolator.universal_kriging(coordinates, values, query);
  EXPECT_NEAR(x, 0.0388, 1e-3);
}

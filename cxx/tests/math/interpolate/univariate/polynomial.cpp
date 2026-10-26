// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/math/interpolate/univariate/polynomial.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>

namespace pyinterp::math::interpolate::univariate {

TEST(Polynomial, Interpolate) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(6);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(6);

  xa << 0.0, 1.0, 2.0, 3.0;
  ya << 0.0, 1.0, 2.0, 3.0;

  xp << 0.0, 0.5, 1.0, 1.5, 2.5, 3.0;
  yp << 0.0, 0.5, 1.0, 1.5, 2.5, 3.0;

  auto interpolator = Polynomial<double>();
  auto y = interpolator(xa, ya, xp);
  for (auto ix = 0; ix < xp.size(); ix++) {
    EXPECT_DOUBLE_EQ(y(ix), yp(ix));
  }
}

TEST(Polynomial, Derivative) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(6);
  Eigen::Matrix<double, Eigen::Dynamic, 1> dyp(6);

  xa << 0.0, 1.0, 2.0, 3.0;
  ya << 0.0, 1.0, 2.0, 3.0;

  xp << 0.0, 0.5, 1.0, 1.5, 2.5, 3.0;
  dyp << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;

  auto interpolator = Polynomial<double>();
  auto dy = interpolator.derivative(xa, ya, xp);
  for (auto ix = 0; ix < xp.size(); ix++) {
    EXPECT_NEAR(dy(ix), dyp(ix), 1e-4);
  }
}

// A polynomial through n points reproduces any polynomial of degree < n.
// With more than 4 points, the divided differences used to read values
// already overwritten.
TEST(Polynomial, ReproducesCubic) {
  const auto cubic = [](const double x) {
    return 0.3 * x * x * x - 2.0 * x * x + x - 5.0;
  };
  for (const auto size : {4, 5, 6, 8, 10}) {
    Eigen::VectorXd xa(size);
    Eigen::VectorXd ya(size);
    for (auto ix = 0; ix < size; ++ix) {
      xa(ix) = 0.5 * ix + 0.1 * ix * ix;  // Irregular abscissas
      ya(ix) = cubic(xa(ix));
    }
    auto interpolator = Polynomial<double>();
    for (auto x = xa(0); x <= xa(size - 1); x += 0.05) {
      EXPECT_NEAR(interpolator(xa, ya, x), cubic(x), 1e-9)
          << "size=" << size << " x=" << x;
    }
  }
}

}  // namespace pyinterp::math::interpolate::univariate

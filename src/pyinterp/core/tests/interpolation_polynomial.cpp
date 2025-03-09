// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include "pyinterp/detail/interpolation/polynomial.hpp"

TEST(Polynomial, Interpolate) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(6);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(6);

  xa << 0.0, 1.0, 2.0, 3.0;
  ya << 0.0, 1.0, 2.0, 3.0;

  xp << 0.0, 0.5, 1.0, 1.5, 2.5, 3.0;
  yp << 0.0, 0.5, 1.0, 1.5, 2.5, 3.0;

  auto interpolator = pyinterp::detail::interpolation::Polynomial<double>();
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

  auto interpolator = pyinterp::detail::interpolation::Polynomial<double>();
  auto dy = interpolator.derivative(xa, ya, xp);
  for (auto ix = 0; ix < xp.size(); ix++) {
    EXPECT_NEAR(dy(ix), dyp(ix), 1e-4);
  }
}

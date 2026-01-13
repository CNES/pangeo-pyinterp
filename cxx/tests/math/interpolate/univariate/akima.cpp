// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/math/interpolate/univariate/akima.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "pyinterp/math/interpolate/univariate/akima_periodic.hpp"

namespace pyinterp::math::interpolate::univariate {

TEST(Akima, Interpolate) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(5);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(5);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(4);

  xa << 0.0, 1.0, 2.0, 3.0, 4.0;
  ya << 0.0, 1.0, 2.0, 3.0, 4.0;

  xp << 0.0, 0.5, 1.0, 2.0;
  yp << 0.0, 0.5, 1.0, 2.0;

  auto interpolator = Akima<double>();
  auto y = interpolator(xa, ya, xp);
  for (auto ix = 0; ix < xp.size(); ix++) {
    EXPECT_DOUBLE_EQ(y(ix), yp(ix));
  }
}

TEST(Akima, Derivative) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(5);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(5);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> dyp(4);

  xa << 0.0, 1.0, 2.0, 3.0, 4.0;
  ya << 0.0, 1.0, 2.0, 3.0, 4.0;

  xp << 0.0, 0.5, 1.0, 2.0;
  dyp << 1.0, 1.0, 1.0, 1.0;

  auto interpolator = Akima<double>();
  auto dy = interpolator.derivative(xa, ya, xp);
  for (auto ix = 0; ix < xp.size(); ix++) {
    EXPECT_DOUBLE_EQ(dy(ix), dyp(ix));
  }
}

TEST(AkimaPeriodic, Interpolate) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(5);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(5);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(4);

  xa << 0.0, 1.0, 2.0, 3.0, 4.0;
  ya << 0.0, 1.0, 2.0, 3.0, 4.0;

  xp << 0.0, 0.5, 1.0, 2.0;
  yp << 0.0, 0.5, 1.0, 2.0;

  auto interpolator = AkimaPeriodic<double>();
  auto y = interpolator(xa, ya, xp);
  for (auto ix = 0; ix < xp.size(); ix++) {
    EXPECT_DOUBLE_EQ(y(ix), yp(ix));
  }
}

}  // namespace pyinterp::math::interpolate::univariate

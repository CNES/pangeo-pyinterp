// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include "pyinterp/detail/interpolation/bilinear.hpp"

TEST(Bilinear, Symmetric) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> za(4, 4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(6);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(6);
  Eigen::Matrix<double, Eigen::Dynamic, 1> zp(6);

  xa << 0.0, 1.0, 2.0, 3.0;
  ya << 0.0, 1.0, 2.0, 3.0;
  za << 1.0, 1.1, 1.2, 1.3, 1.1, 1.2, 1.3, 1.4, 1.2, 1.3, 1.4, 1.5, 1.3, 1.4,
      1.5, 1.6;
  xp << 0.0, 0.5, 1.0, 1.5, 2.5, 3.0;
  yp << 0.0, 0.5, 1.0, 1.5, 2.5, 3.0;
  zp << 1.0, 1.1, 1.2, 1.3, 1.5, 1.6;

  auto bilinear = pyinterp::detail::interpolation::Bilinear<double>();
  auto z = bilinear(xa, ya, za, xp, yp);
  for (Eigen::Index i = 0; i < z.size(); ++i) {
    EXPECT_NEAR(z(i), zp(i), 1.0e-12);
  }
}

TEST(Bilinear, Asymmetric) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> za(4, 4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(12);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(12);
  Eigen::Matrix<double, Eigen::Dynamic, 1> zp(12);

  xa << 0.0, 1.0, 2.0, 3.0;
  ya << 0.0, 1.0, 2.0, 3.0;
  za << 1.0, 1.3, 1.5, 1.6, 1.1, 1.4, 1.6, 1.9, 1.2, 1.5, 1.7, 2.2, 1.4, 1.7,
      1.9, 2.3;
  xp << 0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 1.3954, 1.6476, 0.824957, 2.41108,
      2.98619, 1.36485;
  yp << 0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 0.265371, 2.13849, 1.62114, 1.22198,
      0.724681, 0.0596087;
  zp << 1.0, 1.2, 1.4, 1.55, 2.025, 2.3, 1.2191513, 1.7242442248, 1.5067237,
      1.626612, 1.6146423, 1.15436761;

  auto bilinear = pyinterp::detail::interpolation::Bilinear<double>();
  auto z = bilinear(xa, ya, za, xp, yp);
  for (Eigen::Index i = 0; i < z.size(); ++i) {
    EXPECT_NEAR(z(i), zp(i), 1.0e-12);
  }
}

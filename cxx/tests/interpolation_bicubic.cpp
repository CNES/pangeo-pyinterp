// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include "pyinterp/detail/interpolation/bicubic.hpp"

TEST(Bicubic, Caseone) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> za(4, 4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(4);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(3);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(3);
  Eigen::Matrix<double, Eigen::Dynamic, 1> zp(3);

  xa << 0.0, 1.0, 2.0, 3.0;
  ya << 0.0, 1.0, 2.0, 3.0;
  za << 1.0, 1.1, 1.2, 1.3, 1.1, 1.2, 1.3, 1.4, 1.2, 1.3, 1.4, 1.5, 1.3, 1.4,
      1.5, 1.6;
  xp << 1.0, 1.5, 2.0;
  yp << 1.0, 1.5, 2.0;
  zp << 1.2, 1.3, 1.4;
  auto bicubic = pyinterp::detail::interpolation::Bicubic<double>();
  auto z = bicubic(xa, ya, za, xp, yp);
  for (Eigen::Index i = 0; i < z.size(); ++i) {
    EXPECT_NEAR(z(i), zp(i), 1.0e-12);
  }
}

TEST(Bicubic, Nonlinear) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> za(8, 8);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(8);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(8);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(7);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(7);
  Eigen::Matrix<double, Eigen::Dynamic, 1> zp(7);

  xa << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;
  ya << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;
  za << 1, 2, 3, 4, 5, 6, 7, 8, 2, 2, 6, 4, 10, 6, 14, 8, 3, 6, 3, 12, 15, 6,
      21, 24, 4, 4, 12, 4, 20, 12, 28, 8, 5, 10, 15, 20, 5, 30, 35, 40, 6, 6, 6,
      12, 30, 6, 42, 24, 7, 14, 21, 28, 35, 42, 7, 56, 8, 8, 24, 8, 40, 24, 56,
      8;
  xp << 1.4, 2.3, 4.7, 3.3, 7.5, 6.6, 5.1;
  yp << 1.0, 1.8, 1.9, 2.5, 2.7, 4.1, 3.3;
  zp << 1.4, 3.11183531264736, 8.27114315792559, 5.03218982537718,
      22.13230634702637, 23.63206834997871, 17.28553080971182;
  auto bicubic = pyinterp::detail::interpolation::Bicubic<double>();
  auto z = bicubic(xa, ya, za, xp, yp);
  for (Eigen::Index i = 0; i < z.size(); ++i) {
    EXPECT_NEAR(z(i), zp(i), 1.0e-12);
  }
}

TEST(Bicubic, Nonsquare) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> za(10, 8);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xa(10);
  Eigen::Matrix<double, Eigen::Dynamic, 1> ya(8);
  Eigen::Matrix<double, Eigen::Dynamic, 1> xp(7);
  Eigen::Matrix<double, Eigen::Dynamic, 1> yp(7);
  Eigen::Matrix<double, Eigen::Dynamic, 1> zp(7);

  xa << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;
  ya << 1.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0;
  za << 1, 2, 3, 4, 5, 6, 7, 8, 2, 2, 6, 4, 10, 6, 14, 8, 3, 6, 3, 12, 15, 6,
      21, 24, 4, 4, 12, 4, 20, 12, 28, 8, 5, 10, 15, 20, 5, 30, 35, 40, 6, 6, 6,
      12, 30, 6, 42, 24, 7, 14, 21, 28, 35, 42, 7, 56, 8, 8, 24, 8, 40, 24, 56,
      8, 9, 11, 13, 15, 17, 19, 21, 23, 10, 12, 14, 16, 18, 20, 22, 24;
  xp << 1.4, 2.3, 9.7, 3.3, 9.5, 6.6, 5.1;
  yp << 1.0, 1.8, 1.9, 2.5, 2.7, 4.1, 3.3;
  zp << 1.4, 2.46782030941187003, 10.7717721621846465, 4.80725067958096375,
      11.6747032398627297, 11.2619968682970111, 9.00168877916872567;

  auto bicubic = pyinterp::detail::interpolation::Bicubic<double>();
  auto z = bicubic(xa, ya, za, xp, yp);
  for (Eigen::Index i = 0; i < z.size(); ++i) {
    EXPECT_NEAR(z(i), zp(i), 1.0e-12);
  }
}

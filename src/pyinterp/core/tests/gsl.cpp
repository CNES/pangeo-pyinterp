// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include "pyinterp/detail/gsl/error_handler.hpp"
#include "pyinterp/detail/gsl/interpolate1d.hpp"
#include "pyinterp/detail/gsl/interpolate2d.hpp"

namespace gsl = pyinterp::detail::gsl;

TEST(gsl, interpolate) {
  // Test copied from a unit test of the GSL library.
  gsl::set_error_handler();

  auto data_x = std::vector<double>{0.0, 1.0, 2.0};
  auto data_y = std::vector<double>{0.0, 1.0, 2.0};
  auto test_x = std::vector<double>{0.0, 0.5, 1.0, 2.0};
  auto test_y = std::vector<double>{0.0, 0.5, 1.0, 2.0};
  auto test_dy = std::vector<double>{1.0, 1.0, 1.0, 1.0};
  auto test_iy = std::vector<double>{0.0, 0.125, 0.5, 2.0};

  auto interpolator =
      gsl::Interpolate1D(data_x.size(), gsl_interp_cspline, gsl::Accelerator());

  for (auto ix = 0; ix < 4; ix++) {
    auto x = test_x[ix];
    EXPECT_NEAR(
        interpolator.interpolate(
            Eigen::Map<Eigen::VectorXd>(data_x.data(), data_x.size()),
            Eigen::Map<Eigen::VectorXd>(data_y.data(), data_y.size()), x),
        test_y[ix], 1e-10);
    EXPECT_NEAR(
        interpolator.derivative(
            Eigen::Map<Eigen::VectorXd>(data_x.data(), data_x.size()),
            Eigen::Map<Eigen::VectorXd>(data_y.data(), data_y.size()), x),
        test_dy[ix], 1e-10);
    EXPECT_NEAR(interpolator.integral(
                    Eigen::Map<Eigen::VectorXd>(data_x.data(), data_x.size()),
                    Eigen::Map<Eigen::VectorXd>(data_y.data(), data_y.size()),
                    test_x[0], x),
                test_iy[ix], 1e-10);
  }
}

TEST(gsl, exception) {
  gsl::set_error_handler();

  EXPECT_THROW(gsl::Interpolate1D(1, gsl_interp_cspline, gsl::Accelerator()),
               std::runtime_error);
}

TEST(gsl, bicubic) {
  gsl::set_error_handler();

  auto xarr = std::vector<double>{0.0, 1.0, 2.0, 3.0};
  auto yarr = std::vector<double>{0.0, 1.0, 2.0, 3.0};
  auto zarr = std::vector<double>{1.0, 1.1, 1.2, 1.3, 1.1, 1.2, 1.3, 1.4,
                                  1.2, 1.3, 1.4, 1.5, 1.3, 1.4, 1.5, 1.6};
  auto xval = std::vector<double>{1.0, 1.5, 2.0};
  auto yval = std::vector<double>{1.0, 1.5, 2.0};
  auto zval = std::vector<double>{1.2, 1.3, 1.4};

  auto xacc = gsl::Accelerator();
  auto yacc = gsl::Accelerator();
  auto interpolator =
      gsl::Interpolate2D(xarr.size(), yarr.size(), gsl_interp2d_bicubic,
                         std::move(xacc), std::move(yacc));

  auto xa = Eigen::Map<Eigen::VectorXd>(xarr.data(), xarr.size());
  auto ya = Eigen::Map<Eigen::VectorXd>(yarr.data(), yarr.size());
  auto za = Eigen::Map<Eigen::MatrixXd>(zarr.data(), xarr.size(), yarr.size());

  for (auto ix = 0; ix < 3; ++ix) {
    auto z = interpolator.evaluate(xa, ya, za, xval[ix], yval[ix]);
    EXPECT_NEAR(z, zval[ix], 1e-10);
  }
}

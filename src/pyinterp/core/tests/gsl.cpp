// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>
#include "pyinterp/detail/gsl/error_handler.hpp"
#include "pyinterp/detail/gsl/interpolate1d.hpp"

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

// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include <boost/geometry.hpp>

#include "pyinterp/detail/math/radial_basis_functions.hpp"

namespace math = pyinterp::detail::math;

static void test_1d(math::RadialBasisFunction function) {
  auto x = Eigen::Matrix<double, 1, 9>::LinSpaced(9, 0, 10);
  auto y = x.array().sin();
  auto rbf =
      math::RBF<double>(std::numeric_limits<double>::quiet_NaN(), 0, function);
  auto yi = rbf.interpolate(x, y, x);

  ASSERT_EQ(y.size(), yi.size());
  for (Eigen::Index ix = 0; ix < yi.size(); ++ix) {
    EXPECT_NEAR(y(ix), yi(ix), 1e-9);
  }
}

static void test_2d(math::RadialBasisFunction function) {
  Eigen::Matrix<double, 2, 50> x = Eigen::Matrix<double, 2, 50>::Random();
  Eigen::Matrix<double, 50, 1> y =
      (x.row(0).array().pow(2) - x.row(1).array().pow(2)).array().exp();
  auto rbf =
      math::RBF<double>(std::numeric_limits<double>::quiet_NaN(), 0, function);
  auto yi = rbf.interpolate(x, y, x);

  ASSERT_EQ(y.size(), yi.size());
  for (Eigen::Index ix = 0; ix < yi.size(); ++ix) {
    EXPECT_NEAR(y(ix), yi(ix), 1e-9);
  }
}

static void test_3d(math::RadialBasisFunction function) {
  Eigen::Matrix<double, 3, 50> x = Eigen::Matrix<double, 3, 50>::Random();
  Eigen::Matrix<double, 50, 1> y =
      (x.row(0).array().pow(2) - x.row(1).array().pow(2)).array().exp();
  auto rbf =
      math::RBF<double>(std::numeric_limits<double>::quiet_NaN(), 0, function);
  auto yi = rbf.interpolate(x, y, x);

  ASSERT_EQ(y.size(), yi.size());
  for (Eigen::Index ix = 0; ix < yi.size(); ++ix) {
    EXPECT_NEAR(y(ix), yi(ix), 1e-9);
  }
}

TEST(math_rbf, 1d) {
  test_1d(math::RadialBasisFunction::Cubic);
  test_1d(math::RadialBasisFunction::Gaussian);
  test_1d(math::RadialBasisFunction::InverseMultiquadric);
  test_1d(math::RadialBasisFunction::Linear);
  test_1d(math::RadialBasisFunction::Multiquadric);
  test_1d(math::RadialBasisFunction::ThinPlate);
}

TEST(math_rbf, 2d) {
  test_2d(math::RadialBasisFunction::Cubic);
  test_2d(math::RadialBasisFunction::Gaussian);
  test_2d(math::RadialBasisFunction::InverseMultiquadric);
  test_2d(math::RadialBasisFunction::Linear);
  test_2d(math::RadialBasisFunction::Multiquadric);
  test_2d(math::RadialBasisFunction::ThinPlate);
}

TEST(math_rbf, 3d) {
  test_3d(math::RadialBasisFunction::Cubic);
  test_3d(math::RadialBasisFunction::Gaussian);
  test_3d(math::RadialBasisFunction::InverseMultiquadric);
  test_3d(math::RadialBasisFunction::Linear);
  test_3d(math::RadialBasisFunction::Multiquadric);
  test_3d(math::RadialBasisFunction::ThinPlate);
}

TEST(math_rbf, point) {
  Eigen::Matrix<double, 3, 50> x = Eigen::Matrix<double, 3, 50>::Random();
  Eigen::Matrix<double, 50, 1> y =
      (x.row(0).array().pow(2) - x.row(1).array().pow(2)).array().exp();
  auto rbf = math::RBF<double>(std::numeric_limits<double>::quiet_NaN(), 0,
                               math::RadialBasisFunction::Cubic);

  auto xi = Eigen::Matrix<double, 3, 1>();
  for (Eigen::Index ix = 0; ix < xi.cols(); ++ix) {
    xi << x.col(ix);
    auto yi = rbf.interpolate(x, y, xi);

    ASSERT_EQ(yi.size(), 1);
    EXPECT_NEAR(y(ix), yi(0), 1e-9);
  }
}

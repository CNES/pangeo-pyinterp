// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/math/interpolate/kriging.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <numbers>
#include <optional>
#include <stdexcept>

namespace pyinterp::math::interpolate {

TEST(Kriging, CovarianceFunctions) {
  const double sigma = 2.0;
  const double lambda = 3.0;
  Eigen::Vector3d p1(0.0, 0.0, 0.0);
  Eigen::Vector3d p2(1.0, 0.0, 0.0);
  const double r = (p1 - p2).norm();
  const double d = r / lambda;
  // Matérn 1/2
  {
    EXPECT_NEAR(matern_covariance_12<double>(p1, p2, sigma, lambda),
                sigma * sigma * std::exp(-d), 1e-12);
  }
  // Matérn 3/2
  {
    double sqrt3 = std::numbers::sqrt3_v<double>;
    EXPECT_NEAR(matern_covariance_32<double>(p1, p2, sigma, lambda),
                sigma * sigma * (1.0 + sqrt3 * d) * std::exp(-sqrt3 * d),
                1e-12);
  }
  // Matérn 5/2
  {
    double sqrt5 = std::sqrt(5.0);
    EXPECT_NEAR(matern_covariance_52<double>(p1, p2, sigma, lambda),
                sigma * sigma * (1.0 + sqrt5 * d + (5.0 / 3.0) * d * d) *
                    std::exp(-sqrt5 * d),
                1e-12);
  }
  // Cauchy
  {
    EXPECT_NEAR(cauchy_covariance<double>(p1, p2, sigma, lambda),
                sigma * sigma / (1.0 + (r * r) / (lambda * lambda)), 1e-12);
  }
  // Spherical inside support
  {
    double lambda_sph = 1.0;
    Eigen::Vector3d p3(0.5, 0.0, 0.0);
    double t = p3.norm() / lambda_sph;
    EXPECT_NEAR(spherical_covariance<double>(Eigen::Vector3d::Zero(), p3, sigma,
                                             lambda_sph),
                sigma * sigma * (1.0 - 1.5 * t + 0.5 * t * t * t), 1e-12);
  }
  // Spherical outside support
  {
    double lambda_sph = 1.0;
    Eigen::Vector3d pfar(1.1, 0.0, 0.0);
    EXPECT_NEAR(spherical_covariance<double>(Eigen::Vector3d::Zero(), pfar,
                                             sigma, lambda_sph),
                0.0, 1e-12);
  }
  // Gaussian
  {
    EXPECT_NEAR(gaussian_covariance<double>(p1, p2, sigma, lambda),
                sigma * sigma * std::exp(-d * d), 1e-12);
  }
  // Linear covariance inside
  {
    double lambda_lin = 2.0;
    double t = 1.0 - r / lambda_lin;

    EXPECT_NEAR(wendland_covariance<double>(p1, p2, sigma, lambda_lin),
                sigma * sigma * t * t, 1e-12);
  }
  // Linear covariance outside
  {
    double lambda_lin = 0.5;

    EXPECT_NEAR(wendland_covariance<double>(p1, p2, sigma, lambda_lin), 0.0,
                1e-12);
  }
}

TEST(Kriging, SimpleKrigingReproduction) {
  Eigen::Matrix<double, 3, -1> coords(3, 4);
  coords.col(0) << 0, 0, 0;
  coords.col(1) << 1, 0, 0;
  coords.col(2) << 0, 1, 0;
  coords.col(3) << 0, 0, 1;
  Eigen::VectorXd values(4);
  values << 10.0, 11.0, 12.0, 13.0;
  Kriging<double> krig(1.0, 1.0, 0.0, CovarianceFunction::kGaussian,
                       std::nullopt);
  // Query an existing point -> exact reproduction
  Eigen::Vector3d q = coords.col(2);
  double predicted = krig(coords, values, q);
  EXPECT_NEAR(predicted, values(2), 1e-12);
}

TEST(Kriging, UniversalKrigingLinear) {
  auto f = [](double x, double y, double z) -> double {
    return 2.0 + 3.0 * x - 4.0 * y + 5.0 * z;
  };
  Eigen::Matrix<double, 3, -1> coords(3, 4);
  coords.col(0) << 0, 0, 0;
  coords.col(1) << 1, 0, 0;
  coords.col(2) << 0, 1, 0;
  coords.col(3) << 0, 0, 1;
  Eigen::VectorXd values(4);
  for (int i = 0; i < 4; ++i) {
    values(i) = f(coords(0, i), coords(1, i), coords(2, i));
  }
  Kriging<double> uk(1.0, 2.0, 0.0, CovarianceFunction::kGaussian,
                     DriftFunction::kLinear);
  Eigen::Vector3d q(0.3, 0.2, 0.4);
  EXPECT_NEAR(uk(coords, values, q), f(q[0], q[1], q[2]), 1e-8);
}

TEST(Kriging, UniversalKrigingQuadratic) {
  auto g = [](double x, double y, double z) -> double {
    return 1.0 + x + y + z + x * x + y * y + z * z + x * y + x * z + y * z;
  };
  Eigen::Matrix<double, 3, -1> coords(3, 10);
  coords.col(0) << 0, 0, 0;
  coords.col(1) << 1, 0, 0;
  coords.col(2) << 0, 1, 0;
  coords.col(3) << 0, 0, 1;
  coords.col(4) << 1, 1, 0;
  coords.col(5) << 1, 0, 1;
  coords.col(6) << 0, 1, 1;
  coords.col(7) << 1, 1, 1;
  coords.col(8) << 2, 0, 0;
  coords.col(9) << 0, 2, 0;
  Eigen::VectorXd values(10);
  for (int i = 0; i < 10; ++i) {
    values(i) = g(coords(0, i), coords(1, i), coords(2, i));
  }
  Kriging<double> uq(1.0, 3.0, 0.0, CovarianceFunction::kMatern_32,
                     DriftFunction::kQuadratic);
  Eigen::Vector3d q(0.5, 0.25, 0.75);
  EXPECT_NEAR(uq(coords, values, q), g(q[0], q[1], q[2]), 1e-7);
}

TEST(Kriging, InvalidParameters) {
  // sigma <= 0
  EXPECT_THROW(
      { Kriging<double> k(0.0, 1.0, 0.0, CovarianceFunction::kGaussian); },
      std::invalid_argument);

  EXPECT_THROW(
      { Kriging<double> k(-1.0, 1.0, 0.0, CovarianceFunction::kGaussian); },
      std::invalid_argument);

  EXPECT_THROW(
      { Kriging<double> k(1.0, 0.0, 0.0, CovarianceFunction::kGaussian); },
      std::invalid_argument);
}

TEST(Kriging, MethodSelection) {
  Eigen::Matrix<double, 3, -1> coords(3, 4);
  coords.col(0) << 0, 0, 0;
  coords.col(1) << 1, 0, 0;
  coords.col(2) << 0, 1, 0;
  coords.col(3) << 0, 0, 1;
  // Values not matching any linear drift exactly to force difference
  Eigen::VectorXd values(4);
  values << 0.0, 2.0, -1.0, 3.5;
  Eigen::Vector3d q(0.2, 0.3, 0.1);
  Kriging<double> simple(1.0, 1.5, 0.0, CovarianceFunction::kCauchy,
                         std::nullopt);
  Kriging<double> universal(1.0, 1.5, 0.0, CovarianceFunction::kCauchy,
                            DriftFunction::kLinear);
  EXPECT_NE(simple(coords, values, q), universal(coords, values, q));
}

}  // namespace pyinterp::math::interpolate

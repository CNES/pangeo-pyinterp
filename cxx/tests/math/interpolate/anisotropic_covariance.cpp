// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include <Eigen/Core>

#include "pyinterp/math/interpolate/anisotropic.hpp"
#include "pyinterp/math/interpolate/kriging.hpp"

namespace pyinterp::math::interpolate {

namespace {

constexpr double kTolerance = 1e-12;

// Several point pairs covering small / unit / large separations to exercise
// the kernels across their support.
struct Sample {
  Eigen::Vector3d p1;
  Eigen::Vector3d p2;
};

const std::array<Sample, 4> kSamples = {{
    {.p1 = Eigen::Vector3d(0.0, 0.0, 0.0),
     .p2 = Eigen::Vector3d(1.0, 0.0, 0.0)},
    {.p1 = Eigen::Vector3d(0.0, 0.0, 0.0),
     .p2 = Eigen::Vector3d(0.4, -0.3, 0.7)},
    {.p1 = Eigen::Vector3d(1.5, 2.0, -1.0),
     .p2 = Eigen::Vector3d(0.5, 1.0, 0.0)},
    {.p1 = Eigen::Vector3d(0.0, 0.0, 0.0),
     .p2 = Eigen::Vector3d(3.0, 4.0, 0.0)},
}};

}  // namespace

// Each existing 3D wrapper must agree bit-for-bit with the matching _from_r2
// form evaluated on (p1 - p2).squaredNorm(). This is the refactor invariant.
TEST(AnisotropicCovariance, WrapperEqualsFromR2) {
  constexpr double sigma = 2.0;
  constexpr double lambda = 3.0;

  for (const auto& s : kSamples) {
    const double r2 = (s.p1 - s.p2).squaredNorm();

    EXPECT_DOUBLE_EQ(matern_covariance_12<double>(s.p1, s.p2, sigma, lambda),
                     matern_covariance_12_from_r2<double>(r2, sigma, lambda));
    EXPECT_DOUBLE_EQ(matern_covariance_32<double>(s.p1, s.p2, sigma, lambda),
                     matern_covariance_32_from_r2<double>(r2, sigma, lambda));
    EXPECT_DOUBLE_EQ(matern_covariance_52<double>(s.p1, s.p2, sigma, lambda),
                     matern_covariance_52_from_r2<double>(r2, sigma, lambda));
    EXPECT_DOUBLE_EQ(cauchy_covariance<double>(s.p1, s.p2, sigma, lambda),
                     cauchy_covariance_from_r2<double>(r2, sigma, lambda));
    EXPECT_DOUBLE_EQ(spherical_covariance<double>(s.p1, s.p2, sigma, lambda),
                     spherical_covariance_from_r2<double>(r2, sigma, lambda));
    EXPECT_DOUBLE_EQ(gaussian_covariance<double>(s.p1, s.p2, sigma, lambda),
                     gaussian_covariance_from_r2<double>(r2, sigma, lambda));
    EXPECT_DOUBLE_EQ(wendland_covariance<double>(s.p1, s.p2, sigma, lambda),
                     wendland_covariance_from_r2<double>(r2, sigma, lambda));
  }
}

// At r² = 0 every kernel returns σ² (the sill at zero lag).
TEST(AnisotropicCovariance, ZeroLagReturnsSillSquared) {
  constexpr double sigma = 1.5;
  constexpr double lambda = 1.0;
  const double sill2 = sigma * sigma;

  EXPECT_DOUBLE_EQ(matern_covariance_12_from_r2<double>(0.0, sigma, lambda),
                   sill2);
  EXPECT_DOUBLE_EQ(matern_covariance_32_from_r2<double>(0.0, sigma, lambda),
                   sill2);
  EXPECT_DOUBLE_EQ(matern_covariance_52_from_r2<double>(0.0, sigma, lambda),
                   sill2);
  EXPECT_DOUBLE_EQ(cauchy_covariance_from_r2<double>(0.0, sigma, lambda),
                   sill2);
  EXPECT_DOUBLE_EQ(spherical_covariance_from_r2<double>(0.0, sigma, lambda),
                   sill2);
  EXPECT_DOUBLE_EQ(gaussian_covariance_from_r2<double>(0.0, sigma, lambda),
                   sill2);
  EXPECT_DOUBLE_EQ(wendland_covariance_from_r2<double>(0.0, sigma, lambda),
                   sill2);
}

// Compact-support kernels return exactly zero when r >= lambda.
TEST(AnisotropicCovariance, CompactSupportOutsideRange) {
  constexpr double sigma = 2.0;
  constexpr double lambda = 1.0;
  // Pick r² strictly greater than λ².
  const double r2_far = 1.5 * 1.5;

  EXPECT_DOUBLE_EQ(spherical_covariance_from_r2<double>(r2_far, sigma, lambda),
                   0.0);
  EXPECT_DOUBLE_EQ(wendland_covariance_from_r2<double>(r2_far, sigma, lambda),
                   0.0);
}

// For an isotropic length scale L, the anisotropic squared distance reduces
// to the usual Euclidean squared distance divided by L².
TEST(AnisotropicCovariance, IsotropicDistanceReducesToEuclidean) {
  constexpr double L = 2.5;
  const Eigen::Vector3d inv_L = Eigen::Vector3d::Constant(1.0 / L);

  for (const auto& s : kSamples) {
    const double expected = (s.p1 - s.p2).squaredNorm() / (L * L);
    const double got =
        anisotropic_distance_squared<double, 3>(s.p1, s.p2, inv_L);
    EXPECT_NEAR(expected, got, kTolerance);
  }
}

// Anisotropic distance with axis-specific scales matches the analytic form.
TEST(AnisotropicCovariance, AnisotropicDistanceFormula) {
  const Eigen::Vector3d p1(0.0, 0.0, 0.0);
  const Eigen::Vector3d p2(4.0, 3.0, 2.0);
  const Eigen::Vector3d L(2.0, 1.5, 0.5);
  const Eigen::Vector3d inv_L = L.cwiseInverse();

  const double expected = (4.0 / 2.0) * (4.0 / 2.0) +
                          (3.0 / 1.5) * (3.0 / 1.5) + (2.0 / 0.5) * (2.0 / 0.5);
  const double got = anisotropic_distance_squared<double, 3>(p1, p2, inv_L);
  EXPECT_NEAR(expected, got, kTolerance);
}

// Round-trip via select_radial_covariance must yield the same pointer as the
// direct template instantiation.
TEST(AnisotropicCovariance, SelectRadialCovarianceDispatch) {
  EXPECT_EQ(select_radial_covariance<double>(CovarianceFunction::kMatern_12),
            &matern_covariance_12_from_r2<double>);
  EXPECT_EQ(select_radial_covariance<double>(CovarianceFunction::kMatern_32),
            &matern_covariance_32_from_r2<double>);
  EXPECT_EQ(select_radial_covariance<double>(CovarianceFunction::kMatern_52),
            &matern_covariance_52_from_r2<double>);
  EXPECT_EQ(select_radial_covariance<double>(CovarianceFunction::kCauchy),
            &cauchy_covariance_from_r2<double>);
  EXPECT_EQ(select_radial_covariance<double>(CovarianceFunction::kSpherical),
            &spherical_covariance_from_r2<double>);
  EXPECT_EQ(select_radial_covariance<double>(CovarianceFunction::kGaussian),
            &gaussian_covariance_from_r2<double>);
  EXPECT_EQ(select_radial_covariance<double>(CovarianceFunction::kWendland),
            &wendland_covariance_from_r2<double>);
}

// Float32 specialization compiles and produces sensible numerical values.
TEST(AnisotropicCovariance, Float32Specialization) {
  const Eigen::Vector3f p1(0.0F, 0.0F, 0.0F);
  const Eigen::Vector3f p2(1.0F, 0.5F, -0.25F);
  const Eigen::Vector3f inv_L = Eigen::Vector3f::Constant(0.5F);
  const float r2 = anisotropic_distance_squared<float, 3>(p1, p2, inv_L);

  const float cov = gaussian_covariance_from_r2<float>(r2, 1.0F, 1.0F);
  EXPECT_GT(cov, 0.0F);
  EXPECT_LE(cov, 1.0F);
}

}  // namespace pyinterp::math::interpolate

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/config/rtree.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <optional>
#include <thread>

namespace pyinterp::config::rtree {

// ============================================================================
// Query Config Tests
// ============================================================================

class QueryTest : public ::testing::Test {};

TEST_F(QueryTest, DefaultConstructor) {
  Query config;
  EXPECT_EQ(config.k(), 8);
  EXPECT_DOUBLE_EQ(config.radius(), std::numeric_limits<double>::max());
  EXPECT_EQ(config.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(QueryTest, WithK) {
  Query config;
  auto updated = config.with_k(16);
  EXPECT_EQ(updated.k(), 16);
  EXPECT_EQ(config.k(), 8);  // Original unchanged
}

TEST_F(QueryTest, WithRadius) {
  Query config;
  auto updated = config.with_radius(std::optional<double>(1000.0));
  EXPECT_DOUBLE_EQ(updated.radius(), 1000.0);
  EXPECT_DOUBLE_EQ(config.radius(), std::numeric_limits<double>::max());
}

TEST_F(QueryTest, WithRadiusNullopt) {
  Query config;
  auto updated = config.with_radius(std::nullopt);
  EXPECT_DOUBLE_EQ(updated.radius(), std::numeric_limits<double>::max());
}

TEST_F(QueryTest, WithNumThreads) {
  Query config;
  auto updated = config.with_num_threads(4);
  EXPECT_EQ(updated.num_threads(), 4);
  EXPECT_EQ(config.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(QueryTest, WithBoundaryCheck) {
  Query config;
  auto updated = config.with_boundary_check(geometry::BoundaryCheck::kEnvelope);
  EXPECT_EQ(updated.boundary_check(), geometry::BoundaryCheck::kEnvelope);
}

TEST_F(QueryTest, MethodChaining) {
  Query config;
  auto updated = config.with_k(10)
                     .with_radius(std::optional<double>(500.0))
                     .with_num_threads(8);
  EXPECT_EQ(updated.k(), 10);
  EXPECT_DOUBLE_EQ(updated.radius(), 500.0);
  EXPECT_EQ(updated.num_threads(), 8);
}

// ============================================================================
// InverseDistanceWeighting Config Tests
// ============================================================================

class InverseDistanceWeightingTest : public ::testing::Test {};

TEST_F(InverseDistanceWeightingTest, DefaultConstructor) {
  InverseDistanceWeighting config;
  EXPECT_EQ(config.k(), 8);
  EXPECT_DOUBLE_EQ(config.radius(), std::numeric_limits<double>::max());
  EXPECT_EQ(config.p(), 2);
  EXPECT_EQ(config.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(InverseDistanceWeightingTest, WithP) {
  InverseDistanceWeighting config;
  auto updated = config.with_p(3);
  EXPECT_EQ(updated.p(), 3);
  EXPECT_EQ(config.p(), 2);
}

TEST_F(InverseDistanceWeightingTest, WithK) {
  InverseDistanceWeighting config;
  auto updated = config.with_k(12);
  EXPECT_EQ(updated.k(), 12);
  EXPECT_EQ(config.k(), 8);
}

TEST_F(InverseDistanceWeightingTest, WithRadius) {
  InverseDistanceWeighting config;
  auto updated = config.with_radius(std::optional<double>(2000.0));
  EXPECT_DOUBLE_EQ(updated.radius(), 2000.0);
}

TEST_F(InverseDistanceWeightingTest, WithNumThreads) {
  InverseDistanceWeighting config;
  auto updated = config.with_num_threads(16);
  EXPECT_EQ(updated.num_threads(), 16);
}

TEST_F(InverseDistanceWeightingTest, WithBoundaryCheck) {
  InverseDistanceWeighting config;
  auto updated =
      config.with_boundary_check(geometry::BoundaryCheck::kConvexHull);
  EXPECT_EQ(updated.boundary_check(), geometry::BoundaryCheck::kConvexHull);
}

TEST_F(InverseDistanceWeightingTest, MethodChaining) {
  InverseDistanceWeighting config;
  auto updated = config.with_p(4)
                     .with_k(20)
                     .with_radius(std::optional<double>(1500.0))
                     .with_num_threads(12);
  EXPECT_EQ(updated.p(), 4);
  EXPECT_EQ(updated.k(), 20);
  EXPECT_DOUBLE_EQ(updated.radius(), 1500.0);
  EXPECT_EQ(updated.num_threads(), 12);
}

// ============================================================================
// Kriging Config Tests
// ============================================================================

class KrigingTest : public ::testing::Test {};

TEST_F(KrigingTest, DefaultConstructor) {
  Kriging config;
  EXPECT_EQ(config.k(), 8);
  EXPECT_DOUBLE_EQ(config.radius(), std::numeric_limits<double>::max());
  EXPECT_DOUBLE_EQ(config.sigma(), 1.0);
  EXPECT_DOUBLE_EQ(config.lambda(), 1.0);
  EXPECT_DOUBLE_EQ(config.nugget(), 0.0);
  EXPECT_EQ(config.covariance_model(),
            math::interpolate::CovarianceFunction::kSpherical);
  EXPECT_FALSE(config.drift_function().has_value());
  EXPECT_EQ(config.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(KrigingTest, WithSigma) {
  Kriging config;
  auto updated = config.with_sigma(2.0);
  EXPECT_DOUBLE_EQ(updated.sigma(), 2.0);
  EXPECT_DOUBLE_EQ(config.sigma(), 1.0);
}

TEST_F(KrigingTest, WithLambda) {
  Kriging config;
  auto updated = config.with_lambda(3.0);
  EXPECT_DOUBLE_EQ(updated.lambda(), 3.0);
  EXPECT_DOUBLE_EQ(config.lambda(), 1.0);
}

TEST_F(KrigingTest, WithNugget) {
  Kriging config;
  auto updated = config.with_nugget(0.1);
  EXPECT_DOUBLE_EQ(updated.nugget(), 0.1);
  EXPECT_DOUBLE_EQ(config.nugget(), 0.0);
}

TEST_F(KrigingTest, WithCovarianceModel) {
  Kriging config;
  auto updated = config.with_covariance_model(
      math::interpolate::CovarianceFunction::kGaussian);
  EXPECT_EQ(updated.covariance_model(),
            math::interpolate::CovarianceFunction::kGaussian);
  EXPECT_EQ(config.covariance_model(),
            math::interpolate::CovarianceFunction::kSpherical);
}

TEST_F(KrigingTest, WithDriftFunction) {
  Kriging config;
  auto updated = config.with_drift_function(
      std::optional(math::interpolate::DriftFunction::kLinear));
  EXPECT_TRUE(updated.drift_function().has_value());
  EXPECT_EQ(updated.drift_function().value(),
            math::interpolate::DriftFunction::kLinear);
  EXPECT_FALSE(config.drift_function().has_value());
}

TEST_F(KrigingTest, WithDriftFunctionNullopt) {
  Kriging config;
  auto with_drift = config.with_drift_function(
      std::optional(math::interpolate::DriftFunction::kLinear));
  auto updated = with_drift.with_drift_function(std::nullopt);
  EXPECT_FALSE(updated.drift_function().has_value());
}

TEST_F(KrigingTest, WithK) {
  Kriging config;
  auto updated = config.with_k(10);
  EXPECT_EQ(updated.k(), 10);
}

TEST_F(KrigingTest, WithRadius) {
  Kriging config;
  auto updated = config.with_radius(std::optional<double>(1000.0));
  EXPECT_DOUBLE_EQ(updated.radius(), 1000.0);
}

TEST_F(KrigingTest, WithNumThreads) {
  Kriging config;
  auto updated = config.with_num_threads(8);
  EXPECT_EQ(updated.num_threads(), 8);
}

TEST_F(KrigingTest, WithBoundaryCheck) {
  Kriging config;
  auto updated =
      config.with_boundary_check(geometry::BoundaryCheck::kConvexHull);
  EXPECT_EQ(updated.boundary_check(), geometry::BoundaryCheck::kConvexHull);
}

TEST_F(KrigingTest, MethodChaining) {
  Kriging config;
  auto updated = config.with_sigma(2.5)
                     .with_lambda(4.0)
                     .with_nugget(0.2)
                     .with_covariance_model(
                         math::interpolate::CovarianceFunction::kGaussian)
                     .with_drift_function(std::optional(
                         math::interpolate::DriftFunction::kLinear))
                     .with_k(15)
                     .with_radius(std::optional<double>(3000.0))
                     .with_num_threads(12);

  EXPECT_DOUBLE_EQ(updated.sigma(), 2.5);
  EXPECT_DOUBLE_EQ(updated.lambda(), 4.0);
  EXPECT_DOUBLE_EQ(updated.nugget(), 0.2);
  EXPECT_EQ(updated.covariance_model(),
            math::interpolate::CovarianceFunction::kGaussian);
  EXPECT_TRUE(updated.drift_function().has_value());
  EXPECT_EQ(updated.drift_function().value(),
            math::interpolate::DriftFunction::kLinear);
  EXPECT_EQ(updated.k(), 15);
  EXPECT_DOUBLE_EQ(updated.radius(), 3000.0);
  EXPECT_EQ(updated.num_threads(), 12);
}

// ============================================================================
// RadialBasisFunction Config Tests
// ============================================================================

class RadialBasisFunctionTest : public ::testing::Test {};

TEST_F(RadialBasisFunctionTest, DefaultConstructor) {
  RadialBasisFunction config;
  EXPECT_EQ(config.k(), 8);
  EXPECT_DOUBLE_EQ(config.radius(), std::numeric_limits<double>::max());
  EXPECT_EQ(config.rbf(), math::interpolate::RBFKernel::kMultiquadric);
  EXPECT_TRUE(std::isnan(config.epsilon()));
  EXPECT_DOUBLE_EQ(config.smooth(), 0.0);
  EXPECT_EQ(config.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(RadialBasisFunctionTest, WithRbf) {
  RadialBasisFunction config;
  auto updated = config.with_rbf(math::interpolate::RBFKernel::kGaussian);
  EXPECT_EQ(updated.rbf(), math::interpolate::RBFKernel::kGaussian);
  EXPECT_EQ(config.rbf(), math::interpolate::RBFKernel::kMultiquadric);
}

TEST_F(RadialBasisFunctionTest, WithEpsilon) {
  RadialBasisFunction config;
  auto updated = config.with_epsilon(std::optional<double>(0.5));
  EXPECT_DOUBLE_EQ(updated.epsilon(), 0.5);
  EXPECT_TRUE(std::isnan(config.epsilon()));
}

TEST_F(RadialBasisFunctionTest, WithEpsilonNullopt) {
  RadialBasisFunction config;
  auto updated = config.with_epsilon(std::nullopt);
  EXPECT_TRUE(std::isnan(updated.epsilon()));
}

TEST_F(RadialBasisFunctionTest, WithSmooth) {
  RadialBasisFunction config;
  auto updated = config.with_smooth(0.1);
  EXPECT_DOUBLE_EQ(updated.smooth(), 0.1);
  EXPECT_DOUBLE_EQ(config.smooth(), 0.0);
}

TEST_F(RadialBasisFunctionTest, WithK) {
  RadialBasisFunction config;
  auto updated = config.with_k(12);
  EXPECT_EQ(updated.k(), 12);
}

TEST_F(RadialBasisFunctionTest, WithRadius) {
  RadialBasisFunction config;
  auto updated = config.with_radius(std::optional<double>(2500.0));
  EXPECT_DOUBLE_EQ(updated.radius(), 2500.0);
}

TEST_F(RadialBasisFunctionTest, WithNumThreads) {
  RadialBasisFunction config;
  auto updated = config.with_num_threads(6);
  EXPECT_EQ(updated.num_threads(), 6);
}

TEST_F(RadialBasisFunctionTest, WithBoundaryCheck) {
  RadialBasisFunction config;
  auto updated = config.with_boundary_check(geometry::BoundaryCheck::kEnvelope);
  EXPECT_EQ(updated.boundary_check(), geometry::BoundaryCheck::kEnvelope);
}

TEST_F(RadialBasisFunctionTest, MethodChaining) {
  RadialBasisFunction config;
  auto updated = config.with_rbf(math::interpolate::RBFKernel::kThinPlate)
                     .with_epsilon(std::optional<double>(1.5))
                     .with_smooth(0.05)
                     .with_k(16)
                     .with_radius(std::optional<double>(1800.0))
                     .with_num_threads(10);

  EXPECT_EQ(updated.rbf(), math::interpolate::RBFKernel::kThinPlate);
  EXPECT_DOUBLE_EQ(updated.epsilon(), 1.5);
  EXPECT_DOUBLE_EQ(updated.smooth(), 0.05);
  EXPECT_EQ(updated.k(), 16);
  EXPECT_DOUBLE_EQ(updated.radius(), 1800.0);
  EXPECT_EQ(updated.num_threads(), 10);
}

// ============================================================================
// InterpolationWindow Config Tests
// ============================================================================

class InterpolationWindowTest : public ::testing::Test {};

TEST_F(InterpolationWindowTest, DefaultConstructor) {
  InterpolationWindow config;
  EXPECT_EQ(config.k(), 8);
  EXPECT_DOUBLE_EQ(config.radius(), std::numeric_limits<double>::max());
  EXPECT_EQ(config.wf(), math::interpolate::window::Kernel::kGaussian);
  EXPECT_DOUBLE_EQ(config.arg(), 1.0);  // Default for Gaussian
  EXPECT_EQ(config.num_threads(), std::thread::hardware_concurrency());
}

TEST_F(InterpolationWindowTest, WithWf) {
  InterpolationWindow config;
  auto updated = config.with_wf(math::interpolate::window::Kernel::kLanczos);
  EXPECT_EQ(updated.wf(), math::interpolate::window::Kernel::kLanczos);
  EXPECT_EQ(config.wf(), math::interpolate::window::Kernel::kGaussian);
}

TEST_F(InterpolationWindowTest, WithArg) {
  InterpolationWindow config;
  auto updated = config.with_arg(std::optional<double>(2.5));
  EXPECT_DOUBLE_EQ(updated.arg(), 2.5);
  EXPECT_DOUBLE_EQ(config.arg(), 1.0);
}

TEST_F(InterpolationWindowTest, WithArgNullopt) {
  InterpolationWindow config;
  auto updated = config.with_arg(std::nullopt);
  // Should use default for Gaussian
  EXPECT_DOUBLE_EQ(updated.arg(), 1.0);
}

TEST_F(InterpolationWindowTest, DefaultArgDependsOnWindowFunction) {
  InterpolationWindow config;
  auto gaussian = config.with_wf(math::interpolate::window::Kernel::kGaussian);
  EXPECT_DOUBLE_EQ(gaussian.arg(), 1.0);

  auto lanczos = config.with_wf(math::interpolate::window::Kernel::kLanczos);
  EXPECT_DOUBLE_EQ(lanczos.arg(), 1.0);

  auto blackman = config.with_wf(math::interpolate::window::Kernel::kBlackman);
  EXPECT_DOUBLE_EQ(blackman.arg(), 0.0);  // Default for non-Gaussian/Lanczos
}

TEST_F(InterpolationWindowTest, WithK) {
  InterpolationWindow config;
  auto updated = config.with_k(20);
  EXPECT_EQ(updated.k(), 20);
}

TEST_F(InterpolationWindowTest, WithRadius) {
  InterpolationWindow config;
  auto updated = config.with_radius(std::optional<double>(5000.0));
  EXPECT_DOUBLE_EQ(updated.radius(), 5000.0);
}

TEST_F(InterpolationWindowTest, WithNumThreads) {
  InterpolationWindow config;
  auto updated = config.with_num_threads(4);
  EXPECT_EQ(updated.num_threads(), 4);
}

TEST_F(InterpolationWindowTest, WithBoundaryCheck) {
  InterpolationWindow config;
  auto updated =
      config.with_boundary_check(geometry::BoundaryCheck::kConvexHull);
  EXPECT_EQ(updated.boundary_check(), geometry::BoundaryCheck::kConvexHull);
}

TEST_F(InterpolationWindowTest, MethodChaining) {
  InterpolationWindow config;
  auto updated = config.with_wf(math::interpolate::window::Kernel::kParzen)
                     .with_arg(std::optional<double>(3.0))
                     .with_k(25)
                     .with_radius(std::optional<double>(4000.0))
                     .with_num_threads(14);

  EXPECT_EQ(updated.wf(), math::interpolate::window::Kernel::kParzen);
  EXPECT_DOUBLE_EQ(updated.arg(), 3.0);
  EXPECT_EQ(updated.k(), 25);
  EXPECT_DOUBLE_EQ(updated.radius(), 4000.0);
  EXPECT_EQ(updated.num_threads(), 14);
}

}  // namespace pyinterp::config::rtree

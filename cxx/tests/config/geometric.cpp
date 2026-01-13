// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/config/geometric.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <thread>

namespace pyinterp::config::geometric {

// ============================================================================
// SpatialMethod Tests
// ============================================================================

class SpatialMethodTest : public ::testing::Test {};

TEST_F(SpatialMethodTest, ParseSpatialMethodBilinear) {
  auto method = parse_spatial_method("bilinear");
  EXPECT_EQ(method, SpatialMethod::kBilinear);
}

TEST_F(SpatialMethodTest, ParseSpatialMethodIDW) {
  auto method = parse_spatial_method("idw");
  EXPECT_EQ(method, SpatialMethod::kInverseDistanceWeighting);
}

TEST_F(SpatialMethodTest, ParseSpatialMethodNearest) {
  auto method = parse_spatial_method("nearest");
  EXPECT_EQ(method, SpatialMethod::kNearest);
}

TEST_F(SpatialMethodTest, ParseSpatialMethodUnknownThrows) {
  EXPECT_THROW(static_cast<void>(parse_spatial_method("invalid")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(parse_spatial_method("")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(parse_spatial_method("Bilinear")),
               std::invalid_argument);  // case-sensitive
  EXPECT_THROW(static_cast<void>(parse_spatial_method("IDW")),
               std::invalid_argument);
}

// ============================================================================
// Spatial Config Tests
// ============================================================================

class SpatialTest : public ::testing::Test {};

TEST_F(SpatialTest, DefaultConstructor) {
  Spatial config;
  EXPECT_EQ(config.method(), SpatialMethod::kBilinear);
  EXPECT_EQ(config.exponent(), 2);
}

TEST_F(SpatialTest, ConstructorWithMethod) {
  Spatial config(SpatialMethod::kNearest);
  EXPECT_EQ(config.method(), SpatialMethod::kNearest);
  EXPECT_EQ(config.exponent(), 2);
}

TEST_F(SpatialTest, ConstructorWithMethodAndExponent) {
  Spatial config(SpatialMethod::kInverseDistanceWeighting, 3);
  EXPECT_EQ(config.method(), SpatialMethod::kInverseDistanceWeighting);
  EXPECT_EQ(config.exponent(), 3);
}

TEST_F(SpatialTest, BilinearFactoryMethod) {
  auto config = Spatial::bilinear();
  EXPECT_EQ(config.method(), SpatialMethod::kBilinear);
  EXPECT_EQ(config.exponent(), 2);
}

TEST_F(SpatialTest, NearestFactoryMethod) {
  auto config = Spatial::nearest();
  EXPECT_EQ(config.method(), SpatialMethod::kNearest);
  EXPECT_EQ(config.exponent(), 2);
}

TEST_F(SpatialTest, IdwFactoryMethodDefaultExponent) {
  auto config = Spatial::idw();
  EXPECT_EQ(config.method(), SpatialMethod::kInverseDistanceWeighting);
  EXPECT_EQ(config.exponent(), 2);
}

TEST_F(SpatialTest, IdwFactoryMethodCustomExponent) {
  auto config = Spatial::idw(3);
  EXPECT_EQ(config.method(), SpatialMethod::kInverseDistanceWeighting);
  EXPECT_EQ(config.exponent(), 3);
}

TEST_F(SpatialTest, ConstexprUsage) {
  constexpr Spatial config = Spatial::bilinear();
  constexpr auto method = config.method();
  EXPECT_EQ(method, SpatialMethod::kBilinear);
}

// ============================================================================
// Bivariate Config Tests
// ============================================================================

class BivariateTest : public ::testing::Test {};

TEST_F(BivariateTest, DefaultConstructor) {
  Bivariate config;
  EXPECT_EQ(config.spatial().method(), SpatialMethod::kBilinear);
  EXPECT_FALSE(config.common().bounds_error());
  EXPECT_EQ(config.common().num_threads(), std::thread::hardware_concurrency());
}

TEST_F(BivariateTest, ConstructorWithSpatial) {
  Spatial spatial = Spatial::nearest();
  Bivariate config(spatial);
  EXPECT_EQ(config.spatial().method(), SpatialMethod::kNearest);
}

TEST_F(BivariateTest, ConstructorWithSpatialAndCommon) {
  Spatial spatial = Spatial::idw(3);
  Common common;
  common = common.with_bounds_error(true);
  Bivariate config(spatial, common);
  EXPECT_EQ(config.spatial().method(),
            SpatialMethod::kInverseDistanceWeighting);
  EXPECT_EQ(config.spatial().exponent(), 3);
  EXPECT_TRUE(config.common().bounds_error());
}

TEST_F(BivariateTest, BilinearFactoryMethod) {
  auto config = Bivariate::bilinear();
  EXPECT_EQ(config.spatial().method(), SpatialMethod::kBilinear);
}

TEST_F(BivariateTest, NearestFactoryMethod) {
  auto config = Bivariate::nearest();
  EXPECT_EQ(config.spatial().method(), SpatialMethod::kNearest);
}

TEST_F(BivariateTest, IdwFactoryMethod) {
  auto config = Bivariate::idw(4);
  EXPECT_EQ(config.spatial().method(),
            SpatialMethod::kInverseDistanceWeighting);
  EXPECT_EQ(config.spatial().exponent(), 4);
}

TEST_F(BivariateTest, WithBoundsError) {
  Bivariate config;
  auto updated = config.with_bounds_error(true);
  EXPECT_TRUE(updated.common().bounds_error());
  EXPECT_FALSE(config.common().bounds_error());
}

TEST_F(BivariateTest, WithNumThreads) {
  Bivariate config;
  auto updated = config.with_num_threads(4);
  EXPECT_EQ(updated.common().num_threads(), 4);
  EXPECT_EQ(config.common().num_threads(), std::thread::hardware_concurrency());
}

TEST_F(BivariateTest, WithSpatial) {
  Bivariate config;
  Spatial spatial = Spatial::nearest();
  auto updated = config.with_spatial(spatial);
  EXPECT_EQ(updated.spatial().method(), SpatialMethod::kNearest);
  EXPECT_EQ(config.spatial().method(), SpatialMethod::kBilinear);
}

TEST_F(BivariateTest, MethodChaining) {
  Bivariate config;
  auto updated =
      config.with_bounds_error(true).with_num_threads(8).with_spatial(
          Spatial::idw(3));
  EXPECT_TRUE(updated.common().bounds_error());
  EXPECT_EQ(updated.common().num_threads(), 8);
  EXPECT_EQ(updated.spatial().method(),
            SpatialMethod::kInverseDistanceWeighting);
  EXPECT_EQ(updated.spatial().exponent(), 3);
}

// ============================================================================
// Trivariate Config Tests
// ============================================================================

class TrivariateTest : public ::testing::Test {};

TEST_F(TrivariateTest, DefaultConstructor) {
  Trivariate config;
  EXPECT_EQ(config.spatial().method(), SpatialMethod::kBilinear);
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kLinear);
  EXPECT_FALSE(config.common().bounds_error());
}

TEST_F(TrivariateTest, ConstructorWithSpatialAndAxis) {
  Spatial spatial = Spatial::nearest();
  AxisConfig axis = AxisConfig::nearest();
  Trivariate config(spatial, axis);
  EXPECT_EQ(config.spatial().method(), SpatialMethod::kNearest);
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kNearest);
}

TEST_F(TrivariateTest, ConstructorWithSpatialAxisAndCommon) {
  Spatial spatial = Spatial::idw(3);
  AxisConfig axis = AxisConfig::linear();
  Common common;
  common = common.with_bounds_error(true);
  Trivariate config(spatial, axis, common);
  EXPECT_EQ(config.spatial().method(),
            SpatialMethod::kInverseDistanceWeighting);
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kLinear);
  EXPECT_TRUE(config.common().bounds_error());
}

TEST_F(TrivariateTest, BilinearFactoryMethod) {
  auto config = Trivariate::bilinear();
  EXPECT_EQ(config.spatial().method(), SpatialMethod::kBilinear);
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kLinear);
}

TEST_F(TrivariateTest, NearestFactoryMethod) {
  auto config = Trivariate::nearest();
  EXPECT_EQ(config.spatial().method(), SpatialMethod::kNearest);
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kNearest);
}

TEST_F(TrivariateTest, IdwFactoryMethod) {
  auto config = Trivariate::idw(5);
  EXPECT_EQ(config.spatial().method(),
            SpatialMethod::kInverseDistanceWeighting);
  EXPECT_EQ(config.spatial().exponent(), 5);
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kLinear);
}

TEST_F(TrivariateTest, WithThirdAxis) {
  Trivariate config;
  auto updated = config.with_third_axis(AxisConfig::nearest());
  EXPECT_EQ(updated.third_axis().method(), AxisMethod::kNearest);
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kLinear);
}

TEST_F(TrivariateTest, WithBoundsError) {
  Trivariate config;
  auto updated = config.with_bounds_error(true);
  EXPECT_TRUE(updated.common().bounds_error());
  EXPECT_FALSE(config.common().bounds_error());
}

TEST_F(TrivariateTest, WithNumThreads) {
  Trivariate config;
  auto updated = config.with_num_threads(4);
  EXPECT_EQ(updated.common().num_threads(), 4);
}

TEST_F(TrivariateTest, WithSpatial) {
  Trivariate config;
  auto updated = config.with_spatial(Spatial::nearest());
  EXPECT_EQ(updated.spatial().method(), SpatialMethod::kNearest);
}

TEST_F(TrivariateTest, MethodChaining) {
  Trivariate config;
  auto updated = config.with_bounds_error(true)
                     .with_num_threads(8)
                     .with_spatial(Spatial::idw(3))
                     .with_third_axis(AxisConfig::nearest());
  EXPECT_TRUE(updated.common().bounds_error());
  EXPECT_EQ(updated.common().num_threads(), 8);
  EXPECT_EQ(updated.spatial().method(),
            SpatialMethod::kInverseDistanceWeighting);
  EXPECT_EQ(updated.third_axis().method(), AxisMethod::kNearest);
}

// ============================================================================
// Quadrivariate Config Tests
// ============================================================================

class QuadrivariateTest : public ::testing::Test {};

TEST_F(QuadrivariateTest, DefaultConstructor) {
  Quadrivariate config;
  EXPECT_EQ(config.spatial().method(), SpatialMethod::kBilinear);
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kLinear);
  EXPECT_EQ(config.fourth_axis().method(), AxisMethod::kLinear);
  EXPECT_FALSE(config.common().bounds_error());
}

TEST_F(QuadrivariateTest, ConstructorWithSpatialAndAxes) {
  Spatial spatial = Spatial::nearest();
  AxisConfig axis3 = AxisConfig::nearest();
  AxisConfig axis4 = AxisConfig::linear();
  Quadrivariate config(spatial, axis3, axis4);
  EXPECT_EQ(config.spatial().method(), SpatialMethod::kNearest);
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kNearest);
  EXPECT_EQ(config.fourth_axis().method(), AxisMethod::kLinear);
}

TEST_F(QuadrivariateTest, ConstructorWithSpatialAxesAndCommon) {
  Spatial spatial = Spatial::idw(3);
  AxisConfig axis3 = AxisConfig::linear();
  AxisConfig axis4 = AxisConfig::nearest();
  Common common;
  common = common.with_bounds_error(true);
  Quadrivariate config(spatial, axis3, axis4, common);
  EXPECT_EQ(config.spatial().method(),
            SpatialMethod::kInverseDistanceWeighting);
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kLinear);
  EXPECT_EQ(config.fourth_axis().method(), AxisMethod::kNearest);
  EXPECT_TRUE(config.common().bounds_error());
}

TEST_F(QuadrivariateTest, BilinearFactoryMethod) {
  auto config = Quadrivariate::bilinear();
  EXPECT_EQ(config.spatial().method(), SpatialMethod::kBilinear);
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kLinear);
  EXPECT_EQ(config.fourth_axis().method(), AxisMethod::kLinear);
}

TEST_F(QuadrivariateTest, NearestFactoryMethod) {
  auto config = Quadrivariate::nearest();
  EXPECT_EQ(config.spatial().method(), SpatialMethod::kNearest);
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kNearest);
  EXPECT_EQ(config.fourth_axis().method(), AxisMethod::kNearest);
}

TEST_F(QuadrivariateTest, IdwFactoryMethod) {
  auto config = Quadrivariate::idw(6);
  EXPECT_EQ(config.spatial().method(),
            SpatialMethod::kInverseDistanceWeighting);
  EXPECT_EQ(config.spatial().exponent(), 6);
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kLinear);
  EXPECT_EQ(config.fourth_axis().method(), AxisMethod::kLinear);
}

TEST_F(QuadrivariateTest, WithThirdAxis) {
  Quadrivariate config;
  auto updated = config.with_third_axis(AxisConfig::nearest());
  EXPECT_EQ(updated.third_axis().method(), AxisMethod::kNearest);
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kLinear);
}

TEST_F(QuadrivariateTest, WithFourthAxis) {
  Quadrivariate config;
  auto updated = config.with_fourth_axis(AxisConfig::nearest());
  EXPECT_EQ(updated.fourth_axis().method(), AxisMethod::kNearest);
  EXPECT_EQ(config.fourth_axis().method(), AxisMethod::kLinear);
}

TEST_F(QuadrivariateTest, WithBoundsError) {
  Quadrivariate config;
  auto updated = config.with_bounds_error(true);
  EXPECT_TRUE(updated.common().bounds_error());
  EXPECT_FALSE(config.common().bounds_error());
}

TEST_F(QuadrivariateTest, WithNumThreads) {
  Quadrivariate config;
  auto updated = config.with_num_threads(4);
  EXPECT_EQ(updated.common().num_threads(), 4);
}

TEST_F(QuadrivariateTest, WithSpatial) {
  Quadrivariate config;
  auto updated = config.with_spatial(Spatial::nearest());
  EXPECT_EQ(updated.spatial().method(), SpatialMethod::kNearest);
}

TEST_F(QuadrivariateTest, MethodChaining) {
  Quadrivariate config;
  auto updated = config.with_bounds_error(true)
                     .with_num_threads(8)
                     .with_spatial(Spatial::idw(3))
                     .with_third_axis(AxisConfig::nearest())
                     .with_fourth_axis(AxisConfig::nearest());
  EXPECT_TRUE(updated.common().bounds_error());
  EXPECT_EQ(updated.common().num_threads(), 8);
  EXPECT_EQ(updated.spatial().method(),
            SpatialMethod::kInverseDistanceWeighting);
  EXPECT_EQ(updated.third_axis().method(), AxisMethod::kNearest);
  EXPECT_EQ(updated.fourth_axis().method(), AxisMethod::kNearest);
}

}  // namespace pyinterp::config::geometric

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/config/windowed.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <thread>
#include <variant>

namespace pyinterp::config::windowed {

// ============================================================================
// BoundaryConfig Tests
// ============================================================================

class BoundaryConfigTest : public ::testing::Test {};

TEST_F(BoundaryConfigTest, DefaultConstructor) {
  BoundaryConfig config;
  // Default should be kUndef
  EXPECT_EQ(config.mode(), BoundaryMode::kUndef);
}

TEST_F(BoundaryConfigTest, ConstructorWithMode) {
  BoundaryConfig config(BoundaryMode::kShrink);
  EXPECT_EQ(config.mode(), BoundaryMode::kShrink);
}

TEST_F(BoundaryConfigTest, UndefFactoryMethod) {
  auto config = BoundaryConfig::undef();
  EXPECT_EQ(config.mode(), BoundaryMode::kUndef);
}

// ============================================================================
// FittingModel Tests
// ============================================================================

class FittingModelTest : public ::testing::Test {};

TEST_F(FittingModelTest, ParseFittingModelAkima) {
  auto model = parse_fitting_model("akima");
  EXPECT_TRUE(std::holds_alternative<Spline>(model));
  EXPECT_EQ(std::get<Spline>(model), Spline::kAkima);
}

TEST_F(FittingModelTest, ParseFittingModelAkimaPeriodic) {
  auto model = parse_fitting_model("akima_periodic");
  EXPECT_TRUE(std::holds_alternative<Spline>(model));
  EXPECT_EQ(std::get<Spline>(model), Spline::kAkimaPeriodic);
}

TEST_F(FittingModelTest, ParseFittingModelCSpline) {
  auto model = parse_fitting_model("c_spline");
  EXPECT_TRUE(std::holds_alternative<Spline>(model));
  EXPECT_EQ(std::get<Spline>(model), Spline::kCSpline);
}

TEST_F(FittingModelTest, ParseFittingModelCSplineNotAKnot) {
  auto model = parse_fitting_model("c_spline_not_a_knot");
  EXPECT_TRUE(std::holds_alternative<Spline>(model));
  EXPECT_EQ(std::get<Spline>(model), Spline::kCSplineNotAKnot);
}

TEST_F(FittingModelTest, ParseFittingModelCSplinePeriodic) {
  auto model = parse_fitting_model("c_spline_periodic");
  EXPECT_TRUE(std::holds_alternative<Spline>(model));
  EXPECT_EQ(std::get<Spline>(model), Spline::kCSplinePeriodic);
}

TEST_F(FittingModelTest, ParseFittingModelLinear) {
  auto model = parse_fitting_model("linear");
  EXPECT_TRUE(std::holds_alternative<Spline>(model));
  EXPECT_EQ(std::get<Spline>(model), Spline::kLinear);
}

TEST_F(FittingModelTest, ParseFittingModelSteffen) {
  auto model = parse_fitting_model("steffen");
  EXPECT_TRUE(std::holds_alternative<Spline>(model));
  EXPECT_EQ(std::get<Spline>(model), Spline::kSteffen);
}

TEST_F(FittingModelTest, ParseFittingModelPolynomial) {
  auto model = parse_fitting_model("polynomial");
  EXPECT_TRUE(std::holds_alternative<Spline>(model));
  EXPECT_EQ(std::get<Spline>(model), Spline::kPolynomial);
}

TEST_F(FittingModelTest, ParseFittingModelBilinear) {
  auto model = parse_fitting_model("bilinear");
  EXPECT_TRUE(std::holds_alternative<Bicubic>(model));
  EXPECT_EQ(std::get<Bicubic>(model), Bicubic::kBilinear);
}

TEST_F(FittingModelTest, ParseFittingModelBicubic) {
  auto model = parse_fitting_model("bicubic");
  EXPECT_TRUE(std::holds_alternative<Bicubic>(model));
  EXPECT_EQ(std::get<Bicubic>(model), Bicubic::kBicubic);
}

TEST_F(FittingModelTest, ParseFittingModelUnknownThrows) {
  EXPECT_THROW(static_cast<void>(parse_fitting_model("invalid")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(parse_fitting_model("")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(parse_fitting_model("Linear")),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(parse_fitting_model("BICUBIC")),
               std::invalid_argument);
}

// ============================================================================
// Spatial Config Tests
// ============================================================================

class WindowedSpatialTest : public ::testing::Test {};

TEST_F(WindowedSpatialTest, DefaultConstructor) {
  Spatial config;
  EXPECT_TRUE(std::holds_alternative<Bicubic>(config.method()));
  EXPECT_EQ(std::get<Bicubic>(config.method()), Bicubic::kBicubic);
  EXPECT_EQ(config.boundary_mode(), BoundaryMode::kUndef);
  EXPECT_EQ(config.half_window_size_x(), 3);
  EXPECT_EQ(config.half_window_size_y(), 3);
}

TEST_F(WindowedSpatialTest, ConstructorWithMethod) {
  Spatial config(Spline::kLinear);
  EXPECT_TRUE(std::holds_alternative<Spline>(config.method()));
  EXPECT_EQ(std::get<Spline>(config.method()), Spline::kLinear);
}

TEST_F(WindowedSpatialTest, BilinearFactoryMethod) {
  auto config = Spatial::bilinear();
  EXPECT_TRUE(std::holds_alternative<Bicubic>(config.method()));
  EXPECT_EQ(std::get<Bicubic>(config.method()), Bicubic::kBilinear);
}

TEST_F(WindowedSpatialTest, BicubicFactoryMethod) {
  auto config = Spatial::bicubic();
  EXPECT_TRUE(std::holds_alternative<Bicubic>(config.method()));
  EXPECT_EQ(std::get<Bicubic>(config.method()), Bicubic::kBicubic);
}

TEST_F(WindowedSpatialTest, LinearFactoryMethod) {
  auto config = Spatial::linear();
  EXPECT_TRUE(std::holds_alternative<Spline>(config.method()));
  EXPECT_EQ(std::get<Spline>(config.method()), Spline::kLinear);
}

TEST_F(WindowedSpatialTest, AkimaFactoryMethod) {
  auto config = Spatial::akima();
  EXPECT_TRUE(std::holds_alternative<Spline>(config.method()));
  EXPECT_EQ(std::get<Spline>(config.method()), Spline::kAkima);
}

TEST_F(WindowedSpatialTest, AkimaPeriodicFactoryMethod) {
  auto config = Spatial::akima_periodic();
  EXPECT_TRUE(std::holds_alternative<Spline>(config.method()));
  EXPECT_EQ(std::get<Spline>(config.method()), Spline::kAkimaPeriodic);
}

TEST_F(WindowedSpatialTest, CSplineFactoryMethod) {
  auto config = Spatial::c_spline();
  EXPECT_TRUE(std::holds_alternative<Spline>(config.method()));
  EXPECT_EQ(std::get<Spline>(config.method()), Spline::kCSpline);
}

TEST_F(WindowedSpatialTest, CSplineNotAKnotFactoryMethod) {
  auto config = Spatial::c_spline_not_a_knot();
  EXPECT_TRUE(std::holds_alternative<Spline>(config.method()));
  EXPECT_EQ(std::get<Spline>(config.method()), Spline::kCSplineNotAKnot);
}

TEST_F(WindowedSpatialTest, CSplinePeriodicFactoryMethod) {
  auto config = Spatial::c_spline_periodic();
  EXPECT_TRUE(std::holds_alternative<Spline>(config.method()));
  EXPECT_EQ(std::get<Spline>(config.method()), Spline::kCSplinePeriodic);
}

TEST_F(WindowedSpatialTest, SteffenFactoryMethod) {
  auto config = Spatial::steffen();
  EXPECT_TRUE(std::holds_alternative<Spline>(config.method()));
  EXPECT_EQ(std::get<Spline>(config.method()), Spline::kSteffen);
}

TEST_F(WindowedSpatialTest, PolynomialFactoryMethod) {
  auto config = Spatial::polynomial();
  EXPECT_TRUE(std::holds_alternative<Spline>(config.method()));
  EXPECT_EQ(std::get<Spline>(config.method()), Spline::kPolynomial);
}

TEST_F(WindowedSpatialTest, WithBoundaryMode) {
  Spatial config;
  auto updated = config.with_boundary_mode(BoundaryConfig::shrink());
  EXPECT_EQ(updated.boundary_mode(), BoundaryMode::kShrink);
  EXPECT_EQ(config.boundary_mode(), BoundaryMode::kUndef);
}

TEST_F(WindowedSpatialTest, WithWindowSizeX) {
  Spatial config;
  auto updated = config.with_half_window_size_x(5);
  EXPECT_EQ(updated.half_window_size_x(), 5);
  EXPECT_EQ(config.half_window_size_x(), 3);
}

TEST_F(WindowedSpatialTest, WithWindowSizeY) {
  Spatial config;
  auto updated = config.with_half_window_size_y(7);
  EXPECT_EQ(updated.half_window_size_y(), 7);
  EXPECT_EQ(config.half_window_size_y(), 3);
}

TEST_F(WindowedSpatialTest, MethodChaining) {
  Spatial config;
  auto updated = config.with_boundary_mode(BoundaryConfig::shrink())
                     .with_half_window_size_x(4)
                     .with_half_window_size_y(6);
  EXPECT_EQ(updated.boundary_mode(), BoundaryMode::kShrink);
  EXPECT_EQ(updated.half_window_size_x(), 4);
  EXPECT_EQ(updated.half_window_size_y(), 6);
}

// ============================================================================
// UnivariateMethod Config Tests
// ============================================================================

class UnivariateMethodTest : public ::testing::Test {};

TEST_F(UnivariateMethodTest, DefaultConstructor) {
  UnivariateMethod config;
  EXPECT_EQ(config.method(), Spline::kLinear);
  EXPECT_EQ(config.boundary_mode(), BoundaryMode::kUndef);
  EXPECT_EQ(config.half_window_size(), 3);
}

TEST_F(UnivariateMethodTest, ConstructorWithMethod) {
  UnivariateMethod config(Spline::kAkima);
  EXPECT_EQ(config.method(), Spline::kAkima);
}

TEST_F(UnivariateMethodTest, LinearFactoryMethod) {
  auto config = UnivariateMethod::linear();
  EXPECT_EQ(config.method(), Spline::kLinear);
}

TEST_F(UnivariateMethodTest, AkimaFactoryMethod) {
  auto config = UnivariateMethod::akima();
  EXPECT_EQ(config.method(), Spline::kAkima);
}

TEST_F(UnivariateMethodTest, AkimaPeriodicFactoryMethod) {
  auto config = UnivariateMethod::akima_periodic();
  EXPECT_EQ(config.method(), Spline::kAkimaPeriodic);
}

TEST_F(UnivariateMethodTest, CSplineFactoryMethod) {
  auto config = UnivariateMethod::c_spline();
  EXPECT_EQ(config.method(), Spline::kCSpline);
}

TEST_F(UnivariateMethodTest, CSplineNotAKnotFactoryMethod) {
  auto config = UnivariateMethod::c_spline_not_a_knot();
  EXPECT_EQ(config.method(), Spline::kCSplineNotAKnot);
}

TEST_F(UnivariateMethodTest, CSplinePeriodicFactoryMethod) {
  auto config = UnivariateMethod::c_spline_periodic();
  EXPECT_EQ(config.method(), Spline::kCSplinePeriodic);
}

TEST_F(UnivariateMethodTest, SteffenFactoryMethod) {
  auto config = UnivariateMethod::steffen();
  EXPECT_EQ(config.method(), Spline::kSteffen);
}

TEST_F(UnivariateMethodTest, PolynomialFactoryMethod) {
  auto config = UnivariateMethod::polynomial();
  EXPECT_EQ(config.method(), Spline::kPolynomial);
}

TEST_F(UnivariateMethodTest, WithBoundaryMode) {
  UnivariateMethod config;
  auto updated = config.with_boundary_mode(BoundaryConfig::shrink());
  EXPECT_EQ(updated.boundary_mode(), BoundaryMode::kShrink);
  EXPECT_EQ(config.boundary_mode(), BoundaryMode::kUndef);
}

TEST_F(UnivariateMethodTest, WithWindowSize) {
  UnivariateMethod config;
  auto updated = config.with_half_window_size(5);
  EXPECT_EQ(updated.half_window_size(), 5);
  EXPECT_EQ(config.half_window_size(), 3);
}

TEST_F(UnivariateMethodTest, MethodChaining) {
  UnivariateMethod config;
  auto updated = config.with_boundary_mode(BoundaryConfig::shrink())
                     .with_half_window_size(7);
  EXPECT_EQ(updated.boundary_mode(), BoundaryMode::kShrink);
  EXPECT_EQ(updated.half_window_size(), 7);
}

// ============================================================================
// Univariate Config Tests
// ============================================================================

class UnivariateTest : public ::testing::Test {};

TEST_F(UnivariateTest, DefaultConstructor) {
  Univariate config;
  EXPECT_EQ(config.univariate().method(), Spline::kLinear);
  EXPECT_FALSE(config.common().bounds_error());
  EXPECT_EQ(config.common().num_threads(), std::thread::hardware_concurrency());
}

TEST_F(UnivariateTest, ConstructorWithUnivariateMethod) {
  UnivariateMethod method = UnivariateMethod::akima();
  Univariate config(method);
  EXPECT_EQ(config.univariate().method(), Spline::kAkima);
}

TEST_F(UnivariateTest, LinearFactoryMethod) {
  auto config = Univariate::linear();
  EXPECT_EQ(config.univariate().method(), Spline::kLinear);
}

TEST_F(UnivariateTest, AkimaFactoryMethod) {
  auto config = Univariate::akima();
  EXPECT_EQ(config.univariate().method(), Spline::kAkima);
}

TEST_F(UnivariateTest, WithWindowSize) {
  Univariate config;
  auto updated = config.with_half_window_size(5);
  EXPECT_EQ(updated.univariate().half_window_size(), 5);
}

TEST_F(UnivariateTest, WithBoundaryMode) {
  Univariate config;
  auto updated = config.with_boundary_mode(BoundaryConfig::shrink());
  EXPECT_EQ(updated.univariate().boundary_mode(), BoundaryMode::kShrink);
}

TEST_F(UnivariateTest, WithBoundsError) {
  Univariate config;
  auto updated = config.with_bounds_error(true);
  EXPECT_TRUE(updated.common().bounds_error());
}

TEST_F(UnivariateTest, WithNumThreads) {
  Univariate config;
  auto updated = config.with_num_threads(4);
  EXPECT_EQ(updated.common().num_threads(), 4);
}

TEST_F(UnivariateTest, MethodChaining) {
  Univariate config;
  auto updated = config.with_half_window_size(7)
                     .with_boundary_mode(BoundaryConfig::shrink())
                     .with_bounds_error(true)
                     .with_num_threads(8);
  EXPECT_EQ(updated.univariate().half_window_size(), 7);
  EXPECT_EQ(updated.univariate().boundary_mode(), BoundaryMode::kShrink);
  EXPECT_TRUE(updated.common().bounds_error());
  EXPECT_EQ(updated.common().num_threads(), 8);
}

// ============================================================================
// Bivariate Config Tests
// ============================================================================

class WindowedBivariateTest : public ::testing::Test {};

TEST_F(WindowedBivariateTest, DefaultConstructor) {
  Bivariate config;
  EXPECT_TRUE(std::holds_alternative<Bicubic>(config.spatial().method()));
  EXPECT_EQ(std::get<Bicubic>(config.spatial().method()), Bicubic::kBicubic);
  EXPECT_FALSE(config.common().bounds_error());
}

TEST_F(WindowedBivariateTest, BicubicFactoryMethod) {
  auto config = Bivariate::bicubic();
  EXPECT_TRUE(std::holds_alternative<Bicubic>(config.spatial().method()));
  EXPECT_EQ(std::get<Bicubic>(config.spatial().method()), Bicubic::kBicubic);
}

TEST_F(WindowedBivariateTest, BilinearFactoryMethod) {
  auto config = Bivariate::bilinear();
  EXPECT_TRUE(std::holds_alternative<Bicubic>(config.spatial().method()));
  EXPECT_EQ(std::get<Bicubic>(config.spatial().method()), Bicubic::kBilinear);
}

TEST_F(WindowedBivariateTest, LinearFactoryMethod) {
  auto config = Bivariate::linear();
  EXPECT_TRUE(std::holds_alternative<Spline>(config.spatial().method()));
  EXPECT_EQ(std::get<Spline>(config.spatial().method()), Spline::kLinear);
}

TEST_F(WindowedBivariateTest, WithHalfWindowSizeX) {
  Bivariate config;
  auto updated = config.with_half_window_size_x(5);
  EXPECT_EQ(updated.spatial().half_window_size_x(), 5);
}

TEST_F(WindowedBivariateTest, WithHalfWindowSizeY) {
  Bivariate config;
  auto updated = config.with_half_window_size_y(7);
  EXPECT_EQ(updated.spatial().half_window_size_y(), 7);
}

TEST_F(WindowedBivariateTest, WithBoundaryMode) {
  Bivariate config;
  auto updated = config.with_boundary_mode(BoundaryConfig::shrink());
  EXPECT_EQ(updated.spatial().boundary_mode(), BoundaryMode::kShrink);
}

TEST_F(WindowedBivariateTest, WithBoundsError) {
  Bivariate config;
  auto updated = config.with_bounds_error(true);
  EXPECT_TRUE(updated.common().bounds_error());
}

TEST_F(WindowedBivariateTest, WithNumThreads) {
  Bivariate config;
  auto updated = config.with_num_threads(4);
  EXPECT_EQ(updated.common().num_threads(), 4);
}

TEST_F(WindowedBivariateTest, MethodChaining) {
  Bivariate config;
  auto updated = config.with_half_window_size_x(4)
                     .with_half_window_size_y(6)
                     .with_boundary_mode(BoundaryConfig::shrink())
                     .with_bounds_error(true)
                     .with_num_threads(8);
  EXPECT_EQ(updated.spatial().half_window_size_x(), 4);
  EXPECT_EQ(updated.spatial().half_window_size_y(), 6);
  EXPECT_EQ(updated.spatial().boundary_mode(), BoundaryMode::kShrink);
  EXPECT_TRUE(updated.common().bounds_error());
  EXPECT_EQ(updated.common().num_threads(), 8);
}

// ============================================================================
// Trivariate Config Tests
// ============================================================================

class WindowedTrivariateTest : public ::testing::Test {};

TEST_F(WindowedTrivariateTest, DefaultConstructor) {
  Trivariate config;
  EXPECT_TRUE(std::holds_alternative<Bicubic>(config.spatial().method()));
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kLinear);
  EXPECT_FALSE(config.common().bounds_error());
}

TEST_F(WindowedTrivariateTest, BicubicFactoryMethod) {
  auto config = Trivariate::bicubic();
  EXPECT_TRUE(std::holds_alternative<Bicubic>(config.spatial().method()));
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kLinear);
}

TEST_F(WindowedTrivariateTest, LinearFactoryMethod) {
  auto config = Trivariate::linear();
  EXPECT_TRUE(std::holds_alternative<Spline>(config.spatial().method()));
  EXPECT_EQ(std::get<Spline>(config.spatial().method()), Spline::kLinear);
}

TEST_F(WindowedTrivariateTest, WithThirdAxis) {
  Trivariate config;
  auto updated = config.with_third_axis(AxisConfig::nearest());
  EXPECT_EQ(updated.third_axis().method(), AxisMethod::kNearest);
}

TEST_F(WindowedTrivariateTest, WithHalfWindowSizeX) {
  Trivariate config;
  auto updated = config.with_half_window_size_x(5);
  EXPECT_EQ(updated.spatial().half_window_size_x(), 5);
}

TEST_F(WindowedTrivariateTest, WithHalfWindowSizeY) {
  Trivariate config;
  auto updated = config.with_half_window_size_y(7);
  EXPECT_EQ(updated.spatial().half_window_size_y(), 7);
}

TEST_F(WindowedTrivariateTest, WithBoundaryMode) {
  Trivariate config;
  auto updated = config.with_boundary_mode(BoundaryConfig::shrink());
  EXPECT_EQ(updated.spatial().boundary_mode(), BoundaryMode::kShrink);
}

TEST_F(WindowedTrivariateTest, WithBoundsError) {
  Trivariate config;
  auto updated = config.with_bounds_error(true);
  EXPECT_TRUE(updated.common().bounds_error());
}

TEST_F(WindowedTrivariateTest, WithNumThreads) {
  Trivariate config;
  auto updated = config.with_num_threads(4);
  EXPECT_EQ(updated.common().num_threads(), 4);
}

TEST_F(WindowedTrivariateTest, MethodChaining) {
  Trivariate config;
  auto updated = config.with_third_axis(AxisConfig::nearest())
                     .with_half_window_size_x(4)
                     .with_half_window_size_y(6)
                     .with_boundary_mode(BoundaryConfig::shrink())
                     .with_bounds_error(true)
                     .with_num_threads(8);
  EXPECT_EQ(updated.third_axis().method(), AxisMethod::kNearest);
  EXPECT_EQ(updated.spatial().half_window_size_x(), 4);
  EXPECT_EQ(updated.spatial().half_window_size_y(), 6);
  EXPECT_EQ(updated.spatial().boundary_mode(), BoundaryMode::kShrink);
  EXPECT_TRUE(updated.common().bounds_error());
  EXPECT_EQ(updated.common().num_threads(), 8);
}

// ============================================================================
// Quadrivariate Config Tests
// ============================================================================

class WindowedQuadrivariateTest : public ::testing::Test {};

TEST_F(WindowedQuadrivariateTest, DefaultConstructor) {
  Quadrivariate config;
  EXPECT_TRUE(std::holds_alternative<Bicubic>(config.spatial().method()));
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kLinear);
  EXPECT_EQ(config.fourth_axis().method(), AxisMethod::kLinear);
  EXPECT_FALSE(config.common().bounds_error());
}

TEST_F(WindowedQuadrivariateTest, BicubicFactoryMethod) {
  auto config = Quadrivariate::bicubic();
  EXPECT_TRUE(std::holds_alternative<Bicubic>(config.spatial().method()));
  EXPECT_EQ(config.third_axis().method(), AxisMethod::kLinear);
  EXPECT_EQ(config.fourth_axis().method(), AxisMethod::kLinear);
}

TEST_F(WindowedQuadrivariateTest, LinearFactoryMethod) {
  auto config = Quadrivariate::linear();
  EXPECT_TRUE(std::holds_alternative<Spline>(config.spatial().method()));
  EXPECT_EQ(std::get<Spline>(config.spatial().method()), Spline::kLinear);
}

TEST_F(WindowedQuadrivariateTest, WithThirdAxis) {
  Quadrivariate config;
  auto updated = config.with_third_axis(AxisConfig::nearest());
  EXPECT_EQ(updated.third_axis().method(), AxisMethod::kNearest);
}

TEST_F(WindowedQuadrivariateTest, WithFourthAxis) {
  Quadrivariate config;
  auto updated = config.with_fourth_axis(AxisConfig::nearest());
  EXPECT_EQ(updated.fourth_axis().method(), AxisMethod::kNearest);
}

TEST_F(WindowedQuadrivariateTest, WithHalfWindowSizeX) {
  Quadrivariate config;
  auto updated = config.with_half_window_size_x(5);
  EXPECT_EQ(updated.spatial().half_window_size_x(), 5);
}

TEST_F(WindowedQuadrivariateTest, WithHalfWindowSizeY) {
  Quadrivariate config;
  auto updated = config.with_half_window_size_y(7);
  EXPECT_EQ(updated.spatial().half_window_size_y(), 7);
}

TEST_F(WindowedQuadrivariateTest, WithBoundaryMode) {
  Quadrivariate config;
  auto updated = config.with_boundary_mode(BoundaryConfig::shrink());
  EXPECT_EQ(updated.spatial().boundary_mode(), BoundaryMode::kShrink);
}

TEST_F(WindowedQuadrivariateTest, WithBoundsError) {
  Quadrivariate config;
  auto updated = config.with_bounds_error(true);
  EXPECT_TRUE(updated.common().bounds_error());
}

TEST_F(WindowedQuadrivariateTest, WithNumThreads) {
  Quadrivariate config;
  auto updated = config.with_num_threads(4);
  EXPECT_EQ(updated.common().num_threads(), 4);
}

TEST_F(WindowedQuadrivariateTest, MethodChaining) {
  Quadrivariate config;
  auto updated = config.with_third_axis(AxisConfig::nearest())
                     .with_fourth_axis(AxisConfig::nearest())
                     .with_half_window_size_x(4)
                     .with_half_window_size_y(6)
                     .with_boundary_mode(BoundaryConfig::shrink())
                     .with_bounds_error(true)
                     .with_num_threads(8);
  EXPECT_EQ(updated.third_axis().method(), AxisMethod::kNearest);
  EXPECT_EQ(updated.fourth_axis().method(), AxisMethod::kNearest);
  EXPECT_EQ(updated.spatial().half_window_size_x(), 4);
  EXPECT_EQ(updated.spatial().half_window_size_y(), 6);
  EXPECT_EQ(updated.spatial().boundary_mode(), BoundaryMode::kShrink);
  EXPECT_TRUE(updated.common().bounds_error());
  EXPECT_EQ(updated.common().num_threads(), 8);
}

}  // namespace pyinterp::config::windowed

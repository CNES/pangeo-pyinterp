// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/math/interpolate/geometric/multivariate.hpp"

#include <gtest/gtest.h>

#include <boost/geometry.hpp>
#include <boost/geometry/core/access.hpp>
#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/point.hpp>

namespace pyinterp::math::interpolate::geometric {

// Point alias compatible with header (template< class >)
template <class T>
using Point2D =
    boost::geometry::model::point<T, 2, boost::geometry::cs::cartesian>;

// ============================================================================
// AXIS INTERPOLATOR FACTORY TESTS
// ============================================================================

TEST(GeometryMultivariate, GetAxisInterpolatorLinear) {
  auto interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);
  // Test linear interpolation at midpoint
  EXPECT_DOUBLE_EQ(interpolator(0.5, 0.0, 1.0, 10.0, 20.0), 15.0);
  // Test at endpoints
  EXPECT_DOUBLE_EQ(interpolator(0.0, 0.0, 1.0, 10.0, 20.0), 10.0);
  EXPECT_DOUBLE_EQ(interpolator(1.0, 0.0, 1.0, 10.0, 20.0), 20.0);
}

TEST(GeometryMultivariate, GetAxisInterpolatorNearest) {
  auto interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kNearest);
  // Test nearest neighbor interpolation
  EXPECT_DOUBLE_EQ(interpolator(0.3, 0.0, 1.0, 10.0, 20.0), 10.0);
  EXPECT_DOUBLE_EQ(interpolator(0.6, 0.0, 1.0, 10.0, 20.0), 20.0);
  // Test at exact midpoint (when distances are equal, returns upper value)
  EXPECT_DOUBLE_EQ(interpolator(0.5, 0.0, 1.0, 10.0, 20.0), 20.0);
}

// ============================================================================
// TRIVARIATE INTERPOLATION TESTS
// ============================================================================

TEST(GeometryMultivariate, TrivariateLinearCorners) {
  // Test at all 8 corners of the cube
  Bilinear<Point2D, double> bilinear;
  auto z_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);

  SpatialPoint3D<Point2D, double> p0{Point2D<double>{0.0, 0.0}, 0.0};
  SpatialPoint3D<Point2D, double> p1{Point2D<double>{1.0, 1.0}, 1.0};

  DataCube<double> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

  // Corner (0,0,0) -> q000 = 1.0
  auto result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{0.0, 0.0}, 0.0}, p0, p1,
      data, bilinear, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 1.0);

  // Corner (0,1,0) -> q010 = 2.0
  result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{0.0, 1.0}, 0.0}, p0, p1,
      data, bilinear, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 2.0);

  // Corner (1,0,0) -> q100 = 3.0
  result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{1.0, 0.0}, 0.0}, p0, p1,
      data, bilinear, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 3.0);

  // Corner (1,1,0) -> q110 = 4.0
  result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{1.0, 1.0}, 0.0}, p0, p1,
      data, bilinear, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 4.0);

  // Corner (0,0,1) -> q001 = 5.0
  result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{0.0, 0.0}, 1.0}, p0, p1,
      data, bilinear, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 5.0);

  // Corner (0,1,1) -> q011 = 6.0
  result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{0.0, 1.0}, 1.0}, p0, p1,
      data, bilinear, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 6.0);

  // Corner (1,0,1) -> q101 = 7.0
  result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{1.0, 0.0}, 1.0}, p0, p1,
      data, bilinear, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 7.0);

  // Corner (1,1,1) -> q111 = 8.0
  result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{1.0, 1.0}, 1.0}, p0, p1,
      data, bilinear, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 8.0);
}

TEST(GeometryMultivariate, TrivariateLinearCenter) {
  // Test at the center of the cube (0.5, 0.5, 0.5)
  // Expected: average of all 8 corners
  Bilinear<Point2D, double> bilinear;
  auto z_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);

  SpatialPoint3D<Point2D, double> p0{Point2D<double>{0.0, 0.0}, 0.0};
  SpatialPoint3D<Point2D, double> p1{Point2D<double>{1.0, 1.0}, 1.0};

  DataCube<double> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

  auto result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{0.5, 0.5}, 0.5}, p0, p1,
      data, bilinear, z_interpolator);

  // Center should be the average: (1+2+3+4+5+6+7+8)/8 = 4.5
  EXPECT_DOUBLE_EQ(result, 4.5);
}

TEST(GeometryMultivariate, TrivariateLinearMidplanes) {
  // Test on the midplanes (z=0.5 plane, x=0.5 plane, y=0.5 plane)
  Bilinear<Point2D, double> bilinear;
  auto z_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);

  SpatialPoint3D<Point2D, double> p0{Point2D<double>{0.0, 0.0}, 0.0};
  SpatialPoint3D<Point2D, double> p1{Point2D<double>{1.0, 1.0}, 1.0};

  DataCube<double> data{0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0};

  // Test at (0, 0, 0.5): interpolate between q000 and q001
  auto result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{0.0, 0.0}, 0.5}, p0, p1,
      data, bilinear, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 20.0);  // (0 + 40) / 2

  // Test at (1, 1, 0.5): interpolate between q110 and q111
  result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{1.0, 1.0}, 0.5}, p0, p1,
      data, bilinear, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 50.0);  // (30 + 70) / 2
}

TEST(GeometryMultivariate, TrivariateNearestCorners) {
  // Test nearest neighbor at various points
  Nearest<Point2D, double> nearest;
  auto z_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kNearest);

  SpatialPoint3D<Point2D, double> p0{Point2D<double>{0.0, 0.0}, 0.0};
  SpatialPoint3D<Point2D, double> p1{Point2D<double>{1.0, 1.0}, 1.0};

  DataCube<double> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

  // Point close to (0, 0, 0) -> should return 1.0
  auto result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{0.1, 0.1}, 0.1}, p0, p1,
      data, nearest, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 1.0);

  // Point close to (1, 1, 1) -> should return 8.0
  result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{0.9, 0.9}, 0.9}, p0, p1,
      data, nearest, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 8.0);

  // Point close to (0, 1, 0) -> should return 2.0
  result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{0.1, 0.9}, 0.1}, p0, p1,
      data, nearest, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 2.0);
}

TEST(GeometryMultivariate, TrivariateMixedInterpolation) {
  // Use bilinear for spatial, nearest for z-axis
  Bilinear<Point2D, double> bilinear;
  auto z_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kNearest);

  SpatialPoint3D<Point2D, double> p0{Point2D<double>{0.0, 0.0}, 0.0};
  SpatialPoint3D<Point2D, double> p1{Point2D<double>{1.0, 1.0}, 1.0};

  DataCube<double> data{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0};

  // At (0.5, 0.5, 0.3): spatial center, z closer to 0
  // z=0 plane center: (10+20+30+40)/4 = 25
  // Since z=0.3 is closer to 0 than 1, should return 25
  auto result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{0.5, 0.5}, 0.3}, p0, p1,
      data, bilinear, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 25.0);

  // At (0.5, 0.5, 0.7): spatial center, z closer to 1
  // z=1 plane center: (50+60+70+80)/4 = 65
  result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{0.5, 0.5}, 0.7}, p0, p1,
      data, bilinear, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 65.0);
}

TEST(GeometryMultivariate, TrivariateWithDifferentAxisType) {
  // Test with different axis type (int for z-axis)
  Bilinear<Point2D, double> bilinear;
  auto z_interpolator = get_axis_interpolator<int, double>(AxisMethod::kLinear);

  SpatialPoint3D<Point2D, double, int> p0{Point2D<double>{0.0, 0.0}, 0};
  SpatialPoint3D<Point2D, double, int> p1{Point2D<double>{1.0, 1.0}, 10};

  DataCube<double> data{100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0};

  // At z=5 (midpoint), spatial (0.5, 0.5)
  auto result = trivariate(
      SpatialPoint3D<Point2D, double, int>{Point2D<double>{0.5, 0.5}, 5}, p0,
      p1, data, bilinear, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 450.0);  // Average of all corners
}

// ============================================================================
// QUADRIVARIATE INTERPOLATION TESTS
// ============================================================================

TEST(GeometryMultivariate, QuadrivariateLinearCorners) {
  // Test at all 16 corners of the hypercube
  Bilinear<Point2D, double> bilinear;
  auto z_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);
  auto u_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);

  SpatialPoint4D<Point2D, double> p0{Point2D<double>{0.0, 0.0}, 0.0, 0.0};
  SpatialPoint4D<Point2D, double> p1{Point2D<double>{1.0, 1.0}, 1.0, 1.0};

  DataHypercube<double> data{1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

  // Test corner (0,0,0,0) -> q0000 = 1.0
  auto result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{0.0, 0.0}, 0.0, 0.0}, p0,
      p1, data, bilinear, z_interpolator, u_interpolator);
  EXPECT_DOUBLE_EQ(result, 1.0);

  // Test corner (1,1,1,1) -> q1111 = 16.0
  result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{1.0, 1.0}, 1.0, 1.0}, p0,
      p1, data, bilinear, z_interpolator, u_interpolator);
  EXPECT_DOUBLE_EQ(result, 16.0);

  // Test corner (0,1,0,0) -> q0100 = 2.0
  result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{0.0, 1.0}, 0.0, 0.0}, p0,
      p1, data, bilinear, z_interpolator, u_interpolator);
  EXPECT_DOUBLE_EQ(result, 2.0);

  // Test corner (1,0,1,0) -> q1010 = 7.0
  result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{1.0, 0.0}, 1.0, 0.0}, p0,
      p1, data, bilinear, z_interpolator, u_interpolator);
  EXPECT_DOUBLE_EQ(result, 7.0);

  // Test corner (0,0,1,1) -> q0011 = 13.0
  result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{0.0, 0.0}, 1.0, 1.0}, p0,
      p1, data, bilinear, z_interpolator, u_interpolator);
  EXPECT_DOUBLE_EQ(result, 13.0);
}

TEST(GeometryMultivariate, QuadrivariateLinearCenter) {
  // Test at the center of the hypercube (0.5, 0.5, 0.5, 0.5)
  // Expected: average of all 16 corners
  Bilinear<Point2D, double> bilinear;
  auto z_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);
  auto u_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);

  SpatialPoint4D<Point2D, double> p0{Point2D<double>{0.0, 0.0}, 0.0, 0.0};
  SpatialPoint4D<Point2D, double> p1{Point2D<double>{1.0, 1.0}, 1.0, 1.0};

  DataHypercube<double> data{1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

  auto result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{0.5, 0.5}, 0.5, 0.5}, p0,
      p1, data, bilinear, z_interpolator, u_interpolator);

  // Center should be the average: (1+2+...+16)/16 = 8.5
  EXPECT_DOUBLE_EQ(result, 8.5);
}

TEST(GeometryMultivariate, QuadrivariateFixedU) {
  // Test interpolation with u fixed at 0 (should reduce to trivariate)
  Bilinear<Point2D, double> bilinear;
  auto z_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);
  auto u_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);

  SpatialPoint4D<Point2D, double> p0{Point2D<double>{0.0, 0.0}, 0.0, 0.0};
  SpatialPoint4D<Point2D, double> p1{Point2D<double>{1.0, 1.0}, 1.0, 1.0};

  DataHypercube<double> data{10.0,  20.0,  30.0,  40.0,  50.0,  60.0,
                             70.0,  80.0,  100.0, 110.0, 120.0, 130.0,
                             140.0, 150.0, 160.0, 170.0};

  // At u=0, center of spatial and z
  auto result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{0.5, 0.5}, 0.5, 0.0}, p0,
      p1, data, bilinear, z_interpolator, u_interpolator);

  // Should be average of first 8 values: (10+20+30+40+50+60+70+80)/8 = 45
  EXPECT_DOUBLE_EQ(result, 45.0);

  // At u=1, center of spatial and z
  result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{0.5, 0.5}, 0.5, 1.0}, p0,
      p1, data, bilinear, z_interpolator, u_interpolator);

  // Should be average of last 8 values: (100+110+120+130+140+150+160+170)/8 =
  // 135
  EXPECT_DOUBLE_EQ(result, 135.0);
}

TEST(GeometryMultivariate, QuadrivariateNearestCorner) {
  // Test nearest neighbor at various points
  Nearest<Point2D, double> nearest;
  auto z_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kNearest);
  auto u_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kNearest);

  SpatialPoint4D<Point2D, double> p0{Point2D<double>{0.0, 0.0}, 0.0, 0.0};
  SpatialPoint4D<Point2D, double> p1{Point2D<double>{1.0, 1.0}, 1.0, 1.0};

  DataHypercube<double> data{1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

  // Point close to (0, 0, 0, 0) -> should return 1.0
  auto result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{0.1, 0.1}, 0.1, 0.1}, p0,
      p1, data, nearest, z_interpolator, u_interpolator);
  EXPECT_DOUBLE_EQ(result, 1.0);

  // Point close to (1, 1, 1, 1) -> should return 16.0
  result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{0.9, 0.9}, 0.9, 0.9}, p0,
      p1, data, nearest, z_interpolator, u_interpolator);
  EXPECT_DOUBLE_EQ(result, 16.0);

  // Point close to (1, 0, 1, 0) -> should return 7.0
  result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{0.9, 0.1}, 0.9, 0.1}, p0,
      p1, data, nearest, z_interpolator, u_interpolator);
  EXPECT_DOUBLE_EQ(result, 7.0);
}

TEST(GeometryMultivariate, QuadrivariateMixedInterpolation) {
  // Use bilinear for spatial, linear for z, nearest for u
  Bilinear<Point2D, double> bilinear;
  auto z_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);
  auto u_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kNearest);

  SpatialPoint4D<Point2D, double> p0{Point2D<double>{0.0, 0.0}, 0.0, 0.0};
  SpatialPoint4D<Point2D, double> p1{Point2D<double>{1.0, 1.0}, 1.0, 1.0};

  DataHypercube<double> data{10.0,  20.0,  30.0,  40.0,  50.0,  60.0,
                             70.0,  80.0,  100.0, 110.0, 120.0, 130.0,
                             140.0, 150.0, 160.0, 170.0};

  // At (0.5, 0.5, 0.5, 0.3): spatial center, z center, u closer to 0
  // u=0: center of spatial+z: (10+20+30+40+50+60+70+80)/8 = 45
  auto result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{0.5, 0.5}, 0.5, 0.3}, p0,
      p1, data, bilinear, z_interpolator, u_interpolator);
  EXPECT_DOUBLE_EQ(result, 45.0);

  // At (0.5, 0.5, 0.5, 0.7): spatial center, z center, u closer to 1
  // u=1: center of spatial+z: (100+110+120+130+140+150+160+170)/8 = 135
  result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{0.5, 0.5}, 0.5, 0.7}, p0,
      p1, data, bilinear, z_interpolator, u_interpolator);
  EXPECT_DOUBLE_EQ(result, 135.0);
}

TEST(GeometryMultivariate, QuadrivariateWithDifferentAxisTypes) {
  // Test with different axis types (int for z, float for u)
  Bilinear<Point2D, double> bilinear;
  auto z_interpolator = get_axis_interpolator<int, double>(AxisMethod::kLinear);
  auto u_interpolator =
      get_axis_interpolator<float, double>(AxisMethod::kLinear);

  SpatialPoint4D<Point2D, double, int, float> p0{Point2D<double>{0.0, 0.0}, 0,
                                                 0.0f};
  SpatialPoint4D<Point2D, double, int, float> p1{Point2D<double>{1.0, 1.0}, 10,
                                                 1.0f};

  DataHypercube<double> data{0.0,   10.0,  20.0,  30.0, 40.0,  50.0,
                             60.0,  70.0,  80.0,  90.0, 100.0, 110.0,
                             120.0, 130.0, 140.0, 150.0};

  // At z=5 (midpoint), u=0.5 (midpoint), spatial (0.5, 0.5)
  auto result = quadrivariate(
      SpatialPoint4D<Point2D, double, int, float>{Point2D<double>{0.5, 0.5}, 5,
                                                  0.5f},
      p0, p1, data, bilinear, z_interpolator, u_interpolator);

  // Should be average of all 16 values: (0+10+...+150)/16 = 75
  EXPECT_DOUBLE_EQ(result, 75.0);
}

TEST(GeometryMultivariate, QuadrivariateAsymmetricValues) {
  // Test with non-uniform values to ensure proper interpolation
  Bilinear<Point2D, double> bilinear;
  auto z_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);
  auto u_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);

  SpatialPoint4D<Point2D, double> p0{Point2D<double>{0.0, 0.0}, 0.0, 0.0};
  SpatialPoint4D<Point2D, double> p1{Point2D<double>{2.0, 2.0}, 2.0, 2.0};

  DataHypercube<double> data{1.0,   5.0,   10.0,  20.0,  30.0,  40.0,
                             50.0,  60.0,  100.0, 110.0, 120.0, 130.0,
                             140.0, 150.0, 160.0, 200.0};

  // Test at (1.0, 1.0, 1.0, 1.0) - exact center
  auto result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{1.0, 1.0}, 1.0, 1.0}, p0,
      p1, data, bilinear, z_interpolator, u_interpolator);

  double expected =
      (1.0 + 5.0 + 10.0 + 20.0 + 30.0 + 40.0 + 50.0 + 60.0 + 100.0 + 110.0 +
       120.0 + 130.0 + 140.0 + 150.0 + 160.0 + 200.0) /
      16.0;
  EXPECT_DOUBLE_EQ(result, expected);
}

// ============================================================================
// DATA STRUCTURE TESTS
// ============================================================================

TEST(GeometryMultivariate, SpatialPoint3DConstruction) {
  SpatialPoint3D<Point2D, double> pt{Point2D<double>{1.5, 2.5}, 3.5};
  EXPECT_DOUBLE_EQ(boost::geometry::get<0>(pt.spatial), 1.5);
  EXPECT_DOUBLE_EQ(boost::geometry::get<1>(pt.spatial), 2.5);
  EXPECT_DOUBLE_EQ(pt.third_axis, 3.5);
}

TEST(GeometryMultivariate, SpatialPoint3DDefaultConstruction) {
  SpatialPoint3D<Point2D, double> pt;
  // Just verify it compiles and can be default constructed
  (void)pt;
}

TEST(GeometryMultivariate, SpatialPoint4DConstruction) {
  SpatialPoint4D<Point2D, double> pt{Point2D<double>{1.0, 2.0}, 3.0, 4.0};
  EXPECT_DOUBLE_EQ(boost::geometry::get<0>(pt.spatial), 1.0);
  EXPECT_DOUBLE_EQ(boost::geometry::get<1>(pt.spatial), 2.0);
  EXPECT_DOUBLE_EQ(pt.z_axis, 3.0);
  EXPECT_DOUBLE_EQ(pt.u_axis, 4.0);
}

TEST(GeometryMultivariate, SpatialPoint4DDefaultConstruction) {
  SpatialPoint4D<Point2D, double> pt;
  // Just verify it compiles and can be default constructed
  (void)pt;
}

TEST(GeometryMultivariate, DataCubeConstruction) {
  DataCube<double> cube{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  EXPECT_DOUBLE_EQ(cube.q000, 1.0);
  EXPECT_DOUBLE_EQ(cube.q010, 2.0);
  EXPECT_DOUBLE_EQ(cube.q100, 3.0);
  EXPECT_DOUBLE_EQ(cube.q110, 4.0);
  EXPECT_DOUBLE_EQ(cube.q001, 5.0);
  EXPECT_DOUBLE_EQ(cube.q011, 6.0);
  EXPECT_DOUBLE_EQ(cube.q101, 7.0);
  EXPECT_DOUBLE_EQ(cube.q111, 8.0);
}

TEST(GeometryMultivariate, DataCubeDefaultConstruction) {
  DataCube<double> cube;
  // Just verify it compiles and can be default constructed
  (void)cube;
}

TEST(GeometryMultivariate, DataHypercubeConstruction) {
  DataHypercube<double> hcube{1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                              9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  EXPECT_DOUBLE_EQ(hcube.q0000, 1.0);
  EXPECT_DOUBLE_EQ(hcube.q0100, 2.0);
  EXPECT_DOUBLE_EQ(hcube.q1111, 16.0);
}

TEST(GeometryMultivariate, DataHypercubeDefaultConstruction) {
  DataHypercube<double> hcube;
  // Just verify it compiles and can be default constructed
  (void)hcube;
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

TEST(GeometryMultivariate, TrivariateAllSameValues) {
  // When all corner values are the same, interpolation should return that value
  Bilinear<Point2D, double> bilinear;
  auto z_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);

  SpatialPoint3D<Point2D, double> p0{Point2D<double>{0.0, 0.0}, 0.0};
  SpatialPoint3D<Point2D, double> p1{Point2D<double>{1.0, 1.0}, 1.0};

  DataCube<double> data{42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0};

  auto result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{0.3, 0.7}, 0.5}, p0, p1,
      data, bilinear, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 42.0);
}

TEST(GeometryMultivariate, QuadrivariateAllSameValues) {
  // When all corner values are the same, interpolation should return that value
  Bilinear<Point2D, double> bilinear;
  auto z_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);
  auto u_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);

  SpatialPoint4D<Point2D, double> p0{Point2D<double>{0.0, 0.0}, 0.0, 0.0};
  SpatialPoint4D<Point2D, double> p1{Point2D<double>{1.0, 1.0}, 1.0, 1.0};

  DataHypercube<double> data{99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
                             99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0};

  auto result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{0.3, 0.7}, 0.5, 0.8}, p0,
      p1, data, bilinear, z_interpolator, u_interpolator);
  EXPECT_DOUBLE_EQ(result, 99.0);
}

TEST(GeometryMultivariate, TrivariateWithIDW) {
  // Test trivariate with IDW spatial interpolation
  InverseDistanceWeighting<Point2D, double> idw(2);
  auto z_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);

  SpatialPoint3D<Point2D, double> p0{Point2D<double>{0.0, 0.0}, 0.0};
  SpatialPoint3D<Point2D, double> p1{Point2D<double>{1.0, 1.0}, 1.0};

  DataCube<double> data{0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0};

  // At corner (exact match), should return corner value
  auto result = trivariate(
      SpatialPoint3D<Point2D, double>{Point2D<double>{0.0, 0.0}, 0.5}, p0, p1,
      data, idw, z_interpolator);
  EXPECT_DOUBLE_EQ(result, 20.0);  // Interpolate between q000=0 and q001=40
}

TEST(GeometryMultivariate, QuadrivariateWithIDW) {
  // Test quadrivariate with IDW spatial interpolation
  InverseDistanceWeighting<Point2D, double> idw(2);
  auto z_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);
  auto u_interpolator =
      get_axis_interpolator<double, double>(AxisMethod::kLinear);

  SpatialPoint4D<Point2D, double> p0{Point2D<double>{0.0, 0.0}, 0.0, 0.0};
  SpatialPoint4D<Point2D, double> p1{Point2D<double>{1.0, 1.0}, 1.0, 1.0};

  DataHypercube<double> data{0.0,   10.0,  20.0,  30.0, 40.0,  50.0,
                             60.0,  70.0,  80.0,  90.0, 100.0, 110.0,
                             120.0, 130.0, 140.0, 150.0};

  // At spatial corner (exact match), should return interpolated along z and u
  auto result = quadrivariate(
      SpatialPoint4D<Point2D, double>{Point2D<double>{0.0, 0.0}, 0.5, 0.5}, p0,
      p1, data, idw, z_interpolator, u_interpolator);

  // At (0,0): z0_u0=0, z1_u0=40, z0_u1=80, z1_u1=120
  // At z=0.5: u0=(0+40)/2=20, u1=(80+120)/2=100
  // At u=0.5: (20+100)/2=60
  EXPECT_DOUBLE_EQ(result, 60.0);
}

}  // namespace pyinterp::math::interpolate::geometric

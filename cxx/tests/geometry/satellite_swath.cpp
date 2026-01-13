// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <gtest/gtest.h>

#include <Eigen/Core>

#include "pyinterp/geometry/geographic/coordinates.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"
#include "pyinterp/geometry/point.hpp"
#include "pyinterp/geometry/satellite/transforms/swath.hpp"

namespace satellite = pyinterp::geometry::satellite;
namespace geographic = pyinterp::geometry::geographic;
namespace geometry = pyinterp::geometry;

// Test fixture for satellite swath tests
class SatelliteSwathTest : public ::testing::Test {
 protected:
  static constexpr double kEpsilon = 1e-9;        // 1 nanometer precision
  static constexpr double kAngleEpsilon = 1e-10;  // ~0.00001 arc-second

  // SWOT typical parameters (in km, will be converted to meters)
  static constexpr double kAcrossTrackResolutionKm = 2.0;  // km
  static constexpr double kAlongTrackResolutionKm = 2.0;   // km
  static constexpr double kHalfSwathKm = 70.0;             // km
  static constexpr double kHalfGapKm = 2.0;                // km

  static constexpr int kHalfSwathPixels =
      static_cast<int>((kHalfSwathKm - kHalfGapKm) / kAcrossTrackResolutionKm) +
      1;
  static constexpr double kAcrossTrackResolution =
      kAcrossTrackResolutionKm * 1000.0;                   // 2000 m
  static constexpr double kHalfGap = kHalfGapKm * 1000.0;  // 2000 m

  // Helper to create a simple nadir track (equatorial pass)
  auto create_equatorial_nadir_track(int num_points = 10)
      -> std::pair<Eigen::VectorXd, Eigen::VectorXd> {
    Eigen::VectorXd lon_nadir(num_points);
    Eigen::VectorXd lat_nadir(num_points);

    for (int ix = 0; ix < num_points; ++ix) {
      lon_nadir(ix) = -5.0 + ix * 1.0;  // -5° to +4° longitude
      lat_nadir(ix) = 0.0;              // Equator
    }

    return {lon_nadir, lat_nadir};
  }

  // Helper to create a polar nadir track
  auto create_polar_nadir_track(int num_points = 10)
      -> std::pair<Eigen::VectorXd, Eigen::VectorXd> {
    Eigen::VectorXd lon_nadir(num_points);
    Eigen::VectorXd lat_nadir(num_points);

    for (int ix = 0; ix < num_points; ++ix) {
      lon_nadir(ix) = 0.0;                // Prime meridian
      lat_nadir(ix) = -45.0 + ix * 10.0;  // -45° to +45° latitude
    }

    return {lon_nadir, lat_nadir};
  }

  // Helper to create a realistic SWOT orbit segment
  auto create_swot_orbit_track(int num_points = 10)
      -> std::pair<Eigen::VectorXd, Eigen::VectorXd> {
    Eigen::VectorXd lon_nadir(num_points);
    Eigen::VectorXd lat_nadir(num_points);

    for (int ix = 0; ix < num_points; ++ix) {
      double scaled_position = ix / static_cast<double>(num_points - 1);
      lon_nadir(ix) = 10.0 + scaled_position * 5.0;    // 10° to 15° longitude
      lat_nadir(ix) = -30.0 + scaled_position * 60.0;  // -30° to +30° latitude
    }

    return {lon_nadir, lat_nadir};
  }
};

// Test basic swath calculation with equatorial track
TEST_F(SatelliteSwathTest, EquatorialSwathCalculation) {
  auto [lon_nadir, lat_nadir] = create_equatorial_nadir_track(10);

  auto [lon_swath, lat_swath] = satellite::swath::calculate<double>(
      lon_nadir, lat_nadir, kAcrossTrackResolution, kHalfGap, kHalfSwathPixels,
      std::nullopt);

  // Verify output dimensions
  EXPECT_EQ(lon_swath.rows(), lon_nadir.rows());
  EXPECT_EQ(lon_swath.cols(), 2 * kHalfSwathPixels);
  EXPECT_EQ(lat_swath.rows(), lat_nadir.rows());
  EXPECT_EQ(lat_swath.cols(), 2 * kHalfSwathPixels);

  // For equatorial track, latitude should be symmetric around nadir
  for (int ix = 0; ix < lon_nadir.rows(); ++ix) {
    // Check that left and right sides are approximately symmetric in latitude
    for (int jx = 0; jx < kHalfSwathPixels; ++jx) {
      int left_idx = jx;
      int right_idx = 2 * kHalfSwathPixels - 1 - jx;

      // Latitude difference from nadir should be similar on both sides
      double left_delta = std::abs(lat_swath(ix, left_idx) - lat_nadir(ix));
      double right_delta = std::abs(lat_swath(ix, right_idx) - lat_nadir(ix));

      EXPECT_NEAR(left_delta, right_delta, 0.01)
          << "Asymmetric at row " << ix << ", pixel " << jx;
    }
  }

  // All values should be finite
  EXPECT_TRUE(lon_swath.allFinite());
  EXPECT_TRUE(lat_swath.allFinite());

  // Latitude should remain near equator (within ~1 degree for 70km swath)
  EXPECT_TRUE((lat_swath.array().abs() < 1.0).all())
      << "Latitude values too far from equator";
}

// Test swath calculation with polar track
TEST_F(SatelliteSwathTest, PolarSwathCalculation) {
  auto [lon_nadir, lat_nadir] = create_polar_nadir_track(10);

  auto [lon_swath, lat_swath] = satellite::swath::calculate<double>(
      lon_nadir, lat_nadir, kAcrossTrackResolution, kHalfGap, kHalfSwathPixels,
      std::nullopt);

  // Verify output dimensions
  EXPECT_EQ(lon_swath.rows(), lon_nadir.rows());
  EXPECT_EQ(lon_swath.cols(), 2 * kHalfSwathPixels);

  // All values should be finite
  EXPECT_TRUE(lon_swath.allFinite());
  EXPECT_TRUE(lat_swath.allFinite());

  // For polar track along prime meridian, longitude should spread
  // symmetrically
  for (int ix = 1; ix < lon_nadir.rows() - 1; ++ix) {
    // Note: There's a gap, so we don't have a pixel exactly at nadir
    // Check that longitudes are spread around 0°
    double lon_mean =
        (lon_swath.row(ix).minCoeff() + lon_swath.row(ix).maxCoeff()) / 2.0;
    EXPECT_NEAR(lon_mean, 0.0, 5.0)  // Within 5° of prime meridian
        << "Longitude not centered at row " << ix;
  }
}

// Test with realistic SWOT parameters
TEST_F(SatelliteSwathTest, SWOTRealisticParameters) {
  auto [lon_nadir, lat_nadir] = create_swot_orbit_track(20);

  auto [lon_swath, lat_swath] = satellite::swath::calculate<double>(
      lon_nadir, lat_nadir, kAcrossTrackResolution, kHalfGap, kHalfSwathPixels,
      std::nullopt);

  // Verify dimensions: 20 nadir points × 70 swath pixels (35 left + 35 right)
  EXPECT_EQ(lon_swath.rows(), 20);
  EXPECT_EQ(lon_swath.cols(), 2 * kHalfSwathPixels);  // 70 total pixels
  EXPECT_EQ(lat_swath.rows(), 20);
  EXPECT_EQ(lat_swath.cols(), 2 * kHalfSwathPixels);

  // All values should be finite
  EXPECT_TRUE(lon_swath.allFinite());
  EXPECT_TRUE(lat_swath.allFinite());

  // Verify that swath coordinates are within reasonable bounds
  EXPECT_TRUE((lon_swath.array() >= -180.0).all());
  EXPECT_TRUE((lon_swath.array() <= 180.0).all());
  EXPECT_TRUE((lat_swath.array() >= -90.0).all());
  EXPECT_TRUE((lat_swath.array() <= 90.0).all());

  // For pixel 0 (leftmost): -(34*2000 + 2000) = -70,000 m
  // For pixel 69 (rightmost): +(34*2000 + 2000) = +70,000 m
  auto coordinates = geographic::Coordinates(geographic::Spheroid());
  for (int ix = 1; ix < lon_nadir.rows() - 1; ++ix) {
    auto nadir_ecef =
        coordinates.lla_to_ecef<double>({lon_nadir(ix), lat_nadir(ix), 0.0});

    // Check leftmost pixel (should be at distance ~70 km from nadir)
    auto left_ecef = coordinates.lla_to_ecef<double>(
        {lon_swath(ix, 0), lat_swath(ix, 0), 0.0});

    double dist_left =
        std::sqrt(std::pow(left_ecef.get<0>() - nadir_ecef.get<0>(), 2) +
                  std::pow(left_ecef.get<1>() - nadir_ecef.get<1>(), 2) +
                  std::pow(left_ecef.get<2>() - nadir_ecef.get<2>(), 2));

    // Expected: (half_swath_pix - 1) * delta_ac + half_gap
    double expected_edge_dist =
        (kHalfSwathPixels - 1) * kAcrossTrackResolution + kHalfGap;

    // Allow 5% tolerance
    EXPECT_NEAR(dist_left, expected_edge_dist, expected_edge_dist * 0.05)
        << "Left edge distance incorrect at row " << ix;

    // Check rightmost pixel (should be symmetric)
    auto right_ecef = coordinates.lla_to_ecef<double>(
        {lon_swath(ix, lon_swath.cols() - 1),
         lat_swath(ix, lat_swath.cols() - 1), 0.0});

    double dist_right =
        std::sqrt(std::pow(right_ecef.get<0>() - nadir_ecef.get<0>(), 2) +
                  std::pow(right_ecef.get<1>() - nadir_ecef.get<1>(), 2) +
                  std::pow(right_ecef.get<2>() - nadir_ecef.get<2>(), 2));

    EXPECT_NEAR(dist_right, expected_edge_dist, expected_edge_dist * 0.05)
        << "Right edge distance incorrect at row " << ix;

    // Left and right should be symmetric
    EXPECT_NEAR(dist_left, dist_right, expected_edge_dist * 0.01)
        << "Asymmetry between left and right at row " << ix;
  }
}

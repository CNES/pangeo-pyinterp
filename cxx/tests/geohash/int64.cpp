// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/geohash/int64.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <ranges>
#include <tuple>

#include "pyinterp/geometry/geographic/box.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"

namespace pyinterp::geohash::int64 {

// Test fixture for geohash int64 tests
class GeoHashInt64Test : public ::testing::Test {
 protected:
  static constexpr double kEpsilon = 1e-10;
  static constexpr uint32_t kDefaultPrecision = 32;
  static constexpr uint32_t kMaxPrecision = 64;

  // Helper: Create a point
  static auto make_point(double lon, double lat)
      -> geometry::geographic::Point {
    return {lon, lat};
  }

  // Helper: Check if two points are approximately equal
  static auto points_equal(const geometry::geographic::Point& p1,
                           const geometry::geographic::Point& p2,
                           double eps = kEpsilon) -> bool {
    return std::abs(p1.lon() - p2.lon()) < eps &&
           std::abs(p1.lat() - p2.lat()) < eps;
  }
};

// ============================================================================
// Tests for error_with_precision()
// ============================================================================

TEST_F(GeoHashInt64Test, ErrorWithPrecisionBasic) {
  // Precision 2: 1 lat bit, 1 lon bit
  auto [lon_err, lat_err] = error_with_precision(2);
  EXPECT_DOUBLE_EQ(lon_err, 360.0 / 2.0);  // 180 degrees
  EXPECT_DOUBLE_EQ(lat_err, 180.0 / 2.0);  // 90 degrees
}

TEST_F(GeoHashInt64Test, ErrorWithPrecisionEven) {
  // Precision 32: 16 lat bits, 16 lon bits
  auto [lon_err, lat_err] = error_with_precision(32);
  auto expected_lon = 360.0 / (1ULL << 16);
  auto expected_lat = 180.0 / (1ULL << 16);
  EXPECT_DOUBLE_EQ(lon_err, expected_lon);
  EXPECT_DOUBLE_EQ(lat_err, expected_lat);
}

TEST_F(GeoHashInt64Test, ErrorWithPrecisionOdd) {
  // Precision 33: 16 lat bits, 17 lon bits
  auto [lon_err, lat_err] = error_with_precision(33);
  auto expected_lon = 360.0 / (1ULL << 17);
  auto expected_lat = 180.0 / (1ULL << 16);
  EXPECT_DOUBLE_EQ(lon_err, expected_lon);
  EXPECT_DOUBLE_EQ(lat_err, expected_lat);
}

TEST_F(GeoHashInt64Test, ErrorWithPrecisionZero) {
  // Edge case: precision 0
  auto [lon_err, lat_err] = error_with_precision(0);
  EXPECT_DOUBLE_EQ(lon_err, 360.0);
  EXPECT_DOUBLE_EQ(lat_err, 180.0);
}

TEST_F(GeoHashInt64Test, ErrorWithPrecisionMaximum) {
  // Maximum precision: 64 bits
  auto [lon_err, lat_err] = error_with_precision(64);
  auto expected_lon = 360.0 / (1ULL << 32);
  auto expected_lat = 180.0 / (1ULL << 32);
  EXPECT_DOUBLE_EQ(lon_err, expected_lon);
  EXPECT_DOUBLE_EQ(lat_err, expected_lat);
}

TEST_F(GeoHashInt64Test, ErrorWithPrecisionDecreases) {
  // Verify that error decreases with increased precision
  for (uint32_t p = 2; p < 60; p += 2) {
    auto [lon1, lat1] = error_with_precision(p);
    auto [lon2, lat2] = error_with_precision(p + 2);
    EXPECT_LT(lon2, lon1) << "Error should decrease with precision " << p;
    EXPECT_LT(lat2, lat1) << "Error should decrease with precision " << p;
  }
}

// ============================================================================
// Tests for encode(point, precision)
// ============================================================================

TEST_F(GeoHashInt64Test, EncodeOrigin) {
  // Encode (0, 0)
  auto hash = encode(make_point(0.0, 0.0), kDefaultPrecision);
  EXPECT_GT(hash, 0u) << "Hash of origin should be non-zero";
}

TEST_F(GeoHashInt64Test, EncodeLongitudeNormalization) {
  // Test longitude normalization
  auto hash1 =
      encode(make_point(190.0, 0.0), kDefaultPrecision);  // 190 -> -170
  auto hash2 = encode(make_point(-170.0, 0.0), kDefaultPrecision);
  EXPECT_EQ(hash1, hash2) << "Longitude normalization failed for 190 degrees";

  auto hash3 =
      encode(make_point(-190.0, 0.0), kDefaultPrecision);  // -190 -> 170
  auto hash4 = encode(make_point(170.0, 0.0), kDefaultPrecision);
  EXPECT_EQ(hash3, hash4) << "Longitude normalization failed for -190 degrees";
}

TEST_F(GeoHashInt64Test, EncodeKnownPoints) {
  // Test specific known points
  struct TestCase {
    double lon;
    double lat;
    uint32_t precision;
  };

  const std::array<TestCase, 5> test_cases = {
      TestCase{.lon = 0.0, .lat = 0.0, .precision = 32},  // Origin
      TestCase{
          .lon = 180.0, .lat = 90.0, .precision = 32},  // North-East extreme
      TestCase{
          .lon = -180.0, .lat = -90.0, .precision = 32},  // South-West extreme
      TestCase{.lon = -0.1, .lat = 51.5, .precision = 32},  // London
      TestCase{.lon = 2.3, .lat = 48.9, .precision = 32},   // Paris
  };

  for (const auto& test : test_cases) {
    auto hash = encode(make_point(test.lon, test.lat), test.precision);
    EXPECT_GE(hash, 0u) << "Hash should be valid for lon=" << test.lon
                        << ", lat=" << test.lat;

    // Verify the hash fits in the precision
    if (test.precision < 64) {
      EXPECT_EQ(hash >> test.precision, 0u)
          << "Hash should fit in precision bits";
    }
  }
}

TEST_F(GeoHashInt64Test, EncodeDifferentPrecisions) {
  auto point = make_point(10.0, 20.0);

  // Test various precisions
  std::vector<uint32_t> precisions = {8, 16, 24, 32, 40, 48, 56, 64};

  for (auto precision : precisions) {
    auto hash = encode(point, precision);

    // Verify bits are used correctly
    if (precision < 64) {
      EXPECT_EQ(hash >> precision, 0u)
          << "No bits should be set above precision " << precision;
    }
  }
}

TEST_F(GeoHashInt64Test, EncodeVectorized) {
  // Test vectorized encode function
  Eigen::VectorXd lon(5);
  Eigen::VectorXd lat(5);

  lon << 0.0, 10.0, -10.0, 100.0, -100.0;
  lat << 0.0, 20.0, -20.0, 45.0, -45.0;

  auto hashes = encode(lon, lat, kDefaultPrecision);

  ASSERT_EQ(hashes.size(), 5);

  for (Eigen::Index i = 0; i < hashes.size(); ++i) {
    auto single_hash = encode(make_point(lon[i], lat[i]), kDefaultPrecision);
    EXPECT_EQ(hashes[i], single_hash)
        << "Vectorized encode should match single encode for index " << i;
  }
}

TEST_F(GeoHashInt64Test, EncodeVectorizedNormalization) {
  // Test that longitude normalization works in vectorized version
  Eigen::VectorXd lon(3);
  Eigen::VectorXd lat(3);

  lon << 0.0, 360.0, 720.0;  // 360 and 720 should normalize to 0
  lat << 0.0, 0.0, 0.0;

  auto hashes = encode(lon, lat, kDefaultPrecision);

  // All should encode to the same hash (after normalization)
  EXPECT_EQ(hashes[0], hashes[1]) << "360 should normalize to 0";
  EXPECT_EQ(hashes[0], hashes[2]) << "720 should normalize to 0";
}

TEST_F(GeoHashInt64Test, EncodeVectorizedSizeMismatch) {
  // Test that mismatched sizes throw
  Eigen::VectorXd lon(5);
  Eigen::VectorXd lat(3);

  lon << 0.0, 10.0, -10.0, 100.0, -100.0;
  lat << 0.0, 20.0, -20.0;

  EXPECT_THROW(static_cast<void>(encode(lon, lat, kDefaultPrecision)),
               std::invalid_argument);
}

// ============================================================================
// Tests for bounding_box()
// ============================================================================

TEST_F(GeoHashInt64Test, BoundingBoxBasic) {
  auto point = make_point(10.0, 20.0);
  auto hash = encode(point, kDefaultPrecision);
  auto bbox = bounding_box(hash, kDefaultPrecision);

  auto min_pt = bbox.min_corner();
  auto max_pt = bbox.max_corner();

  // Check that point is within bounding box
  EXPECT_LE(min_pt.lon(), 10.0);
  EXPECT_LE(min_pt.lat(), 20.0);
  EXPECT_GE(max_pt.lon(), 10.0);
  EXPECT_GE(max_pt.lat(), 20.0);
}

TEST_F(GeoHashInt64Test, BoundingBoxSize) {
  auto hash = encode(make_point(0.0, 0.0), kDefaultPrecision);
  auto bbox = bounding_box(hash, kDefaultPrecision);

  auto [lon_err, lat_err] = error_with_precision(kDefaultPrecision);
  auto [delta_lon, delta_lat] = bbox.delta(false);

  // Bounding box size should match the error
  EXPECT_NEAR(delta_lon, lon_err, kEpsilon);
  EXPECT_NEAR(delta_lat, lat_err, kEpsilon);
}

TEST_F(GeoHashInt64Test, BoundingBoxIncreasesPrecision) {
  auto point = make_point(10.0, 20.0);

  // Higher precision should give smaller boxes
  for (uint32_t p1 = 16; p1 < 48; p1 += 8) {
    uint32_t p2 = p1 + 8;

    auto hash1 = encode(point, p1);
    auto hash2 = encode(point, p2);

    auto bbox1 = bounding_box(hash1, p1);
    auto bbox2 = bounding_box(hash2, p2);

    auto [delta_lon1, delta_lat1] = bbox1.delta(false);
    auto [delta_lon2, delta_lat2] = bbox2.delta(false);

    EXPECT_GT(delta_lon1, delta_lon2)
        << "Higher precision should give smaller lon delta";
    EXPECT_GT(delta_lat1, delta_lat2)
        << "Higher precision should give smaller lat delta";
  }
}

TEST_F(GeoHashInt64Test, BoundingBoxExtremes) {
  // Test extreme points
  struct TestCase {
    double lon;
    double lat;
  };

  const std::array<TestCase, 4> test_cases = {
      TestCase{.lon = 180.0, .lat = 90.0},    // NE corner
      TestCase{.lon = -180.0, .lat = -90.0},  // SW corner
      TestCase{.lon = 180.0, .lat = -90.0},   // SE corner
      TestCase{.lon = -180.0, .lat = 90.0},   // NW corner
  };

  for (const auto& test : test_cases) {
    auto hash = encode(make_point(test.lon, test.lat), kDefaultPrecision);
    auto bbox = bounding_box(hash, kDefaultPrecision);

    // Should not throw and should produce valid box
    auto min_pt = bbox.min_corner();
    auto max_pt = bbox.max_corner();

    EXPECT_LE(min_pt.lon(), max_pt.lon());
    EXPECT_LE(min_pt.lat(), max_pt.lat());
  }
}

// ============================================================================
// Tests for decode()
// ============================================================================

TEST_F(GeoHashInt64Test, DecodeBasic) {
  auto hash = encode(make_point(10.0, 20.0), kDefaultPrecision);
  auto decoded = decode(hash, kDefaultPrecision, false);

  // Decoded point should be close to original (centroid)
  EXPECT_NEAR(decoded.lon(), 10.0, 0.01);
  EXPECT_NEAR(decoded.lat(), 20.0, 0.01);
}

TEST_F(GeoHashInt64Test, DecodeRoundVsCenter) {
  auto hash = encode(make_point(10.0, 20.0), kDefaultPrecision);

  auto centroid = decode(hash, kDefaultPrecision, false);
  auto rounded = decode(hash, kDefaultPrecision, true);

  // Both should be valid points
  EXPECT_GE(centroid.lon(), -180.0);
  EXPECT_LE(centroid.lon(), 180.0);
  EXPECT_GE(centroid.lat(), -90.0);
  EXPECT_LE(centroid.lat(), 90.0);

  EXPECT_GE(rounded.lon(), -180.0);
  EXPECT_LE(rounded.lon(), 180.0);
  EXPECT_GE(rounded.lat(), -90.0);
  EXPECT_LE(rounded.lat(), 90.0);

  // Rounded and centroid may be different
  // (depending on the specific hash)
}

TEST_F(GeoHashInt64Test, DecodeVectorized) {
  // Create vector of hashes
  Vector<uint64_t> hashes(5);
  Eigen::VectorXd orig_lon(5);
  Eigen::VectorXd orig_lat(5);

  orig_lon << 0.0, 10.0, -10.0, 100.0, -100.0;
  orig_lat << 0.0, 20.0, -20.0, 45.0, -45.0;

  for (Eigen::Index i = 0; i < 5; ++i) {
    hashes[i] = encode(make_point(orig_lon[i], orig_lat[i]), kDefaultPrecision);
  }

  auto [decoded_lon, decoded_lat] = decode(hashes, kDefaultPrecision, false);

  ASSERT_EQ(decoded_lon.size(), 5);
  ASSERT_EQ(decoded_lat.size(), 5);

  for (Eigen::Index i = 0; i < 5; ++i) {
    EXPECT_NEAR(decoded_lon[i], orig_lon[i], 0.01) << "Mismatch at index " << i;
    EXPECT_NEAR(decoded_lat[i], orig_lat[i], 0.01) << "Mismatch at index " << i;
  }
}

TEST_F(GeoHashInt64Test, EncodeDecodeRoundTrip) {
  // Test round-trip encoding/decoding
  struct TestCase {
    double lon;
    double lat;
    uint32_t precision;
  };

  const std::array<TestCase, 6> test_cases = {
      TestCase{.lon = 0.0, .lat = 0.0, .precision = 32},
      TestCase{.lon = 10.0, .lat = 20.0, .precision = 32},
      TestCase{.lon = -10.0, .lat = -20.0, .precision = 32},
      TestCase{.lon = 179.9, .lat = 89.9, .precision = 40},
      TestCase{.lon = -179.9, .lat = -89.9, .precision = 40},
      TestCase{.lon = 50.123, .lat = 30.456, .precision = 48},
  };

  for (const auto& test : test_cases) {
    auto orig_point = make_point(test.lon, test.lat);
    auto hash = encode(orig_point, test.precision);
    auto decoded = decode(hash, test.precision, false);

    auto [lon_err, lat_err] = error_with_precision(test.precision);

    EXPECT_NEAR(decoded.lon(), test.lon, lon_err)
        << "Longitude round-trip failed for precision " << test.precision;
    EXPECT_NEAR(decoded.lat(), test.lat, lat_err)
        << "Latitude round-trip failed for precision " << test.precision;
  }
}

// ============================================================================
// Tests for neighbors()
// ============================================================================

TEST_F(GeoHashInt64Test, NeighborsBasic) {
  auto hash = encode(make_point(10.0, 20.0), kDefaultPrecision);
  auto neighbor_hashes = neighbors(hash, kDefaultPrecision);

  // Should return 8 neighbors
  ASSERT_EQ(neighbor_hashes.size(), 8);

  // All neighbors should be different from center
  for (Eigen::Index i = 0; i < 8; ++i) {
    EXPECT_NE(neighbor_hashes[i], hash)
        << "Neighbor " << i << " should differ from center";
  }

  // All neighbors should be different from each other
  for (Eigen::Index i = 0; i < 8; ++i) {
    for (Eigen::Index j = i + 1; j < 8; ++j) {
      EXPECT_NE(neighbor_hashes[i], neighbor_hashes[j])
          << "Neighbors " << i << " and " << j << " should differ";
    }
  }
}

TEST_F(GeoHashInt64Test, NeighborsPattern) {
  // Test the neighbor pattern: N, NE, E, SE, S, SW, W, NW
  auto center_point = make_point(10.0, 20.0);
  auto hash = encode(center_point, kDefaultPrecision);
  auto neighbor_hashes = neighbors(hash, kDefaultPrecision);

  // Decode neighbors and verify positions relative to center
  std::array<std::string, 8> directions = {"N", "NE", "E", "SE",
                                           "S", "SW", "W", "NW"};

  for (Eigen::Index i = 0; i < 8; ++i) {
    auto neighbor_point = decode(neighbor_hashes[i], kDefaultPrecision, false);
    auto n_lon = neighbor_point.lon();
    auto n_lat = neighbor_point.lat();

    // Just verify they decode to valid points
    EXPECT_GE(n_lon, -180.0) << "Neighbor " << directions[i] << " invalid lon";
    EXPECT_LE(n_lon, 180.0) << "Neighbor " << directions[i] << " invalid lon";
    EXPECT_GE(n_lat, -90.0) << "Neighbor " << directions[i] << " invalid lat";
    EXPECT_LE(n_lat, 90.0) << "Neighbor " << directions[i] << " invalid lat";
  }
}

TEST_F(GeoHashInt64Test, NeighborsSymmetry) {
  // Test that opposite neighbors are symmetric
  auto hash = encode(make_point(0.0, 0.0), kDefaultPrecision);
  auto neighbor_hashes = neighbors(hash, kDefaultPrecision);

  // N (0) and S (4) should be opposite in latitude
  auto n_point = decode(neighbor_hashes[0], kDefaultPrecision, false);
  auto s_point = decode(neighbor_hashes[4], kDefaultPrecision, false);

  EXPECT_NEAR(n_point.lon(), s_point.lon(), 0.01)
      << "N and S should have same longitude";

  // E (2) and W (6) should be opposite in longitude
  auto e_point = decode(neighbor_hashes[2], kDefaultPrecision, false);
  auto w_point = decode(neighbor_hashes[6], kDefaultPrecision, false);

  EXPECT_NEAR(e_point.lat(), w_point.lat(), 0.01)
      << "E and W should have same latitude";
}

TEST_F(GeoHashInt64Test, NeighborsNearPole) {
  // Test neighbors near pole (edge case)
  auto hash = encode(make_point(0.0, 85.0), kDefaultPrecision);
  auto neighbor_hashes = neighbors(hash, kDefaultPrecision);

  // Should still produce 8 neighbors without error
  EXPECT_EQ(neighbor_hashes.size(), 8);

  // All should be valid hashes
  for (Eigen::Index i = 0; i < 8; ++i) {
    EXPECT_GE(neighbor_hashes[i], 0u);
  }
}

// ============================================================================
// Tests for grid_properties()
// ============================================================================

TEST_F(GeoHashInt64Test, GridPropertiesBasic) {
  auto box =
      geometry::geographic::Box(make_point(0.0, 0.0), make_point(10.0, 10.0));
  auto [hash_sw, lon_step, lat_step] = grid_properties(box, kDefaultPrecision);

  // Should have valid hash and positive steps
  EXPECT_GE(hash_sw, 0u);
  EXPECT_GT(lon_step, 0u) << "Longitude step should be positive";
  EXPECT_GT(lat_step, 0u) << "Latitude step should be positive";
}

TEST_F(GeoHashInt64Test, GridPropertiesSinglePoint) {
  // Box that is a single point
  auto box =
      geometry::geographic::Box(make_point(10.0, 20.0), make_point(10.0, 20.0));
  auto [hash_sw, lon_step, lat_step] = grid_properties(box, kDefaultPrecision);

  // Should return 1x1 grid
  EXPECT_EQ(lon_step, 1u) << "Single point should have lon_step = 1";
  EXPECT_EQ(lat_step, 1u) << "Single point should have lat_step = 1";
}

TEST_F(GeoHashInt64Test, GridPropertiesIncreasesPrecision) {
  auto box =
      geometry::geographic::Box(make_point(0.0, 0.0), make_point(10.0, 10.0));

  // Higher precision should give more grid cells
  for (uint32_t p1 = 16; p1 < 32; p1 += 4) {
    uint32_t p2 = p1 + 4;

    auto [hash1, lon_step1, lat_step1] = grid_properties(box, p1);
    auto [hash2, lon_step2, lat_step2] = grid_properties(box, p2);

    EXPECT_GE(lon_step2, lon_step1)
        << "Higher precision should have more/equal lon steps";
    EXPECT_GE(lat_step2, lat_step1)
        << "Higher precision should have more/equal lat steps";
  }
}

TEST_F(GeoHashInt64Test, GridPropertiesLargeBox) {
  // Test with a large box
  auto box = geometry::geographic::Box(make_point(-100.0, -50.0),
                                       make_point(100.0, 50.0));
  auto [hash_sw, lon_step, lat_step] = grid_properties(box, 24);

  // Large box should have many cells
  EXPECT_GT(lon_step, 10u) << "Large box should have many lon cells";
  EXPECT_GT(lat_step, 10u) << "Large box should have many lat cells";
}

TEST_F(GeoHashInt64Test, GridPropertiesEdgeCases) {
  // Test box at 180 degree boundary
  auto box1 = geometry::geographic::Box(make_point(170.0, 0.0),
                                        make_point(180.0, 10.0));
  auto [hash1, lon_step1, lat_step1] = grid_properties(box1, kDefaultPrecision);

  EXPECT_GT(lon_step1, 0u);
  EXPECT_GT(lat_step1, 0u);

  // Test box at 90 degree boundary
  auto box2 =
      geometry::geographic::Box(make_point(0.0, 80.0), make_point(10.0, 90.0));
  auto [hash2, lon_step2, lat_step2] = grid_properties(box2, kDefaultPrecision);

  EXPECT_GT(lon_step2, 0u);
  EXPECT_GT(lat_step2, 0u);
}

// ============================================================================
// Tests for area()
// ============================================================================

TEST_F(GeoHashInt64Test, AreaBasic) {
  auto hash = encode(make_point(0.0, 0.0), kDefaultPrecision);
  auto computed_area = area(hash, kDefaultPrecision, std::nullopt);

  // Area should be positive
  EXPECT_GT(computed_area, 0.0);
}

TEST_F(GeoHashInt64Test, AreaWithSpheroid) {
  auto hash = encode(make_point(0.0, 0.0), kDefaultPrecision);

  auto wgs84 = geometry::geographic::Spheroid();  // Default WGS84

  auto area_with_wgs = area(hash, kDefaultPrecision, wgs84);

  // Area should be positive
  EXPECT_GT(area_with_wgs, 0.0);
}

TEST_F(GeoHashInt64Test, AreaDecreasesPrecision) {
  auto point = make_point(0.0, 0.0);

  // Higher precision should give smaller areas
  for (uint32_t p1 = 16; p1 < 40; p1 += 4) {
    uint32_t p2 = p1 + 4;

    auto hash1 = encode(point, p1);
    auto hash2 = encode(point, p2);

    auto area1 = area(hash1, p1, std::nullopt);
    auto area2 = area(hash2, p2, std::nullopt);

    EXPECT_GT(area1, area2)
        << "Higher precision should give smaller area at precision " << p1;
  }
}

TEST_F(GeoHashInt64Test, AreaNearPole) {
  // Areas near poles should be smaller (due to spherical geometry)
  auto hash_equator = encode(make_point(0.0, 0.0), kDefaultPrecision);
  auto hash_pole = encode(make_point(0.0, 85.0), kDefaultPrecision);

  auto area_equator = area(hash_equator, kDefaultPrecision, std::nullopt);
  auto area_pole = area(hash_pole, kDefaultPrecision, std::nullopt);

  EXPECT_GT(area_equator, area_pole)
      << "Area at equator should be larger than near pole";
}

// ============================================================================
// Tests for bounding_boxes()
// ============================================================================

TEST_F(GeoHashInt64Test, BoundingBoxesPointZero) {
  auto box = geometry::geographic::Box(make_point(-180, -90.0),
                                       make_point(-135.0, -45.0));
  auto boxes = bounding_boxes(box, 10);

  EXPECT_EQ(boxes.size(), 32);
}

TEST_F(GeoHashInt64Test, BoundingBoxesPolygonAntiMeridian) {
  // Create a polygon covering the same region as BoundingBoxesPointZero
  // This tests that safe_envelope() properly handles polygons at the
  // anti-meridian
  geometry::geographic::Polygon polygon;

  // Create outer ring for polygon from (-180, -90) to (-135, -45)
  polygon.outer().push_back(make_point(-180.0, -90.0));
  polygon.outer().push_back(make_point(-135.0, -90.0));
  polygon.outer().push_back(make_point(-135.0, -45.0));
  polygon.outer().push_back(make_point(-180.0, -45.0));
  polygon.outer().push_back(make_point(-180.0, -90.0));  // Close the ring

  auto boxes = bounding_boxes(polygon, 10);

  // Should return the same number of boxes as the equivalent Box test
  EXPECT_EQ(boxes.size(), 32);
}

TEST_F(GeoHashInt64Test, BoundingBoxesBasic) {
  auto box =
      geometry::geographic::Box(make_point(0.0, 0.0), make_point(10.0, 10.0));
  auto boxes = bounding_boxes(box, kDefaultPrecision);

  // Should return non-empty vector
  EXPECT_GT(boxes.size(), 0);
}

TEST_F(GeoHashInt64Test, BoundingBoxesCount) {
  auto box =
      geometry::geographic::Box(make_point(0.0, 0.0), make_point(10.0, 10.0));
  auto boxes = bounding_boxes(box, kDefaultPrecision);

  auto [hash_sw, lon_step, lat_step] = grid_properties(box, kDefaultPrecision);

  // Number of boxes should match grid properties
  EXPECT_EQ(boxes.size(), lon_step * lat_step)
      << "Number of boxes should match grid size";
}

TEST_F(GeoHashInt64Test, BoundingBoxesSingleCell) {
  // Small box that fits in one geohash cell
  auto [lon_err, lat_err] = error_with_precision(kDefaultPrecision);
  auto box = geometry::geographic::Box(make_point(0.0, 0.0),
                                       make_point(lon_err / 2, lat_err / 2));

  auto boxes = bounding_boxes(box, kDefaultPrecision);

  // Should have at least 1 box
  EXPECT_GE(boxes.size(), 1u);
}

TEST_F(GeoHashInt64Test, BoundingBoxesIncreasesPrecision) {
  auto box =
      geometry::geographic::Box(make_point(0.0, 0.0), make_point(10.0, 10.0));

  // Higher precision should give more boxes
  for (uint32_t p1 = 16; p1 < 32; p1 += 4) {
    uint32_t p2 = p1 + 4;

    auto boxes1 = bounding_boxes(box, p1);
    auto boxes2 = bounding_boxes(box, p2);

    EXPECT_LT(boxes1.size(), boxes2.size())
        << "Higher precision should produce more boxes";
  }
}

TEST_F(GeoHashInt64Test, BoundingBoxesUnique) {
  auto box =
      geometry::geographic::Box(make_point(0.0, 0.0), make_point(10.0, 10.0));
  auto boxes = bounding_boxes(box, kDefaultPrecision);

  // All hashes should be unique (convert to set and check size)
  std::set<uint64_t> unique_hashes(boxes.data(), boxes.data() + boxes.size());

  EXPECT_EQ(unique_hashes.size(), static_cast<size_t>(boxes.size()))
      << "All bounding box hashes should be unique";
}

TEST_F(GeoHashInt64Test, BoundingBoxesCoverage) {
  auto box =
      geometry::geographic::Box(make_point(0.0, 0.0), make_point(5.0, 5.0));
  auto boxes = bounding_boxes(box, 24);

  // Verify that all boxes are within or adjacent to the original box
  for (Eigen::Index i = 0; i < boxes.size(); ++i) {
    auto bbox = bounding_box(boxes[i], 24);
    auto center = bbox.centroid();

    // Center should be reasonably close to the original box
    auto lon = center.lon();
    auto lat = center.lat();

    // Allow some tolerance for edge boxes
    EXPECT_GE(lon, -1.0) << "Box " << i << " lon too far west";
    EXPECT_LE(lon, 6.0) << "Box " << i << " lon too far east";
    EXPECT_GE(lat, -1.0) << "Box " << i << " lat too far south";
    EXPECT_LE(lat, 6.0) << "Box " << i << " lat too far north";
  }
}

TEST_F(GeoHashInt64Test, BoundingBoxesLargeArea) {
  // Test with a large area
  auto box = geometry::geographic::Box(make_point(-50.0, -30.0),
                                       make_point(50.0, 30.0));
  auto boxes = bounding_boxes(box, 20);

  // Should produce many boxes
  EXPECT_GT(boxes.size(), 100u) << "Large box should produce many cells";

  // All should be valid
  for (const auto& hash : boxes) {
    EXPECT_GE(hash, 0u);
  }
}

// ============================================================================
// Tests for NeighborHashes type
// ============================================================================

TEST_F(GeoHashInt64Test, NeighborHashesType) {
  // Verify NeighborHashes is the correct type
  NeighborHashes nh{};
  EXPECT_EQ(nh.size(), 8);
  EXPECT_EQ(nh.rows(), 8);
  EXPECT_EQ(nh.cols(), 1);
}

// ============================================================================
// Edge case and stress tests
// ============================================================================

TEST_F(GeoHashInt64Test, EdgeCaseDateLine) {
  // Test near international date line
  auto point_west = make_point(179.9, 0.0);
  auto point_east = make_point(-179.9, 0.0);

  auto hash_west = encode(point_west, kDefaultPrecision);
  auto hash_east = encode(point_east, kDefaultPrecision);

  // Should be different hashes
  EXPECT_NE(hash_west, hash_east);

  // Both should decode close to originals
  auto decoded_west = decode(hash_west, kDefaultPrecision, false);
  auto decoded_east = decode(hash_east, kDefaultPrecision, false);

  EXPECT_NEAR(decoded_west.lon(), 179.9, 0.1);
  EXPECT_NEAR(decoded_east.lon(), -179.9, 0.1);
}

TEST_F(GeoHashInt64Test, EdgeCasePoles) {
  // Test exactly at poles
  auto north_pole = make_point(0.0, 90.0);
  auto south_pole = make_point(0.0, -90.0);

  auto hash_north = encode(north_pole, kDefaultPrecision);
  auto hash_south = encode(south_pole, kDefaultPrecision);

  EXPECT_NE(hash_north, hash_south);

  auto decoded_north = decode(hash_north, kDefaultPrecision, false);
  auto decoded_south = decode(hash_south, kDefaultPrecision, false);

  EXPECT_NEAR(decoded_north.lat(), 90.0, 0.1);
  EXPECT_NEAR(decoded_south.lat(), -90.0, 0.1);
}

TEST_F(GeoHashInt64Test, StressPrecision1) {
  // Test with very low precision
  auto point = make_point(10.0, 20.0);
  auto hash = encode(point, 1);

  EXPECT_GE(hash, 0u);
  EXPECT_LE(hash, 1u);  // Only 1 bit

  auto [lon_err, lat_err] = error_with_precision(1);

  // With precision 1: 0 lat bits, 1 lon bit -> lon_err = 360/2 = 180
  EXPECT_NEAR(lon_err, 180.0, kEpsilon);
  EXPECT_NEAR(lat_err, 180.0, kEpsilon);
}

TEST_F(GeoHashInt64Test, StressHighPrecision) {
  // Test with very high precision
  auto point = make_point(10.123456789, 20.987654321);

  for (uint32_t precision : {56, 60, 64}) {
    auto hash = encode(point, precision);
    auto decoded = decode(hash, precision, false);

    auto [lon_err, lat_err] = error_with_precision(precision);

    EXPECT_NEAR(decoded.lon(), 10.123456789, lon_err);
    EXPECT_NEAR(decoded.lat(), 20.987654321, lat_err);
  }
}

TEST_F(GeoHashInt64Test, StressRandomPoints) {
  // Test with many random points
  std::srand(42);  // Fixed seed for reproducibility

  for (int i = 0; i < 100; ++i) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    double lon = (std::rand() / static_cast<double>(RAND_MAX)) * 360.0 - 180.0;
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    double lat = (std::rand() / static_cast<double>(RAND_MAX)) * 180.0 - 90.0;

    auto point = make_point(lon, lat);
    auto hash = encode(point, kDefaultPrecision);
    auto decoded = decode(hash, kDefaultPrecision, false);

    EXPECT_NEAR(decoded.lon(), lon, 0.01) << "Failed for random point " << i;
    EXPECT_NEAR(decoded.lat(), lat, 0.01) << "Failed for random point " << i;
  }
}

// ============================================================================
// Constexpr tests
// ============================================================================

TEST(GeoHashInt64ConstexprTest, ErrorWithPrecisionConstexpr) {
  // Note: error_with_precision uses std::ldexp which is not constexpr
  // so we test it at runtime but verify it's correct
  auto [lon_err, lat_err] = error_with_precision(32);
  EXPECT_DOUBLE_EQ(lon_err, 360.0 / (1ULL << 16));
  EXPECT_DOUBLE_EQ(lat_err, 180.0 / (1ULL << 16));
}

// ============================================================================
// Performance-oriented tests
// ============================================================================

TEST_F(GeoHashInt64Test, PerformanceVectorizedEncode) {
  // Test vectorized performance with large dataset
  constexpr int64_t N = 10000;
  Eigen::VectorXd lon =
      (Eigen::VectorXd::Random(N).array() * 360.0 - 180.0).matrix();
  Eigen::VectorXd lat =
      (Eigen::VectorXd::Random(N).array() * 180.0 - 90.0).matrix();

  auto hashes = encode(lon, lat, kDefaultPrecision);

  EXPECT_EQ(hashes.size(), N);

  // Verify all hashes are valid
  for (int64_t i = 0; i < N; ++i) {
    EXPECT_GE(hashes[i], 0u);
  }
}

TEST_F(GeoHashInt64Test, PerformanceBoundingBoxes) {
  // Test bounding boxes performance with large area
  auto box = geometry::geographic::Box(make_point(-100.0, -50.0),
                                       make_point(100.0, 50.0));

  auto boxes = bounding_boxes(box, 20);

  // Should complete without error and produce many boxes
  EXPECT_GT(boxes.size(), 1000u);
}

}  // namespace pyinterp::geohash::int64

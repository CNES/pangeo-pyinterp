// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/geohash/string.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "pyinterp/geometry/geographic/box.hpp"
#include "pyinterp/geometry/geographic/multi_polygon.hpp"
#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/geographic/polygon.hpp"
#include "pyinterp/geometry/geographic/ring.hpp"

namespace pyinterp::geohash {

// Test fixture for geohash string tests
class GeoHashStringTest : public ::testing::Test {
 protected:
  static constexpr double kEpsilon = 1e-6;
  static constexpr double kEpsilonDecode = 1e-5;

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

  // Helper: Convert span to string
  static auto span_to_string(std::span<const char> span) -> std::string {
    return {span.begin(), span.end()};
  }

  // Helper: Compare encoded hash with expected string (array version)
  template <size_t N>
  static auto hash_equals(const std::array<char, N>& hash,
                          std::string_view expected) -> bool {
    return std::equal(hash.begin(), hash.end(), expected.begin(),
                      expected.end());
  }

  // Helper: Compare encoded hash with expected string (span version)
  static auto hash_equals(std::span<const char> hash,
                          std::span<const char> expected) -> bool {
    return std::ranges::equal(hash, expected);
  }

  // Helper: Compare array with array
  template <size_t N>
  static auto hash_equals(const std::array<char, N>& hash1,
                          const std::array<char, N>& hash2) -> bool {
    return std::equal(hash1.begin(), hash1.end(), hash2.begin(), hash2.end());
  }
};

// ============================================================================
// Tests for encode(point, buffer)
// ============================================================================

TEST_F(GeoHashStringTest, EncodePointBasic) {
  // Test encoding a simple point
  auto point = make_point(0.0, 0.0);
  std::array<char, 5> buffer{};
  encode(point, buffer);

  // "s0000" is the geohash for (0, 0) at precision 5
  EXPECT_TRUE(hash_equals(buffer, "s0000"));
}

TEST_F(GeoHashStringTest, EncodePointKnownLocations) {
  // Test some well-known locations
  std::array<char, 6> buffer{};

  // Eiffel Tower area (48.858, 2.294)
  encode(make_point(2.294, 48.858), buffer);
  std::string result(buffer.begin(), buffer.end());
  EXPECT_EQ(result.substr(0, 3), "u09");

  // Statue of Liberty area (-74.044, 40.689)
  encode(make_point(-74.044, 40.689), buffer);
  result = std::string(buffer.begin(), buffer.end());
  EXPECT_EQ(result.substr(0, 3), "dr5");
}

TEST_F(GeoHashStringTest, EncodePointDifferentPrecisions) {
  auto point = make_point(38.198505253, 0.497818518);

  // Precision 5
  std::array<char, 5> buffer5{};
  encode(point, buffer5);
  EXPECT_EQ(buffer5.size(), 5);
  EXPECT_EQ(span_to_string(buffer5), "sb54v");

  // Precision 8
  std::array<char, 8> buffer8{};
  encode(point, buffer8);
  EXPECT_EQ(buffer8.size(), 8);
  EXPECT_EQ(span_to_string(buffer8), "sb54v4xk");

  // Precision 12
  std::array<char, 12> buffer12{};
  encode(point, buffer12);
  EXPECT_EQ(buffer12.size(), 12);
  EXPECT_EQ(span_to_string(buffer12), "sb54v4xk18jg");
}

// ============================================================================
// Tests for encode(lon, lat, precision) - vector encoding
// ============================================================================

TEST_F(GeoHashStringTest, EncodeVectorsSinglePoint) {
  Eigen::VectorXd lon(1);
  Eigen::VectorXd lat(1);
  lon << 0.0;
  lat << 0.0;

  auto result = encode(lon, lat, 5);

  EXPECT_EQ(result.count, 1);
  EXPECT_EQ(result.precision, 5);

  // Decode and verify we get back (0, 0) approximately
  // At precision 5, geohash cells are ~5km wide, so use 0.05 degrees (~5.5km)
  auto [lon_out, lat_out] = decode(result, false);
  EXPECT_NEAR(lon_out(0), 0.0, 0.05);
  EXPECT_NEAR(lat_out(0), 0.0, 0.05);
}

TEST_F(GeoHashStringTest, EncodeVectorsMultiplePoints) {
  Eigen::VectorXd lon(3);
  Eigen::VectorXd lat(3);
  lon << 0.0, 2.294, -74.044;
  lat << 0.0, 48.858, 40.689;

  auto result = encode(lon, lat, 6);

  EXPECT_EQ(result.count, 3);
  EXPECT_EQ(result.precision, 6);

  // Check first few characters of each hash
  auto hash0 = span_to_string(result.get(0));
  auto hash1 = span_to_string(result.get(1));
  auto hash2 = span_to_string(result.get(2));

  EXPECT_EQ(hash0, "s00000");
  EXPECT_EQ(hash1, "u09tun");
  EXPECT_EQ(hash2, "dr5r7p");
}

TEST_F(GeoHashStringTest, EncodeVectorsLongitudeNormalization) {
  // Test that longitude [0, 360] is normalized to [-180, 180]
  Eigen::VectorXd lon1(1);
  Eigen::VectorXd lon2(1);
  Eigen::VectorXd lat(1);

  lon1 << -74.044;
  lon2 << 285.956;  // Same as -74.044 in [0, 360] range
  lat << 40.689;

  auto result1 = encode(lon1, lat, 8);
  auto result2 = encode(lon2, lat, 8);

  EXPECT_TRUE(hash_equals(result1.get(0), result2.get(0)));
}

// ============================================================================
// Tests for bounding_box(geohash, precision)
// ============================================================================

TEST_F(GeoHashStringTest, BoundingBoxZeroPoint) {
  std::array<char, 1> hash{'0'};
  auto bbox = bounding_box(hash);

  EXPECT_EQ(bbox.min_corner().lon(), -180.0);
  EXPECT_EQ(bbox.max_corner().lon(), -135.0);
  EXPECT_EQ(bbox.min_corner().lat(), -90.0);
  EXPECT_EQ(bbox.max_corner().lat(), -45.0);
}

TEST_F(GeoHashStringTest, BoundingBoxBasic) {
  std::array<char, 5> hash{'s', '0', '0', '0', '0'};
  auto bbox = bounding_box(hash);

  // s0000 should encompass (0, 0)
  EXPECT_LE(bbox.min_corner().lon(), 0.0);
  EXPECT_GE(bbox.max_corner().lon(), 0.0);
  EXPECT_LE(bbox.min_corner().lat(), 0.0);
  EXPECT_GE(bbox.max_corner().lat(), 0.0);
}

TEST_F(GeoHashStringTest, BoundingBoxReturnsValidBox) {
  std::array<char, 6> hash{'u', '0', '9', 't', 'u', 'n'};
  auto bbox = bounding_box(hash);

  // Check that min < max
  EXPECT_LT(bbox.min_corner().lon(), bbox.max_corner().lon());
  EXPECT_LT(bbox.min_corner().lat(), bbox.max_corner().lat());

  // Check that box is within valid ranges
  EXPECT_GE(bbox.min_corner().lon(), -180.0);
  EXPECT_LE(bbox.max_corner().lon(), 180.0);
  EXPECT_GE(bbox.min_corner().lat(), -90.0);
  EXPECT_LE(bbox.max_corner().lat(), 90.0);
}

TEST_F(GeoHashStringTest, BoundingBoxPrecisionOutput) {
  std::array<char, 8> hash{'u', '4', 'p', 'r', 'u', 'y', 'd', 'q'};
  uint32_t precision = 0;
  auto bbox = bounding_box(hash, &precision);

  EXPECT_EQ(precision, 8);
  EXPECT_NEAR(bbox.min_corner().lon(), 10.407142639160156, 1e-6);
  EXPECT_NEAR(bbox.min_corner().lat(), 57.64904022216797, 1e-6);
  EXPECT_NEAR(bbox.max_corner().lon(), 10.407485961914062, 1e-6);
  EXPECT_NEAR(bbox.max_corner().lat(), 57.64921188354492, 1e-6);
}

// ============================================================================
// Tests for decode(hash, round)
// ============================================================================

TEST_F(GeoHashStringTest, DecodeBasicCentroid) {
  std::array<char, 5> hash{'s', '0', '0', '0', '0'};
  auto point = decode(hash, false);

  // Should be close to origin
  EXPECT_NEAR(point.lon(), 0.02197265625, 1e-6);
  EXPECT_NEAR(point.lat(), 0.02197265625, 1e-6);
}

TEST_F(GeoHashStringTest, DecodeWithRounding) {
  std::array<char, 6> hash{'u', '0', '9', 't', 'u', 'n'};

  auto centroid = decode(hash, false);
  auto rounded = decode(hash, true);

  // Both should be close, but rounded should have cleaner coordinates
  EXPECT_NEAR(centroid.lon(), rounded.lon(), 1.0);
  EXPECT_NEAR(centroid.lat(), rounded.lat(), 1.0);
}

TEST_F(GeoHashStringTest, DecodeEncodedHashesVector) {
  Eigen::VectorXd lon_in(2);
  Eigen::VectorXd lat_in(2);
  lon_in << 2.294, -74.044;
  lat_in << 48.858, 40.689;

  auto encoded = encode(lon_in, lat_in, 8);
  auto [lon_out, lat_out] = decode(encoded, false);

  EXPECT_EQ(lon_out.size(), 2);
  EXPECT_EQ(lat_out.size(), 2);

  // Use a larger epsilon due to geohash discretization at precision 8
  EXPECT_NEAR(lon_out(0), lon_in(0), 0.01);
  EXPECT_NEAR(lat_out(0), lat_in(0), 0.01);
  EXPECT_NEAR(lon_out(1), lon_in(1), 0.01);
  EXPECT_NEAR(lat_out(1), lat_in(1), 0.01);
}

// ============================================================================
// Tests for neighbors(hash)
// ============================================================================

TEST_F(GeoHashStringTest, NeighborsReturnsEight) {
  std::array<char, 5> hash{'s', '0', '0', '0', '0'};
  auto neighbors_result = neighbors(hash);

  EXPECT_EQ(neighbors_result.count, 8);
  EXPECT_EQ(neighbors_result.precision, 5);
  std::vector<std::string> expected_neighbors = {
      "s0002", "s0003", "s0001", "kpbpc", "kpbpb", "7zzzz", "ebpbp", "ebpbr"};
  for (size_t i = 0; i < neighbors_result.count; ++i) {
    EXPECT_EQ(span_to_string(neighbors_result.get(i)), expected_neighbors[i]);
  }
}

TEST_F(GeoHashStringTest, NeighborsAreAdjacent) {
  std::array<char, 5> hash{'u', '0', '9', 't', 'u'};
  auto center_bbox = bounding_box(hash);
  auto center = center_bbox.centroid();

  auto neighbors_result = neighbors(hash);

  // At least one neighbor should be to the north (higher latitude)
  bool has_north = false;
  for (size_t i = 0; i < neighbors_result.count; ++i) {
    auto neighbor_bbox = bounding_box(neighbors_result.get(i));
    auto neighbor_center = neighbor_bbox.centroid();
    if (neighbor_center.lat() > center.lat() + 0.001) {
      has_north = true;
      break;
    }
  }
  EXPECT_TRUE(has_north);
}

// ============================================================================
// Tests for area(hash, spheroid)
// ============================================================================

TEST_F(GeoHashStringTest, AreaSingleHash) {
  std::array<char, 5> hash{'s', '0', '0', '0', '0'};
  auto area_value = area(hash, std::nullopt);

  // Area should be positive
  EXPECT_GT(area_value, 0.0);
}

TEST_F(GeoHashStringTest, AreaDecreasesWithPrecision) {
  std::array<char, 3> hash3{'s', '0', '0'};
  std::array<char, 5> hash5{'s', '0', '0', '0', '0'};
  std::array<char, 7> hash7{'s', '0', '0', '0', '0', '0', '0'};

  auto area3 = area(hash3, std::nullopt);
  auto area5 = area(hash5, std::nullopt);
  auto area7 = area(hash7, std::nullopt);

  // Higher precision = smaller area
  EXPECT_GT(area3, area5);
  EXPECT_GT(area5, area7);
}

TEST_F(GeoHashStringTest, AreaEncodedHashesVector) {
  Eigen::VectorXd lon(2);
  Eigen::VectorXd lat(2);
  lon << 0.0, 45.0;
  lat << 0.0, 45.0;

  auto encoded = encode(lon, lat, 5);
  auto areas = area(encoded, std::nullopt);

  EXPECT_EQ(areas.size(), 2);
  EXPECT_GT(areas(0), 0.0);
  EXPECT_GT(areas(1), 0.0);
}

// ============================================================================
// Tests for bounding_boxes(box, precision)
// ============================================================================

TEST_F(GeoHashStringTest, BoundingBoxesSmallRegion) {
  auto box = geometry::geographic::Box({-1.0, -1.0}, {1.0, 1.0});
  auto result = bounding_boxes(std::make_optional(box), 3);

  EXPECT_GT(result.count, 0);
  EXPECT_EQ(result.precision, 3);
  EXPECT_EQ(span_to_string(result.get(0)), "7zz");

  // All returned hashes should be within the box
  for (size_t i = 0; i < result.count; ++i) {
    auto hash_box = bounding_box(result.get(i));
    auto center = hash_box.centroid();
    // Center should be roughly within bounds (with some tolerance)
    EXPECT_GE(center.lon(), -2.0);
    EXPECT_LE(center.lon(), 2.0);
    EXPECT_GE(center.lat(), -2.0);
    EXPECT_LE(center.lat(), 2.0);
  }
}

TEST_F(GeoHashStringTest, BoundingBoxesNulloptGlobal) {
  // No box = global coverage
  auto result = bounding_boxes(std::nullopt, 1);

  EXPECT_EQ(result.count, 32);
  EXPECT_EQ(result.precision, 1);
  std::vector<std::string> expected_hashes = {
      "0", "1", "4", "5", "h", "j", "n", "p", "2", "3", "6",
      "7", "k", "m", "q", "r", "8", "9", "d", "e", "s", "t",
      "w", "x", "b", "c", "f", "g", "u", "v", "y", "z"};
  for (size_t i = 0; i < expected_hashes.size(); ++i) {
    EXPECT_EQ(span_to_string(result.get(i)), expected_hashes[i]);
  }
}

TEST_F(GeoHashStringTest, BoundingBoxesIncreasingPrecision) {
  auto box = geometry::geographic::Box({-10.0, -10.0}, {10.0, 10.0});

  auto result2 = bounding_boxes(std::make_optional(box), 2);
  auto result3 = bounding_boxes(std::make_optional(box), 3);

  // Higher precision = more boxes
  EXPECT_GT(result3.count, result2.count);
}

// ============================================================================
// Tests for bounding_boxes(polygon, precision, num_threads)
// ============================================================================

TEST_F(GeoHashStringTest, BoundingBoxesPolygonSimple) {
  // Create a simple square polygon (larger to ensure we get some geohashes)
  geometry::geographic::Ring ring;
  ring.push_back({-5.0, -5.0});
  ring.push_back({5.0, -5.0});
  ring.push_back({5.0, 5.0});
  ring.push_back({-5.0, 5.0});
  ring.push_back({-5.0, -5.0});

  geometry::geographic::Polygon polygon(ring);

  // First at precision 1 - Should return one geohash 7
  auto result = bounding_boxes(polygon, 1, 1);
  EXPECT_EQ(result.count, 1);
  EXPECT_EQ(result.precision, 1);
  EXPECT_EQ(span_to_string(result.get(0)), "7");

  // Second at precision 2 - Should return one geohash 7z
  result = bounding_boxes(polygon, 2, 1);
  EXPECT_EQ(result.count, 1);
  EXPECT_EQ(result.precision, 2);
  EXPECT_EQ(span_to_string(result.get(0)), "7z");

  // Third at precision 3 - Should return multiple geohashes
  result = bounding_boxes(polygon, 3, 1);
  std::vector<std::string> expected_hashes = {
      "7zh", "7zj", "7zn", "7zp", "kp0", "kp1", "kp4", "7zk", "7zm", "7zq",
      "7zr", "kp2", "kp3", "kp6", "7zs", "7zt", "7zw", "7zx", "kp8", "kp9",
      "kpd", "7zu", "7zv", "7zy", "7zz", "kpb", "kpc", "kpf", "ebh", "ebj",
      "ebn", "ebp", "s00", "s01", "s04", "ebk", "ebm", "ebq", "ebr", "s02",
      "s03", "s06", "ebs", "ebt", "ebw", "ebx", "s08", "s09", "s0d"};
  EXPECT_EQ(result.count, expected_hashes.size());
  for (size_t i = 0; i < result.count; ++i) {
    EXPECT_EQ(span_to_string(result.get(i)), expected_hashes[i]);
  }
  EXPECT_EQ(result.precision, 3);
}

TEST_F(GeoHashStringTest, BoundingBoxesPolygonWithThreads) {
  geometry::geographic::Ring ring;
  ring.push_back({-5.0, -5.0});
  ring.push_back({5.0, -5.0});
  ring.push_back({5.0, 5.0});
  ring.push_back({-5.0, 5.0});
  ring.push_back({-5.0, -5.0});

  geometry::geographic::Polygon polygon(ring);

  auto result_single = bounding_boxes(polygon, 4, 1);
  auto result_multi = bounding_boxes(polygon, 4, 2);

  // Results should be the same regardless of thread count
  EXPECT_EQ(result_single.count, result_multi.count);
  EXPECT_EQ(result_single.precision, result_multi.precision);
  for (size_t i = 0; i < result_single.count; ++i) {
    EXPECT_TRUE(hash_equals(result_single.get(i), result_multi.get(i)));
  }
}

// ============================================================================
// Tests for bounding_boxes(multipolygon, precision, num_threads)
// ============================================================================

TEST_F(GeoHashStringTest, BoundingBoxesMultiPolygon) {
  // Create two separate polygons (larger to ensure we get results)
  geometry::geographic::Ring ring1;
  ring1.push_back({-10.0, -10.0});
  ring1.push_back({-5.0, -10.0});
  ring1.push_back({-5.0, -5.0});
  ring1.push_back({-10.0, -5.0});
  ring1.push_back({-10.0, -10.0});

  geometry::geographic::Ring ring2;
  ring2.push_back({5.0, 5.0});
  ring2.push_back({10.0, 5.0});
  ring2.push_back({10.0, 10.0});
  ring2.push_back({5.0, 10.0});
  ring2.push_back({5.0, 5.0});

  geometry::geographic::MultiPolygon multipolygon;
  multipolygon.push_back(geometry::geographic::Polygon(ring1));
  multipolygon.push_back(geometry::geographic::Polygon(ring2));

  auto result = bounding_boxes(multipolygon, 2, 0);  // Lower precision

  EXPECT_GT(result.count, 0);
  EXPECT_EQ(result.precision, 2);
}

// ============================================================================
// Tests for where(hash, rows, cols)
// ============================================================================

TEST_F(GeoHashStringTest, WhereUniformGrid) {
  // Create a 3x3 grid of the same geohash
  EncodedHashes hashes{
      .buffer = std::vector<char>(9 * 5, 's'),
      .precision = 5,
      .count = 9,
  };
  // Fill all with "s0000"
  for (size_t i = 0; i < 9; ++i) {
    auto span = hashes.get(i);
    span[0] = 's';
    span[1] = '0';
    span[2] = '0';
    span[3] = '0';
    span[4] = '0';
  }

  auto result = where(hashes, 3, 3);

  EXPECT_EQ(result.size(), 1);  // Only one unique hash
  auto& bounds = result["s0000"];
  auto [row_min, row_max] = std::get<0>(bounds);
  auto [col_min, col_max] = std::get<1>(bounds);

  EXPECT_EQ(row_min, 0);
  EXPECT_EQ(row_max, 2);
  EXPECT_EQ(col_min, 0);
  EXPECT_EQ(col_max, 2);
}

TEST_F(GeoHashStringTest, WhereMultipleRegions) {
  // Create a 2x2 grid with two different hashes
  EncodedHashes hashes{
      .buffer = std::vector<char>(4 * 3),
      .precision = 3,
      .count = 4,
  };

  // [0,0] and [0,1]: "s00"
  std::copy_n("s00", 3, hashes.get(0).begin());
  std::copy_n("s00", 3, hashes.get(1).begin());

  // [1,0] and [1,1]: "s01"
  std::copy_n("s01", 3, hashes.get(2).begin());
  std::copy_n("s01", 3, hashes.get(3).begin());

  auto result = where(hashes, 2, 2);

  EXPECT_EQ(result.size(), 2);
  EXPECT_TRUE(result.contains("s00"));
  EXPECT_TRUE(result.contains("s01"));
}

// ============================================================================
// Tests for transform(hash, precision)
// ============================================================================

TEST_F(GeoHashStringTest, TransformSamePrecision) {
  Eigen::VectorXd lon(2);
  Eigen::VectorXd lat(2);
  lon << 0.0, 45.0;
  lat << 0.0, 45.0;

  auto encoded = encode(lon, lat, 5);
  auto result = transform(encoded, 5);

  EXPECT_EQ(result.count, encoded.count);
  EXPECT_EQ(result.precision, 5);

  // Should be identical
  for (size_t i = 0; i < result.count; ++i) {
    EXPECT_TRUE(hash_equals(result.get(i), encoded.get(i)));
  }
}

TEST_F(GeoHashStringTest, TransformZoomOut) {
  Eigen::VectorXd lon(1);
  Eigen::VectorXd lat(1);
  lon << 0.0;
  lat << 0.0;

  auto encoded = encode(lon, lat, 6);
  auto result = transform(encoded, 3);

  EXPECT_EQ(result.precision, 3);
  EXPECT_LE(result.count, 1);
  EXPECT_EQ(span_to_string(result.get(0)), "s00");
}

TEST_F(GeoHashStringTest, TransformZoomIn) {
  Eigen::VectorXd lon(1);
  Eigen::VectorXd lat(1);
  lon << 0.0;
  lat << 0.0;

  auto encoded = encode(lon, lat, 3);
  auto result = transform(encoded, 6);

  EXPECT_EQ(result.precision, 6);
  EXPECT_GE(result.count, 32768);  // 32^3 = 32768
  EXPECT_EQ(span_to_string(result.get(0)), "s00000");
  EXPECT_EQ(span_to_string(result.get(1)), "s00002");
  EXPECT_EQ(span_to_string(result.get(2)), "s00008");
  EXPECT_EQ(span_to_string(result.get(32765)), "s00zzr");
  EXPECT_EQ(span_to_string(result.get(32766)), "s00zzx");
  EXPECT_EQ(span_to_string(result.get(32767)), "s00zzz");
}

TEST_F(GeoHashStringTest, TransformZoomInThenOut) {
  Eigen::VectorXd lon(1);
  Eigen::VectorXd lat(1);
  lon << 2.5;
  lat << 3.5;

  auto original = encode(lon, lat, 4);
  auto zoomed_in = transform(original, 7);
  auto zoomed_out = transform(zoomed_in, 4);

  EXPECT_EQ(zoomed_out.precision, 4);

  // After zoom in/out, should cover at least the original area
  // The decoded point should be close
  auto [lon_orig, lat_orig] = decode(original, false);
  auto [lon_final, lat_final] = decode(zoomed_out, false);

  EXPECT_NEAR(lon_orig(0), lon_final(0), 1.0);
  EXPECT_NEAR(lat_orig(0), lat_final(0), 1.0);
}

// ============================================================================
// Tests for EncodedHashes iterator functionality
// ============================================================================

TEST_F(GeoHashStringTest, EncodedHashesIteration) {
  Eigen::VectorXd lon(3);
  Eigen::VectorXd lat(3);
  lon << 0.0, 45.0, -45.0;
  lat << 0.0, 45.0, -45.0;

  auto encoded = encode(lon, lat, 5);

  size_t count = 0;
  for (auto hash_span : encoded) {
    EXPECT_EQ(hash_span.size(), 5);
    ++count;
  }

  EXPECT_EQ(count, 3);
}

TEST_F(GeoHashStringTest, EncodedHashesRandomAccess) {
  Eigen::VectorXd lon(5);
  Eigen::VectorXd lat(5);
  lon << 0.0, 1.0, 2.0, 3.0, 4.0;
  lat << 0.0, 1.0, 2.0, 3.0, 4.0;

  auto encoded = encode(lon, lat, 6);

  auto it = encoded.begin();
  EXPECT_EQ((it + 2) - it, 2);
  EXPECT_EQ(it[2].size(), 6);

  auto it2 = it + 3;
  EXPECT_EQ(it2 - it, 3);
}

// ============================================================================
// Edge cases and error conditions
// ============================================================================

TEST_F(GeoHashStringTest, EncodeExtremeLatitudes) {
  std::array<char, 5> buffer{};

  // North pole
  encode(make_point(0.0, 89.9), buffer);
  auto decoded_north = decode(buffer, false);
  EXPECT_NEAR(decoded_north.lat(), 89.9, 1.0);

  // South pole
  encode(make_point(0.0, -89.9), buffer);
  auto decoded_south = decode(buffer, false);
  EXPECT_NEAR(decoded_south.lat(), -89.9, 1.0);
}

TEST_F(GeoHashStringTest, EncodeInternationalDateLine) {
  std::array<char, 6> buffer1{};
  std::array<char, 6> buffer2{};

  // Just east and west of date line
  encode(make_point(179.9, 0.0), buffer1);
  encode(make_point(-179.9, 0.0), buffer2);

  // Should be different hashes
  EXPECT_FALSE(hash_equals(buffer1, buffer2));
}

TEST_F(GeoHashStringTest, EmptyVectorEncoding) {
  Eigen::VectorXd lon(0);
  Eigen::VectorXd lat(0);

  auto result = encode(lon, lat, 5);

  EXPECT_EQ(result.count, 0);
  EXPECT_EQ(result.precision, 5);
}

}  // namespace pyinterp::geohash

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <ranges>
#include <tuple>

#include "pyinterp/broadcast.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/geographic/algorithms/area.hpp"
#include "pyinterp/geometry/geographic/box.hpp"
#include "pyinterp/geometry/geographic/multi_polygon.hpp"
#include "pyinterp/geometry/geographic/polygon.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"
#include "pyinterp/math.hpp"

namespace pyinterp::geohash::int64 {

/// @brief Fixed-size vector of 8 neighboring geohashes
using NeighborHashes = Eigen::Matrix<uint64_t, 8, 1>;

/// @brief Calculate the maximum latitude and longitude error for a given
/// precision
/// @param[in] precision Geohash precision (number of bits)
/// @return A tuple containing the longitude and latitude error in degrees
[[nodiscard]] constexpr auto error_with_precision(
    const uint32_t precision) noexcept -> std::tuple<double, double> {
  auto lat_bits = static_cast<int32_t>(precision >> 1U);
  auto lng_bits = static_cast<int32_t>(precision - lat_bits);

  return {360 * math::power2(-lng_bits), 180 * math::power2(-lat_bits)};
}

/// @brief Encode a geographic point into an integer geohash
/// @param[in] point Geodetic point (longitude, latitude)
/// @param[in] precision Geohash precision (number of bits)
/// @return Encoded integer geohash
[[nodiscard]] auto encode(const geometry::geographic::Point &point,
                          uint32_t precision) -> uint64_t;

/// @brief Encode points into geohash with the given precision
/// @param[in] lon Vector of longitudes
/// @param[in] lat Vector of latitudes
/// @param[in] precision Geohash precision (number of bits)
/// @return Vector of encoded geohashes
[[nodiscard]] inline auto encode(const Eigen::Ref<const Eigen::VectorXd> &lon,
                                 const Eigen::Ref<const Eigen::VectorXd> &lat,
                                 uint32_t precision) -> Vector<uint64_t> {
  broadcast::check_eigen_shape("lon", lon, "lat", lat);
  auto size = lon.size();
  auto result = Vector<uint64_t>(size);
  for (Eigen::Index ix = 0; ix < size; ++ix) {
    result(ix) = encode({lon[ix], lat[ix]}, precision);
  }
  return result;
}

/// @brief Returns the region encoded by the integer geohash with the specified
/// precision.
/// @param[in] hash Integer geohash
/// @param[in] precision Geohash precision (number of bits)
/// @return Bounding box representing the region encoded by the geohash
[[nodiscard]] auto bounding_box(uint64_t hash, uint32_t precision) noexcept
    -> geometry::geographic::Box;

/// @brief Decode the geohash into a point (centroid or rounded corner)
/// @param[in] hash Integer geohash
/// @param[in] precision Geohash precision (number of bits)
/// @param[in] round If true, returns the rounded corner of the bounding box;
/// otherwise, returns the centroid.
/// @return Decoded point
[[nodiscard]] inline auto decode(const uint64_t hash, const uint32_t precision,
                                 const bool round)
    -> geometry::geographic::Point {
  auto bbox = bounding_box(hash, precision);
  return round ? bbox.round() : bbox.centroid();
}

/// @brief Decode geohashes into points
/// @param[in] hash Vector of integer geohashes
/// @param[in] precision Geohash precision (number of bits)
/// @param[in] center If true, returns the centroid of the bounding box;
/// otherwise, returns the rounded corner.
/// @return A tuple containing vectors of longitudes and latitudes
[[nodiscard]] inline auto decode(const Eigen::Ref<const Vector<uint64_t>> &hash,
                                 const uint32_t precision, const bool center)
    -> std::tuple<Eigen::VectorXd, Eigen::VectorXd> {
  auto lon = Eigen::VectorXd(hash.size());
  auto lat = Eigen::VectorXd(hash.size());

  for (auto [hash_item, lon_item, lat_item] : std::views::zip(hash, lon, lat)) {
    auto point = decode(hash_item, precision, center);
    lon_item = point.lon();
    lat_item = point.lat();
  }
  return {lon, lat};
}

/// @brief Returns all neighboring geohashes around a given geohash.
/// @code
/// 7 0 1
/// 6 x 2
/// 5 4 3
/// @endcode
/// @param[in] hash Integer geohash
/// @param[in] precision Geohash precision (number of bits)
/// @return Matrix of 8 neighboring geohashes
[[nodiscard]] auto neighbors(uint64_t hash, uint32_t precision)
    -> NeighborHashes;

/// @brief Returns the property of the grid covering the given box
/// @param[in] box Geodetic box to cover (longitudes must be in [-180, 180])
/// @param[in] precision Geohash precision (number of bits)
/// @return A tuple containing the geohash of the minimum corner point, the
/// number of boxes in longitudes and latitudes
[[nodiscard]] auto grid_properties(const geometry::geographic::Box &box,
                                   uint32_t precision)
    -> std::tuple<uint64_t, size_t, size_t>;

/// @brief Returns the area covered by the GeoHash
/// @param[in] hash Integer geohash
/// @param[in] precision Geohash precision (number of bits)
/// @param[in] wgs Optional spheroid for area calculation
/// @return Area in square meters
[[nodiscard]] inline auto area(
    uint64_t hash, uint32_t precision,
    const std::optional<geometry::geographic::Spheroid> &wgs) -> double {
  return geometry::geographic::area<
      geometry::geographic::Box,
      geometry::geographic::StrategyMethod::kVincenty>(
      bounding_box(hash, precision), wgs);
}

/// @brief Returns all the GeoHash codes within the box.
/// @param[in] box Geodetic box to cover
/// @param[in] precision Geohash precision (number of bits)
/// @param[in] num_threads Number of threads to use for parallel computation
/// @return Vector of integer geohashes covering the box
[[nodiscard]] auto bounding_boxes(const geometry::geographic::Box &box,
                                  uint32_t precision, size_t num_threads = 1)
    -> Vector<uint64_t>;

/// @brief Returns all the GeoHash codes within the polygon.
/// @param[in] polygon Geodetic polygon to cover
/// @param[in] precision Geohash precision (number of bits)
/// @param[in] num_threads Number of threads to use for parallel computation
/// @return Vector of integer geohashes covering the polygon
[[nodiscard]] auto bounding_boxes(const geometry::geographic::Polygon &polygon,
                                  uint32_t precision, size_t num_threads = 1)
    -> Vector<uint64_t>;

/// @brief Returns all the GeoHash codes within the multipolygon.
/// @param[in] multipolygon Geodetic multipolygon to cover
/// @param[in] precision Geohash precision (number of bits)
/// @param[in] num_threads Number of threads to use for parallel computation
/// @return Vector of integer geohashes covering the multipolygon
[[nodiscard]] auto bounding_boxes(
    const geometry::geographic::MultiPolygon &multipolygon, uint32_t precision,
    size_t num_threads = 1) -> Vector<uint64_t>;

}  // namespace pyinterp::geohash::int64

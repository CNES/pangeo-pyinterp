// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/eigen.h>

#include <Eigen/Core>
#include <list>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/math.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/geodetic/box.hpp"
#include "pyinterp/geodetic/point.hpp"
#include "pyinterp/geodetic/polygon.hpp"

namespace pyinterp::geohash::int64 {

/// Format bytes as text
[[nodiscard]] auto format_bytes(size_t bytes) -> std::string;

/// Returns the precision in longitude/latitude and degrees for the given
/// precision
[[nodiscard]] inline auto error_with_precision(const uint32_t precision)
    -> std::tuple<double, double> {
  auto lat_bits = static_cast<int32_t>(precision >> 1U);
  auto lng_bits = static_cast<int32_t>(precision - lat_bits);

  return std::make_tuple(360 * detail::math::power2(-lng_bits),
                         180 * detail::math::power2(-lat_bits));
}

// Encode a point into geohash with the given precision
[[nodiscard]] auto encode(const geodetic::Point &point, uint32_t precision)
    -> uint64_t;

// Encode points into geohash with the given precision
[[nodiscard]] inline auto encode(const Eigen::Ref<const Eigen::VectorXd> &lon,
                                 const Eigen::Ref<const Eigen::VectorXd> &lat,
                                 uint32_t precision) -> Vector<uint64_t> {
  detail::check_eigen_shape("lon", lon, "lat", lat);
  auto size = lon.size();
  auto result = Vector<uint64_t>(size);
  for (Eigen::Index ix = 0; ix < size; ++ix) {
    result(ix) =
        encode({detail::math::normalize_angle<double>(lon[ix], -180.0, 360.0),
                lat[ix]},
               precision);
  }
  return result;
}

// Returns the region encoded by the integer geohash with the specified
// precision.
[[nodiscard]] auto bounding_box(uint64_t hash, uint32_t precision)
    -> geodetic::Box;

// Decode a hash into a geographic point with the given precision.
// If round is true, the coordinates of the points will be rounded to the
// accuracy defined by the GeoHash.
[[nodiscard]] inline auto decode(const uint64_t hash, const uint32_t precision,
                                 const bool round) -> geodetic::Point {
  auto bbox = bounding_box(hash, precision);
  return round ? bbox.round() : bbox.centroid();
}

// Decode hashes into a geographic points with the given bit depth.
// If round is true, the coordinates of the points will be rounded to the
// accuracy defined by the GeoHash.
[[nodiscard]] inline auto decode(const Eigen::Ref<const Vector<uint64_t>> &hash,
                                 const uint32_t precision, const bool center)
    -> std::tuple<Eigen::VectorXd, Eigen::VectorXd> {
  auto lon = Eigen::VectorXd(hash.size());
  auto lat = Eigen::VectorXd(hash.size());

  for (Eigen::Index ix = 0; ix < hash.size(); ++ix) {
    auto point = decode(hash(ix), precision, center);
    lon[ix] = point.lon();
    lat[ix] = point.lat();
  }
  return std::make_tuple(lon, lat);
}

// Returns all neighbors hash clockwise from north around northwest at the given
// precision.
// 7 0 1
// 6 x 2
// 5 4 3
[[nodiscard]] auto neighbors(uint64_t hash, uint32_t precision)
    -> Eigen::Matrix<uint64_t, 8, 1>;

// Returns the property of the grid covering the given box: geohash of the
// minimum corner point, number of boxes in longitudes and latitudes.
[[nodiscard]] auto grid_properties(const geodetic::Box &box, uint32_t precision)
    -> std::tuple<uint64_t, size_t, size_t>;

// Returns the area covered by the GeoHash
[[nodiscard]] inline auto area(uint64_t hash, uint32_t precision,
                               const std::optional<geodetic::Spheroid> &wgs)
    -> double {
  return bounding_box(hash, precision).area(wgs);
}

// Returns all the GeoHash codes within the box.
[[nodiscard]] auto bounding_boxes(const geodetic::Box &box, uint32_t precision)
    -> Vector<uint64_t>;

}  // namespace pyinterp::geohash::int64

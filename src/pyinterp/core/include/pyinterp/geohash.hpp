// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "pyinterp/geodetic/box.hpp"
#include "pyinterp/geodetic/spheroid.hpp"
#include "pyinterp/geohash/base32.hpp"
#include "pyinterp/geohash/int64.hpp"
#include "pyinterp/geohash/string.hpp"

namespace pyinterp::geohash {

/// Geohashing is a geocoding method used to encode geographic coordinates
/// (latitude and longitude) into a short string of digits and letters
/// delineating an area on a map, which is called a cell, with varying
/// resolutions. The more characters in the string, the more precise the
/// location. The table below gives the correspondence between the number of
/// characters, the size of the boxes of the grid at the equator and the total
/// number of boxes.
///
///             =========  ===============  ==========
///             precision  lng/lat (km)     samples
///             =========  ===============  ==========
///             1          4950/4950        32
///             2          618.75/1237.50   1024
///             3          154.69/154.69    32768
///             4          19.34/38.67      1048576
///             5          4.83/4.83        33554432
///             6          0.60/1.21        1073741824
///             =========  ===============  ==========
///
/// Geohashes use Base-32 alphabet encoding (characters can be 0 to 9 and A to
/// Z, excl A, I, L and O).
///
/// The geohash is a compact way of representing a location, and is useful for
/// storing a location in a database, or for indexing a location in a
/// database.
class GeoHash {
 public:
  /// GeoHash from longitude, latitude with number of characters.
  ///
  /// @param[in] longitude Longitude of the point.
  /// @param[in] latitude Latitude of the point.
  /// @param[in] precision Number of characters in the geohash.
  GeoHash(double longitude, double latitude, uint32_t precision)
      : code_(precision, '\0') {
    if (precision > 12) {
      throw std::invalid_argument("GeoHash precision must be <= 12");
    }
    string::encode(
        {detail::math::normalize_angle(longitude, -180.0, 360.0), latitude},
        code_.data(), precision);
  }

  /// GeoHash from its string representation.
  ///
  /// @param[in] geohash String representation of the geohash.
  /// @param[in] round If true, the coordinates of the point will be rounded to
  /// the accuracy defined by the GeoHash.
  /// @throw std::invalid_argument if the geohash is not valid.
  static auto from_string(const std::string &code, bool round) -> GeoHash {
    auto precision = static_cast<uint32_t>(code.size());
    if (precision > 12) {
      throw std::invalid_argument("GeoHash precision must be <= 12");
    }
    if (!Base32().validate(code.data(), precision)) {
      throw std::invalid_argument("GeoHash is not valid");
    }
    auto result = GeoHash(precision);
    string::encode(string::decode(code.data(), precision, round),
                   result.code_.data(), static_cast<uint32_t>(precision));
    return result;
  }

  /// Returns the bounding box of the geohash.
  [[nodiscard]] inline auto bounding_box() const -> geodetic::Box {
    return string::bounding_box(code_.data(), precision());
  }

  /// Returns the center point of this.
  [[nodiscard]] inline auto center() const -> geodetic::Point {
    return bounding_box().centroid();
  }

  /// Returns the geohash code.
  [[nodiscard]] inline auto string_value() const -> std::string {
    return code_;
  }

  /// Returns the precision of the geohash.
  [[nodiscard]] inline auto precision() const -> uint32_t {
    return static_cast<uint32_t>(code_.length());
  }

  /// Returns the number of bits used to represent the geohash.
  [[nodiscard]] inline auto number_of_bits() const -> uint32_t {
    return precision() * 5;
  }

  /// Returns the value of the integer64 stored in the geohash.
  [[nodiscard]] auto integer_value(bool round) const -> uint64_t {
    return int64::encode(string::decode(code_.data(), precision(), round),
                         number_of_bits());
  }

  /// Returns the eight neighbors of this.
  ///
  /// @return An array of GeoHash in the order N, NE, E, SE, S, SW, W, NW.
  [[nodiscard]] auto neighbors() const -> std::vector<GeoHash> {
    auto neighbors = int64::neighbors(integer_value(false), number_of_bits());
    auto result = std::vector<GeoHash>();
    result.reserve(8);
    for (auto ix = 0; ix < 8; ++ix) {
      auto code = std::string(precision(), '\0');
      Base32::encode(neighbors[ix], code.data(), code.size());
      result.emplace_back(GeoHash::from_string(code, false));
    }
    return result;
  }

  /// Returns the area covered by this.
  ///
  /// @return The area of the geohash in square meters.
  [[nodiscard]] inline auto area(
      const std::optional<geodetic::Spheroid> &wgs) const -> double {
    return string::area(code_.data(), precision(), wgs);
  }

  /// Gets the property of the grid covering the given box.
  ///
  /// @return A tuple of three elements containing: The GeoHash of the minimum
  /// corner point, the number of cells in longitudes and latitudes.
  [[nodiscard]] static auto grid_properties(const geodetic::Box &box,
                                            uint32_t precision)
      -> std::tuple<GeoHash, size_t, size_t> {
    auto [code, lng_boxes, lat_boxes] =
        int64::grid_properties(box, precision * 5);
    return std::make_tuple(
        GeoHash(int64::decode(code, precision * 5, false), precision),
        lng_boxes, lat_boxes);
  }

  /// Returns the precision in longitude/latitude and degrees for the given
  /// precision
  [[nodiscard]] static auto error_with_precision(const uint32_t precision)
      -> std::tuple<double, double> {
    if (precision > 12) {
      throw std::invalid_argument("GeoHash precision must be <= 12");
    }
    return int64::error_with_precision(precision * 5);
  }

  /// Returns the arguments to rebuild this instance.
  [[nodiscard]] auto reduce() const -> std::tuple<double, double, uint32_t> {
    auto point = center();
    return std::make_tuple(point.lon(), point.lat(), precision());
  }

 private:
  std::string code_{};

  /// Default constructor.
  explicit GeoHash(const size_t precision) : code_(precision, '\0') {}

  /// GeoHash from lon/lat and number of characters.
  GeoHash(const geodetic::Point &point, uint32_t precision)
      : GeoHash(point.lon(), point.lat(), precision) {}
};

}  // namespace pyinterp::geohash

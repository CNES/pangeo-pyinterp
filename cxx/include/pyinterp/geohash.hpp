// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "pyinterp/geometry/geographic/box.hpp"
#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"

namespace pyinterp {

/// @brief Encode geographic coordinates into a geohash string.
///
/// The more characters in the string, the more precise the location. The table
/// below gives the correspondence between the number of characters, the size of
/// the boxes of the grid at the equator and the total number of boxes.
///
/// | precision | lng/lat (km)      | samples      |
/// |-----------|-------------------|--------------|
/// | 1         | 4950/4950         | 32           |
/// | 2         | 618.75/1237.50    | 1024         |
/// | 3         | 154.69/154.69     | 32768        |
/// | 4         | 19.34/38.67       | 1048576      |
/// | 5         | 4.83/4.83         | 33554432     |
/// | 6         | 0.60/1.21         | 1073741824   |
///
/// Geohashes use Base-32 alphabet encoding (characters can be 0 to 9 and A to
/// Z, excl A, I, L and O).
///
/// The geohash is a compact way of representing a location, and is useful for
/// storing a location in a database, or for indexing a location in a database.
class GeoHash {
 public:
  /// @brief Construct a GeoHash from longitude, latitude with number of
  /// characters
  /// @param[in] longitude Longitude of the point
  /// @param[in] latitude Latitude of the point
  /// @param[in] precision Number of characters in the geohash (must be <= 12)
  /// @throw std::invalid_argument if precision > 12
  GeoHash(double longitude, double latitude, uint32_t precision);

  /// @brief Construct a GeoHash from its string representation
  /// @param[in] code String representation of the geohash
  /// @param[in] round If true, the coordinates of the point will be rounded to
  /// the accuracy defined by the GeoHash
  /// @return GeoHash instance
  /// @throw std::invalid_argument if the geohash is not valid or precision > 12
  [[nodiscard]] static auto from_string(const std::string& code, bool round)
      -> GeoHash;

  /// @brief Returns the bounding box of the geohash
  /// @return Geodetic box representing the geohash bounds
  [[nodiscard]] auto bounding_box() const -> geometry::geographic::Box;

  /// @brief Returns the center point of this geohash
  /// @return Geodetic point at the center of the geohash
  [[nodiscard]] auto center() const -> geometry::geographic::Point;

  /// @brief Returns the geohash code as a string
  /// @return String representation of the geohash
  [[nodiscard]] auto string_value() const -> std::string { return code_; }

  /// @brief Returns the precision of the geohash
  /// @return Number of characters in the geohash
  [[nodiscard]] auto precision() const -> uint32_t {
    return static_cast<uint32_t>(code_.length());
  }

  /// @brief Returns the number of bits used to represent the geohash
  /// @return Number of bits (precision * 5)
  [[nodiscard]] auto number_of_bits() const -> uint32_t {
    return precision() * 5;
  }

  /// @brief Returns the value of the integer64 stored in the geohash
  /// @param[in] round If true, returns rounded corner; otherwise centroid
  /// @return 64-bit integer representation
  [[nodiscard]] auto integer_value(bool round) const -> uint64_t;

  /// @brief Returns the eight neighbors of this geohash
  /// @return Vector of GeoHash in the order N, NE, E, SE, S, SW, W, NW
  [[nodiscard]] auto neighbors() const -> std::vector<GeoHash>;

  /// @brief Returns the area covered by this geohash
  /// @param[in] wgs Optional spheroid for area calculation
  /// @return The area of the geohash in square meters
  [[nodiscard]] auto area(
      const std::optional<geometry::geographic::Spheroid>& wgs) const -> double;

  /// @brief Gets the property of the grid covering the given box
  /// @param[in] box Geodetic box to cover
  /// @param[in] precision Number of characters in the geohash
  /// @return Tuple containing: The GeoHash of the minimum corner point, the
  /// number of cells in longitudes and latitudes
  [[nodiscard]] static auto grid_properties(
      const geometry::geographic::Box& box, uint32_t precision)
      -> std::tuple<GeoHash, size_t, size_t>;

  /// @brief Returns the precision in longitude/latitude degrees for the given
  /// precision
  /// @param[in] precision Number of characters in the geohash
  /// @return Tuple of (longitude_error, latitude_error) in degrees
  /// @throw std::invalid_argument if precision > 12
  [[nodiscard]] static auto error_with_precision(uint32_t precision)
      -> std::tuple<double, double>;

  /// @brief Returns the arguments to rebuild this instance
  /// @return Tuple of (longitude, latitude, precision)
  [[nodiscard]] auto getstate() const -> std::tuple<double, double, uint32_t>;

  /// @brief Return a string representation of the geohash
  /// @return String representation
  [[nodiscard]] explicit operator std::string() const {
    return std::format("GeoHash('{}')", code_);
  }

 private:
  /// @brief String representation of the geohash
  std::string code_{};

  /// @brief Default constructor for internal use
  /// @param[in] precision Number of characters to reserve
  explicit GeoHash(size_t precision);

  /// @brief GeoHash from point and precision
  /// @param[in] point Geodetic point
  /// @param[in] precision Number of characters
  GeoHash(const geometry::geographic::Point& point, uint32_t precision);
};

}  // namespace pyinterp

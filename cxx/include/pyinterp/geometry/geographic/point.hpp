// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>

#include "pyinterp/geometry/point.hpp"

namespace pyinterp::geometry::geographic {

/// @brief Type representing a point in geographic coordinates
class Point {
 public:
  /// @brief Build an undefined point
  constexpr Point() noexcept = default;

  /// @brief Constructs a point from longitude and latitude
  /// @param[in] lon Longitude in degrees
  /// @param[in] lat Latitude in degrees
  constexpr Point(double lon, double lat) noexcept : lon_{lon}, lat_{lat} {}

  /// @brief Get the longitude
  /// @return Longitude in degrees
  [[nodiscard]] constexpr auto lon() const noexcept -> double { return lon_; }

  /// @brief Get the latitude
  /// @return Latitude in degrees
  [[nodiscard]] constexpr auto lat() const noexcept -> double { return lat_; }

  /// @brief Get a mutable reference to the longitude
  /// @return Reference to longitude in degrees
  [[nodiscard]] constexpr auto lon() noexcept -> double& { return lon_; }

  /// @brief Get a mutable reference to the latitude
  /// @return Reference to latitude in degrees
  [[nodiscard]] constexpr auto lat() noexcept -> double& { return lat_; }

  /// @brief Templated access for Boost.Geometry traits
  /// @tparam I Index (0 for lon, 1 for lat)
  /// @return Value at index I
  template <std::size_t I>
  [[nodiscard]] constexpr auto get() const noexcept -> double {
    static_assert(I < 2, "Index out of bounds");
    if constexpr (I == 0) {
      return lon_;
    }
    return lat_;
  }

  /// @brief Templated setter for Boost.Geometry traits
  /// @tparam I Index (0 for lon, 1 for lat)
  /// @param[in] v Value to set at index I
  template <std::size_t I>
  constexpr void set(double v) noexcept {
    static_assert(I < 2, "Index out of bounds");
    I == 0 ? lon_ = v : lat_ = v;
  }

  /// @brief Equality operator
  /// @param[in] other Point to compare with
  /// @return True if both points are equal
  constexpr auto operator==(const Point& other) const noexcept -> bool {
    return lon_ == other.lon_ && lat_ == other.lat_;
  }

 private:
  /// @brief Longitude in degrees
  double lon_{};
  /// @brief Latitude in degrees
  double lat_{};
};

}  // namespace pyinterp::geometry::geographic

// Boost.Geometry traits
namespace boost::geometry::traits {

template <>
struct tag<pyinterp::geometry::geographic::Point> {
  using type = point_tag;
};

template <>
struct coordinate_type<pyinterp::geometry::geographic::Point> {
  using type = double;
};

template <>
struct coordinate_system<pyinterp::geometry::geographic::Point> {
  using type = cs::geographic<degree>;
};

template <>
struct dimension<pyinterp::geometry::geographic::Point> : boost::mpl::int_<2> {
};

template <std::size_t I>
struct access<pyinterp::geometry::geographic::Point, I> {
  static auto get(const pyinterp::geometry::geographic::Point& p) -> double {
    return p.get<I>();
  }
  static void set(pyinterp::geometry::geographic::Point& p, double v) {
    p.set<I>(v);
  }
};

}  // namespace boost::geometry::traits

template <>
struct std::hash<pyinterp::geometry::geographic::Point> {
  auto operator()(const pyinterp::geometry::geographic::Point& point)
      const noexcept -> std::size_t {
    return pyinterp::geometry::hash(point);
  }
};

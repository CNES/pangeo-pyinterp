// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>

#include "pyinterp/geometry/box.hpp"
#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/math.hpp"

namespace pyinterp::geometry::geographic {

/// @brief Type representing a bounding box in geographic coordinates
class Box : public pyinterp::geometry::Box<Point> {
 public:
  using pyinterp::geometry::Box<Point>::Box;

  /// @brief Constructor from corner points
  /// @param[in] min_corner Minimum corner point (lon, lat)
  /// @param[in] max_corner Maximum corner point (lon, lat)
  constexpr Box(const Point& min_corner, const Point& max_corner) noexcept
      : pyinterp::geometry::Box<Point>(
            Point(min_corner.lon(), min_corner.lat()),
            Point(max_corner.lon() < min_corner.lon()
                      ? math::normalize_period(max_corner.lon(),
                                               min_corner.lon(), 360.0)
                      : max_corner.lon(),
                  max_corner.lat())) {}

  /// @brief Returns the global bounding box covering the entire Earth
  [[nodiscard]]
  static constexpr auto global_bounding_box() -> Box {
    return {{-180, -90}, {180, 90}};
  }

  /// @brief Returns the center of the box.
  /// @return Center point (lon, lat)
  [[nodiscard]] inline auto centroid() const -> Point {
    return boost::geometry::return_centroid<Point, Box>(*this);
  }

  /// @brief Check if two boxes are equal
  /// @param[in] other Other box to compare with
  /// @return True if the boxes are equal, false otherwise
  [[nodiscard]] __CONSTEXPR auto operator==(const Box& other) const noexcept {
    return boost::geometry::equals(*this, other);
  }

  /// @brief Returns the delta of the box in latitude and longitude.
  /// @param[in] round If true, rounds the delta to the nearest power of 10.
  /// @return A tuple containing the delta in longitude and latitude.
  [[nodiscard]] constexpr auto delta(bool round_val) const
      -> std::tuple<double, double> {
    auto x = max_corner().template get<0>() - min_corner().template get<0>();
    auto y = max_corner().template get<1>() - min_corner().template get<1>();
    if (round_val) {
      x = max_decimal_power(x);
      y = max_decimal_power(y);
    }
    return {x, y};
  }

  /// @brief Returns a point inside the box, making an effort to round to
  /// minimal precision.
  [[nodiscard]] constexpr auto round() const -> Point {
    const auto xy = delta(true);
    const auto x = std::get<0>(xy);
    const auto y = std::get<1>(xy);
    return {std::ceil(min_corner().template get<0>() / x) * x,
            std::ceil(min_corner().template get<1>() / y) * y};
  }

 private:
  /// @brief Returns the maximum power of 10 from a number (x > 0)
  /// @param[in] x Input number
  /// @return Maximum power of 10 less than or equal to x
  [[nodiscard]]
  static __CONSTEXPR auto max_decimal_power(const double x) -> double {
    auto m = static_cast<int32_t>(std::floor(std::log10(x)));
    return math::power10(m);
  }
};

}  // namespace pyinterp::geometry::geographic

// Boost.Geometry traits
namespace boost::geometry::traits {

template <>
struct tag<pyinterp::geometry::geographic::Box> {
  using type = box_tag;
};

template <>
struct point_type<pyinterp::geometry::geographic::Box> {
  using type = pyinterp::geometry::geographic::Point;
};

template <std::size_t I, std::size_t D>
struct indexed_access<pyinterp::geometry::geographic::Box, I, D> {
  static auto get(const pyinterp::geometry::geographic::Box& b) -> double {
    if constexpr (I == min_corner) {
      return b.min_corner().get<D>();
    } else {
      return b.max_corner().get<D>();
    }
  }
  static void set(pyinterp::geometry::geographic::Box& b, double v) {
    if constexpr (I == min_corner) {
      b.min_corner().set<D>(v);
    } else {
      b.max_corner().set<D>(v);
    }
  };
};

}  // namespace boost::geometry::traits

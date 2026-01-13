// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <array>
#include <boost/geometry.hpp>
#include <cstddef>
#include <limits>
#include <utility>

#include "boost/geometry/core/coordinate_dimension.hpp"
#include "boost/geometry/core/coordinate_type.hpp"

namespace pyinterp::geometry::point {

/// Concept for Boost.Geometry point types
template <typename T>
concept BoostGeometryPoint = requires(T t) {
  typename boost::geometry::traits::dimension<T>;
  typename boost::geometry::traits::coordinate_type<T>;
};

/// Helper to get coordinate type from a point
template <BoostGeometryPoint Point>
using coordinate_t =
    typename boost::geometry::traits::coordinate_type<Point>::type;

/// Helper to get dimension count from a point
template <BoostGeometryPoint Point>
inline constexpr size_t dimension_v = boost::geometry::dimension<Point>::value;

/// Get coordinate value for a given axis (compile-time axis selection).
/// @tparam Axis Axis index (0-based, must be < dimension)
/// @param[in] point Point to access
/// @return Coordinate value at the specified axis
namespace detail {
template <BoostGeometryPoint Point, size_t... Is>
[[nodiscard]] constexpr auto get_impl(const Point& point, const size_t axis,
                                      std::index_sequence<Is...>) noexcept
    -> coordinate_t<Point> {
  coordinate_t<Point> result = std::numeric_limits<coordinate_t<Point>>::max();

  // Fold expression with side effects (C++17)
  (void)((Is == axis ? (result = boost::geometry::get<Is>(point), true)
                     : false) ||
         ...);

  return result;
}

/// Set coordinate value for a given axis (compile-time axis selection).
/// @tparam Axis Axis index (0-based, must be < dimension)
/// @param[in,out] point Point to modify
/// @param[in] value Value to set
template <BoostGeometryPoint Point, size_t... Is>
constexpr auto set_impl(Point& point, const coordinate_t<Point> value,
                        const size_t axis, std::index_sequence<Is...>) noexcept
    -> void {
  // Fold expression with side effects (C++17)
  (void)((Is == axis ? (boost::geometry::set<Is>(point, value), true)
                     : false) ||
         ...);
}

}  // namespace detail

/// Get coordinate value for a given axis (runtime axis selection).
/// @param[in] point Point to access
/// @param[in] axis Axis index (0-based, must be < dimension)
/// @return Coordinate value at the specified axis
template <BoostGeometryPoint Point>
[[nodiscard]] constexpr auto get(const Point& point, const size_t axis) noexcept
    -> coordinate_t<Point> {
  return detail::get_impl(point, axis,
                          std::make_index_sequence<dimension_v<Point>>{});
}

/// Set coordinate value for a given axis (runtime axis selection).
/// @param[in,out] point Point to modify
/// @param[in] value Value to set
/// @param[in] axis Axis index (0-based, must be < dimension)
template <BoostGeometryPoint Point>
constexpr auto set(Point& point, const coordinate_t<Point> value,
                   const size_t axis) noexcept -> void {
  // Use modern index_sequence approach
  detail::set_impl(point, value, axis,
                   std::make_index_sequence<dimension_v<Point>>{});
}

/// Get all coordinates of a point as an array (compile-time known size).
/// @param[in] point Point to access
/// @return Array of coordinate values
template <BoostGeometryPoint Point>
[[nodiscard]] constexpr auto to_array(const Point& point) noexcept
    -> std::array<coordinate_t<Point>, dimension_v<Point>> {
  return [&]<size_t... Is>(std::index_sequence<Is...>) {
    return std::array<coordinate_t<Point>, dimension_v<Point>>{
        boost::geometry::get<Is>(point)...};
  }(std::make_index_sequence<dimension_v<Point>>{});
}

/// Create a point from an array of coordinates.
/// @param[in] coords Array of coordinate values
/// @return Constructed point
template <BoostGeometryPoint Point>
[[nodiscard]] constexpr auto from_array(
    const std::array<coordinate_t<Point>, dimension_v<Point>>& coords) noexcept
    -> Point {
  Point result;
  [&]<size_t... Is>(std::index_sequence<Is...>) {
    (boost::geometry::set<Is>(result, coords[Is]), ...);
  }(std::make_index_sequence<dimension_v<Point>>{});
  return result;
}

}  // namespace pyinterp::geometry::point

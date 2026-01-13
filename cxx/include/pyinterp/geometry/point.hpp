// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>
#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <concepts>

#include "pyinterp/geometry/point_traits.hpp"

namespace pyinterp::geometry {

/// @brief 3D point in geodetic coordinates (longitude, latitude, altitude)
/// Longitude and latitude are expressed in degrees, altitude in meters
/// @tparam T Floating-point type
template <std::floating_point T>
using LLA = boost::geometry::model::point<
    T, 3, boost::geometry::cs::geographic<boost::geometry::degree>>;

/// @brief 3D point in Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates
/// x, y, z are expressed in meters
template <std::floating_point T>
using ECEF =
    boost::geometry::model::point<T, 3, boost::geometry::cs::cartesian>;

/// @brief 2D point in geodetic coordinates (longitude, latitude)
/// projected onto a spheroidal model for enhanced accuracy
/// Longitude and latitude are expressed in degrees
/// @tparam T Floating-point type
template <std::floating_point T>
using GeographicPoint = boost::geometry::model::point<
    T, 2, boost::geometry::cs::geographic<boost::geometry::degree>>;

/// @brief 2D point in spherical coordinates (longitude, latitude)
/// Uses a perfect sphere model for calculations
/// Longitude and latitude are expressed in degrees
/// @tparam T Floating-point type
template <std::floating_point T>
using SphericalPoint = boost::geometry::model::point<
    T, 2, boost::geometry::cs::spherical_equatorial<boost::geometry::degree>>;

/// @brief Hash function for a point
/// @tparam Point Type of point
/// @param[in] point Point to hash
/// @return Hash value
template <typename Point>
constexpr auto hash(const Point& point) noexcept -> size_t {
  size_t seed = 0;
  auto combine = [&](double v) {
    seed ^= std::hash<double>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  };
  for (size_t ix = 0; ix < boost::geometry::dimension<Point>::value; ++ix) {
    combine(geometry::point::get(point, ix));
  }
  return seed;
}

}  // namespace pyinterp::geometry

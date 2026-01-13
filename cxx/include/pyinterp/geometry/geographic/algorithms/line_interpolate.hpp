// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>

#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"

namespace pyinterp::geometry::geographic {

/// @brief Create line_interpolate strategy based on the specified method and
/// spheroid
/// @tparam Method Strategy method to use
/// @param[in] spheroid Spheroid for geodetic calculations
/// @return line_interpolate strategy
template <StrategyMethod Method>
[[nodiscard]] constexpr auto make_line_interpolate_strategy(
    const boost::geometry::srs::spheroid<double> &spheroid) {
  if constexpr (Method == StrategyMethod::kAndoyer) {
    return boost::geometry::strategies::line_interpolate::geographic<
        boost::geometry::strategy::andoyer>(spheroid);
  } else if constexpr (Method == StrategyMethod::kKarney) {
    return boost::geometry::strategies::line_interpolate::geographic<
        boost::geometry::strategy::karney>(spheroid);
  } else if constexpr (Method == StrategyMethod::kThomas) {
    return boost::geometry::strategies::line_interpolate::geographic<
        boost::geometry::strategy::thomas>(spheroid);
  } else if constexpr (Method == StrategyMethod::kVincenty) {
    return boost::geometry::strategies::line_interpolate::geographic<
        boost::geometry::strategy::vincenty>(spheroid);
  } else {
    static_assert(false, "Unhandled StrategyMethod");
  }
}

/// @brief Interpolate a point along a geometry using a compile-time strategy
/// @tparam Geometry Geometry type
/// @tparam Method Strategy method to use
/// @param[in] geometry Geometry object
/// @param[in] distance Distance along the geometry
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @return Interpolated point
template <typename Geometry, StrategyMethod Method>
[[nodiscard]] inline auto line_interpolate(
    const Geometry &geometry, double distance,
    const std::optional<Spheroid> &spheroid) -> Point {
  Point result;
  boost::geometry::line_interpolate(
      geometry, distance, result,
      make_line_interpolate_strategy<Method>(make_spheroid(spheroid)));
  return result;
}

/// @brief Interpolate a point along a geometry using a runtime strategy
/// @tparam Geometry Geometry type
/// @param[in] geometry Geometry object
/// @param[in] distance Distance along the geometry
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @param[in] strategy Strategy method to use
/// @return Interpolated point
template <typename Geometry>
[[nodiscard]] inline auto line_interpolate(
    const Geometry &geometry, double distance,
    const std::optional<Spheroid> &spheroid, const StrategyMethod strategy)
    -> Point {
  using enum StrategyMethod;
  switch (strategy) {
    case kAndoyer:
      return line_interpolate<Geometry, kAndoyer>(geometry, distance, spheroid);
    case kKarney:
      return line_interpolate<Geometry, kKarney>(geometry, distance, spheroid);
    case kThomas:
      return line_interpolate<Geometry, kThomas>(geometry, distance, spheroid);
    case kVincenty:
      return line_interpolate<Geometry, kVincenty>(geometry, distance,
                                                   spheroid);
  }
  std::unreachable();
}

}  // namespace pyinterp::geometry::geographic

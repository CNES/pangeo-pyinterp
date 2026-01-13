// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/detail/closest_points/interface.hpp>

#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/geographic/segment.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"

namespace pyinterp::geometry::geographic {

/// @brief Create closest points strategy based on the specified method
/// and spheroid
/// @tparam Method Strategy method to use
/// @param[in] spheroid Spheroid for geodetic calculations
/// @return Closest points strategy
template <StrategyMethod Method>
[[nodiscard]] constexpr auto make_closest_points_strategy(
    const boost::geometry::srs::spheroid<double> &spheroid) {
  if constexpr (Method == StrategyMethod::kAndoyer) {
    return boost::geometry::strategies::closest_points::geographic(spheroid);
  } else if constexpr (Method == StrategyMethod::kKarney) {
    return boost::geometry::strategies::closest_points::geographic(spheroid);
  } else if constexpr (Method == StrategyMethod::kThomas) {
    return boost::geometry::strategies::closest_points::geographic(spheroid);
  } else if constexpr (Method == StrategyMethod::kVincenty) {
    return boost::geometry::strategies::closest_points::geographic(spheroid);
  } else {
    static_assert(false, "Unhandled StrategyMethod");
  }
}

/// @brief Calculate the closest points using a compile-time strategy
/// @tparam Geometry Geometry type
/// @tparam Method Strategy method to use
/// @param[in] geometry1 First geometry
/// @param[in] geometry2 Second geometry
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @return Calculated closest points
template <typename Geometry1, typename Geometry2, StrategyMethod Method>
[[nodiscard]] inline auto closest_points(
    const Geometry1 &geometry1, const Geometry2 &geometry2,
    const std::optional<Spheroid> &spheroid) -> Segment {
  Segment segment;
  boost::geometry::closest_points(
      geometry1, geometry2, segment,
      make_closest_points_strategy<Method>(make_spheroid(spheroid)));
  return segment;
}

/// @brief Calculate the closest points using a runtime strategy
/// @tparam Geometry Geometry type
/// @param[in] geometry1 First geometry
/// @param[in] geometry2 Second geometry
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @param[in] strategy Strategy method to use
/// @return Calculated closest points
template <typename Geometry1, typename Geometry2>
[[nodiscard]] inline auto closest_points(
    const Geometry1 &geometry1, const Geometry2 &geometry2,
    const std::optional<Spheroid> &spheroid, const StrategyMethod strategy)
    -> Segment {
  using enum StrategyMethod;
  switch (strategy) {
    case kAndoyer:
      return closest_points<Geometry1, Geometry2, kAndoyer>(
          geometry1, geometry2, spheroid);
    case kKarney:
      return closest_points<Geometry1, Geometry2, kKarney>(geometry1, geometry2,
                                                           spheroid);
    case kThomas:
      return closest_points<Geometry1, Geometry2, kThomas>(geometry1, geometry2,
                                                           spheroid);
    case kVincenty:
      return closest_points<Geometry1, Geometry2, kVincenty>(
          geometry1, geometry2, spheroid);
  }
  std::unreachable();
}

}  // namespace pyinterp::geometry::geographic

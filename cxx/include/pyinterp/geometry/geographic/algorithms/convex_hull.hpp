// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/detail/convex_hull/interface.hpp>

#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/geographic/polygon.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"

namespace pyinterp::geometry::geographic {

/// @brief Create convex hull strategy based on the specified method
/// and spheroid
/// @tparam Method Strategy method to use
/// @param[in] spheroid Spheroid for geodetic calculations
/// @return convex hull strategy
template <StrategyMethod Method>
[[nodiscard]] constexpr auto make_convex_hull_strategy(
    const boost::geometry::srs::spheroid<double> &spheroid) {
  if constexpr (Method == StrategyMethod::kAndoyer) {
    return boost::geometry::strategies::convex_hull::geographic<
        boost::geometry::strategy::andoyer>(spheroid);
  } else if constexpr (Method == StrategyMethod::kKarney) {
    return boost::geometry::strategies::convex_hull::geographic<
        boost::geometry::strategy::karney>(spheroid);
  } else if constexpr (Method == StrategyMethod::kThomas) {
    return boost::geometry::strategies::convex_hull::geographic<
        boost::geometry::strategy::thomas>(spheroid);
  } else if constexpr (Method == StrategyMethod::kVincenty) {
    return boost::geometry::strategies::convex_hull::geographic<
        boost::geometry::strategy::vincenty>(spheroid);
  } else {
    static_assert(false, "Unhandled StrategyMethod");
  }
}

/// @brief Calculate the convex hull using a compile-time strategy
/// @tparam Geometry Geometry type
/// @tparam Method Strategy method to use
/// @param[in] geometry Geometry object
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @return Calculated convex hull
template <typename Geometry, StrategyMethod Method>
[[nodiscard]] inline auto convex_hull(const Geometry &geometry,
                                      const std::optional<Spheroid> &spheroid)
    -> Polygon {
  Polygon result;
  boost::geometry::convex_hull(
      geometry, result,
      make_convex_hull_strategy<Method>(make_spheroid(spheroid)));
  return result;
}

/// @brief Calculate the convex hull using a runtime strategy
/// @tparam Geometry Geometry type
/// @param[in] geometry Geometry object
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @param[in] strategy Strategy method to use
/// @return Calculated convex hull
template <typename Geometry>
[[nodiscard]] inline auto convex_hull(const Geometry &geometry,
                                      const std::optional<Spheroid> &spheroid,
                                      const StrategyMethod strategy)
    -> Polygon {
  using enum StrategyMethod;
  switch (strategy) {
    case kAndoyer:
      return convex_hull<Geometry, kAndoyer>(geometry, spheroid);
    case kKarney:
      return convex_hull<Geometry, kKarney>(geometry, spheroid);
    case kThomas:
      return convex_hull<Geometry, kThomas>(geometry, spheroid);
    case kVincenty:
      return convex_hull<Geometry, kVincenty>(geometry, spheroid);
  }
  std::unreachable();
}

}  // namespace pyinterp::geometry::geographic

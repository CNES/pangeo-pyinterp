// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>

#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"

namespace pyinterp::geometry::geographic {

/// @brief Create perimeter strategy based on the specified method and spheroid
/// @tparam Method Strategy method to use
/// @param[in] spheroid Spheroid for geodetic calculations
/// @return perimeter strategy (uses distance strategy)
template <StrategyMethod Method>
[[nodiscard]] constexpr auto make_perimeter_strategy(
    const boost::geometry::srs::spheroid<double> &spheroid) {
  // Perimeter uses distance strategy
  if constexpr (Method == StrategyMethod::kAndoyer) {
    return boost::geometry::strategy::distance::geographic<
        boost::geometry::strategy::andoyer>(spheroid);
  } else if constexpr (Method == StrategyMethod::kKarney) {
    return boost::geometry::strategy::distance::geographic<
        boost::geometry::strategy::karney>(spheroid);
  } else if constexpr (Method == StrategyMethod::kThomas) {
    return boost::geometry::strategy::distance::geographic<
        boost::geometry::strategy::thomas>(spheroid);
  } else if constexpr (Method == StrategyMethod::kVincenty) {
    return boost::geometry::strategy::distance::geographic<
        boost::geometry::strategy::vincenty>(spheroid);
  } else {
    static_assert(false, "Unhandled StrategyMethod");
  }
}

/// @brief Calculate perimeter using a compile-time strategy
/// @tparam Geometry Geometry type
/// @tparam Method Strategy method to use
/// @param[in] geometry Geometry object
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @return Perimeter in meters
template <typename Geometry, StrategyMethod Method>
[[nodiscard]] inline auto perimeter(const Geometry &geometry,
                                    const std::optional<Spheroid> &spheroid)
    -> double {
  return boost::geometry::perimeter(
      geometry, make_perimeter_strategy<Method>(make_spheroid(spheroid)));
}

/// @brief Calculate perimeter using a runtime strategy
/// @tparam Geometry Geometry type
/// @param[in] geometry Geometry object
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @param[in] strategy Strategy method to use
/// @return Perimeter in meters
template <typename Geometry>
[[nodiscard]] inline auto perimeter(const Geometry &geometry,
                                    const std::optional<Spheroid> &spheroid,
                                    const StrategyMethod strategy) -> double {
  using enum StrategyMethod;
  switch (strategy) {
    case kAndoyer:
      return perimeter<Geometry, kAndoyer>(geometry, spheroid);
    case kKarney:
      return perimeter<Geometry, kKarney>(geometry, spheroid);
    case kThomas:
      return perimeter<Geometry, kThomas>(geometry, spheroid);
    case kVincenty:
      return perimeter<Geometry, kVincenty>(geometry, spheroid);
  }
  std::unreachable();
}

}  // namespace pyinterp::geometry::geographic

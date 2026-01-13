// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>

#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"

namespace pyinterp::geometry::geographic {

/// @brief Create simplify strategy based on the specified method and spheroid
/// @tparam Method Strategy method to use
/// @param[in] spheroid Spheroid for geodetic calculations
/// @return simplify strategy
template <StrategyMethod Method>
[[nodiscard]] constexpr auto make_simplify_strategy(
    const boost::geometry::srs::spheroid<double> &spheroid) {
  if constexpr (Method == StrategyMethod::kAndoyer) {
    return boost::geometry::strategies::simplify::geographic<
        boost::geometry::strategy::andoyer>(spheroid);
  } else if constexpr (Method == StrategyMethod::kKarney) {
    return boost::geometry::strategies::simplify::geographic<
        boost::geometry::strategy::karney>(spheroid);
  } else if constexpr (Method == StrategyMethod::kThomas) {
    return boost::geometry::strategies::simplify::geographic<
        boost::geometry::strategy::thomas>(spheroid);
  } else if constexpr (Method == StrategyMethod::kVincenty) {
    return boost::geometry::strategies::simplify::geographic<
        boost::geometry::strategy::vincenty>(spheroid);
  } else {
    static_assert(false, "Unhandled StrategyMethod");
  }
}

/// @brief Simplify a geometry using a compile-time strategy
/// @tparam Geometry Geometry type
/// @tparam Method Strategy method to use
/// @param[in] geometry Geometry object
/// @param[in] max_distance Maximum distance for simplification
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @return Simplified geometry
template <typename Geometry, StrategyMethod Method>
[[nodiscard]] inline auto simplify(const Geometry &geometry,
                                   double max_distance,
                                   const std::optional<Spheroid> &spheroid)
    -> Geometry {
  Geometry result;
  boost::geometry::simplify(
      geometry, result, max_distance,
      make_simplify_strategy<Method>(make_spheroid(spheroid)));
  return result;
}

/// @brief Simplify a geometry using a runtime strategy
/// @tparam Geometry Geometry type
/// @param[in] geometry Geometry object
/// @param[in] max_distance Maximum distance for simplification
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @param[in] strategy Strategy method to use
/// @return Simplified geometry
template <typename Geometry>
[[nodiscard]] inline auto simplify(const Geometry &geometry,
                                   double max_distance,
                                   const std::optional<Spheroid> &spheroid,
                                   const StrategyMethod strategy) -> Geometry {
  using enum StrategyMethod;
  switch (strategy) {
    case kAndoyer:
      return simplify<Geometry, kAndoyer>(geometry, max_distance, spheroid);
    case kKarney:
      return simplify<Geometry, kKarney>(geometry, max_distance, spheroid);
    case kThomas:
      return simplify<Geometry, kThomas>(geometry, max_distance, spheroid);
    case kVincenty:
      return simplify<Geometry, kVincenty>(geometry, max_distance, spheroid);
  }
  std::unreachable();
}

}  // namespace pyinterp::geometry::geographic

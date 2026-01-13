// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>
#include <vector>

#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/geographic/polygon.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"

namespace pyinterp::geometry::geographic {

/// @brief Create difference strategy based on the specified method and spheroid
/// @tparam Method Strategy method to use
/// @param[in] spheroid Spheroid for geodetic calculations
/// @return difference strategy
template <StrategyMethod Method>
[[nodiscard]] constexpr auto make_relate_strategy(
    const boost::geometry::srs::spheroid<double> &spheroid) {
  if constexpr (Method == StrategyMethod::kAndoyer) {
    return boost::geometry::strategies::relate::geographic<
        boost::geometry::strategy::andoyer>(spheroid);
  } else if constexpr (Method == StrategyMethod::kKarney) {
    return boost::geometry::strategies::relate::geographic<
        boost::geometry::strategy::karney>(spheroid);
  } else if constexpr (Method == StrategyMethod::kThomas) {
    return boost::geometry::strategies::relate::geographic<
        boost::geometry::strategy::thomas>(spheroid);
  } else if constexpr (Method == StrategyMethod::kVincenty) {
    return boost::geometry::strategies::relate::geographic<
        boost::geometry::strategy::vincenty>(spheroid);
  } else {
    static_assert(false, "Unhandled StrategyMethod");
  }
}

/// @brief Calculate the difference using a compile-time strategy
/// @tparam Geometry1 First geometry type
/// @tparam Geometry2 Second geometry type
/// @tparam Method Strategy method to use
/// @param[in] geometry1 First geometry object
/// @param[in] geometry2 Second geometry object
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @return Vector of polygons representing the difference
template <typename Geometry1, typename Geometry2, StrategyMethod Method>
[[nodiscard]] inline auto difference(const Geometry1 &geometry1,
                                     const Geometry2 &geometry2,
                                     const std::optional<Spheroid> &spheroid)
    -> std::vector<Polygon> {
  std::vector<Polygon> result;
  boost::geometry::difference(
      geometry1, geometry2, result,
      make_relate_strategy<Method>(make_spheroid(spheroid)));
  return result;
}

/// @brief Calculate the difference using a runtime strategy
/// @tparam Geometry1 First geometry type
/// @tparam Geometry2 Second geometry type
/// @param[in] geometry1 First geometry object
/// @param[in] geometry2 Second geometry object
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @param[in] strategy Strategy method to use
/// @return Vector of polygons representing the difference
template <typename Geometry1, typename Geometry2>
[[nodiscard]] inline auto difference(const Geometry1 &geometry1,
                                     const Geometry2 &geometry2,
                                     const std::optional<Spheroid> &spheroid,
                                     const StrategyMethod strategy)
    -> std::vector<Polygon> {
  using enum StrategyMethod;
  switch (strategy) {
    case kAndoyer:
      return difference<Geometry1, Geometry2, kAndoyer>(geometry1, geometry2,
                                                        spheroid);
    case kKarney:
      return difference<Geometry1, Geometry2, kKarney>(geometry1, geometry2,
                                                       spheroid);
    case kThomas:
      return difference<Geometry1, Geometry2, kThomas>(geometry1, geometry2,
                                                       spheroid);
    case kVincenty:
      return difference<Geometry1, Geometry2, kVincenty>(geometry1, geometry2,
                                                         spheroid);
  }
  std::unreachable();
}

}  // namespace pyinterp::geometry::geographic

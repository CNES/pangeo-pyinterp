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

/// @brief Create azimuth strategy based on the specified method
/// and spheroid
/// @tparam Method Strategy method to use
/// @param[in] spheroid Spheroid for geodetic calculations
/// @return Azimuth strategy
template <StrategyMethod Method>
[[nodiscard]] constexpr auto make_azimuth_strategy(
    const boost::geometry::srs::spheroid<double> &spheroid) {
  if constexpr (Method == StrategyMethod::kAndoyer) {
    return boost::geometry::strategy::azimuth::geographic<
        boost::geometry::strategy::andoyer>(spheroid);
  } else if constexpr (Method == StrategyMethod::kKarney) {
    return boost::geometry::strategy::azimuth::geographic<
        boost::geometry::strategy::karney>(spheroid);
  } else if constexpr (Method == StrategyMethod::kThomas) {
    return boost::geometry::strategy::azimuth::geographic<
        boost::geometry::strategy::thomas>(spheroid);
  } else if constexpr (Method == StrategyMethod::kVincenty) {
    return boost::geometry::strategy::azimuth::geographic<
        boost::geometry::strategy::vincenty>(spheroid);
  } else {
    static_assert(false, "Unhandled StrategyMethod");
  }
}

/// @brief Calculate the azimuth using a compile-time strategy
/// @tparam Geometry Geometry type
/// @tparam Method Strategy method to use
/// @param[in] point1 First point
/// @param[in] point2 Second point
/// @param[in] wgs Optional Spheroid for geodetic calculations
/// @return Calculated azimuth
template <StrategyMethod Method>
[[nodiscard]] inline auto azimuth(const Point &point1, const Point &point2,
                                  const std::optional<Spheroid> &wgs)
    -> double {
  return boost::geometry::azimuth(
      point1, point2, make_azimuth_strategy<Method>(make_spheroid(wgs)));
}

/// @brief Calculate the azimuth using a runtime strategy
/// @tparam Geometry Geometry type
/// @param[in] point1 First point
/// @param[in] point2 Second point
/// @param[in] wgs Optional Spheroid for geodetic calculations
/// @param[in] strategy Strategy method to use
/// @return Calculated azimuth
[[nodiscard]] inline auto azimuth(const Point &point1, const Point &point2,
                                  const std::optional<Spheroid> &wgs,
                                  const StrategyMethod strategy) -> double {
  using enum StrategyMethod;
  switch (strategy) {
    case kAndoyer:
      return azimuth<kAndoyer>(point1, point2, wgs);
    case kKarney:
      return azimuth<kKarney>(point1, point2, wgs);
    case kThomas:
      return azimuth<kThomas>(point1, point2, wgs);
    case kVincenty:
      return azimuth<kVincenty>(point1, point2, wgs);
  }
  std::unreachable();
}

}  // namespace pyinterp::geometry::geographic

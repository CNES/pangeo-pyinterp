// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>
#include <vector>

#include "pyinterp/geometry/geographic/algorithms/difference.hpp"
#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/geographic/linestring.hpp"
#include "pyinterp/geometry/geographic/polygon.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"

namespace pyinterp::geometry::geographic {

/// @brief Calculate polygon union using a compile-time strategy
/// @tparam Geometry1 First geometry type
/// @tparam Geometry2 Second geometry type
/// @tparam Method Strategy method to use
/// @param[in] geometry1 First geometry object
/// @param[in] geometry2 Second geometry object
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @return Vector of polygons representing the union
template <typename Geometry1, typename Geometry2, StrategyMethod Method>
[[nodiscard]] inline auto union_polygon(const Geometry1 &geometry1,
                                        const Geometry2 &geometry2,
                                        const std::optional<Spheroid> &spheroid)
    -> std::vector<Polygon> {
  std::vector<Polygon> result;
  boost::geometry::union_(
      geometry1, geometry2, result,
      make_relate_strategy<Method>(make_spheroid(spheroid)));
  return result;
}

/// @brief Calculate polygon union using a runtime strategy
/// @tparam Geometry1 First geometry type
/// @tparam Geometry2 Second geometry type
/// @param[in] geometry1 First geometry object
/// @param[in] geometry2 Second geometry object
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @param[in] strategy Strategy method to use
/// @return Vector of polygons representing the union
template <typename Geometry1, typename Geometry2>
[[nodiscard]] inline auto union_polygon(const Geometry1 &geometry1,
                                        const Geometry2 &geometry2,
                                        const std::optional<Spheroid> &spheroid,
                                        const StrategyMethod strategy)
    -> std::vector<Polygon> {
  using enum StrategyMethod;
  switch (strategy) {
    case kAndoyer:
      return union_polygon<Geometry1, Geometry2, kAndoyer>(geometry1, geometry2,
                                                           spheroid);
    case kKarney:
      return union_polygon<Geometry1, Geometry2, kKarney>(geometry1, geometry2,
                                                          spheroid);
    case kThomas:
      return union_polygon<Geometry1, Geometry2, kThomas>(geometry1, geometry2,
                                                          spheroid);
    case kVincenty:
      return union_polygon<Geometry1, Geometry2, kVincenty>(
          geometry1, geometry2, spheroid);
  }
  std::unreachable();
}

/// @brief Calculate linestring union using a compile-time strategy
/// @tparam Geometry1 First geometry type
/// @tparam Geometry2 Second geometry type
/// @tparam Method Strategy method to use
/// @param[in] geometry1 First geometry object
/// @param[in] geometry2 Second geometry object
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @return Vector of linestrings representing the union
template <typename Geometry1, typename Geometry2, StrategyMethod Method>
[[nodiscard]] inline auto union_linestring(
    const Geometry1 &geometry1, const Geometry2 &geometry2,
    const std::optional<Spheroid> &spheroid) -> std::vector<LineString> {
  std::vector<LineString> result;
  boost::geometry::union_(
      geometry1, geometry2, result,
      make_relate_strategy<Method>(make_spheroid(spheroid)));
  return result;
}

/// @brief Calculate linestring union using a runtime strategy
/// @tparam Geometry1 First geometry type
/// @tparam Geometry2 Second geometry type
/// @param[in] geometry1 First geometry object
/// @param[in] geometry2 Second geometry object
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @param[in] strategy Strategy method to use
/// @return Vector of linestrings representing the union
template <typename Geometry1, typename Geometry2>
[[nodiscard]] inline auto union_linestring(
    const Geometry1 &geometry1, const Geometry2 &geometry2,
    const std::optional<Spheroid> &spheroid, const StrategyMethod strategy)
    -> std::vector<LineString> {
  using enum StrategyMethod;
  switch (strategy) {
    case kAndoyer:
      return union_linestring<Geometry1, Geometry2, kAndoyer>(
          geometry1, geometry2, spheroid);
    case kKarney:
      return union_linestring<Geometry1, Geometry2, kKarney>(
          geometry1, geometry2, spheroid);
    case kThomas:
      return union_linestring<Geometry1, Geometry2, kThomas>(
          geometry1, geometry2, spheroid);
    case kVincenty:
      return union_linestring<Geometry1, Geometry2, kVincenty>(
          geometry1, geometry2, spheroid);
  }
  std::unreachable();
}

}  // namespace pyinterp::geometry::geographic

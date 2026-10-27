// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

// IWYU pragma: begin_keep
#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/geographic/linestring.hpp"
#include "pyinterp/geometry/geographic/multi_linestring.hpp"
#include "pyinterp/geometry/geographic/multi_polygon.hpp"
#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/geographic/polygon.hpp"
#include "pyinterp/geometry/geographic/ring.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"
// IWYU pragma: end_keep

// The overlay operations (difference, intersection and union) instantiate the
// Boost.Geometry overlay algorithm for every pair of geometries and every
// strategy, which made their bindings by far the slowest translation units of
// the extension. The bindings therefore only see the declaration of the
// operation specialized for a strategy; each strategy is explicitly
// instantiated in its own translation unit (see overlay_instances.hpp), so
// that these instantiations are compiled in parallel.

/// @brief Geometry pairs whose difference is a list of polygons
#define DIFFERENCE_POLYGON_PAIRS(NS)                                         \
  std::pair<NS::Ring, NS::Ring>, std::pair<NS::Ring, NS::Polygon>,           \
      std::pair<NS::Polygon, NS::Ring>, std::pair<NS::Polygon, NS::Polygon>, \
      std::pair<NS::MultiPolygon, NS::Ring>,                                 \
      std::pair<NS::MultiPolygon, NS::Polygon>,                              \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>,                         \
      std::pair<NS::Ring, NS::MultiPolygon>,                                 \
      std::pair<NS::Polygon, NS::MultiPolygon>

/// @brief Geometry pairs whose intersection is a list of polygons
#define INTERSECTION_POLYGON_PAIRS(NS)                                       \
  std::pair<NS::Ring, NS::Ring>, std::pair<NS::Ring, NS::Polygon>,           \
      std::pair<NS::Polygon, NS::Ring>, std::pair<NS::Polygon, NS::Polygon>, \
      std::pair<NS::MultiPolygon, NS::Ring>,                                 \
      std::pair<NS::MultiPolygon, NS::Polygon>,                              \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>,                         \
      std::pair<NS::Ring, NS::MultiPolygon>,                                 \
      std::pair<NS::Polygon, NS::MultiPolygon>

/// @brief Geometry pairs whose intersection is a list of linestrings
#define INTERSECTION_LINESTRING_POLYGON_PAIRS(NS)                              \
  std::pair<NS::LineString, NS::Ring>, std::pair<NS::LineString, NS::Polygon>, \
      std::pair<NS::LineString, NS::MultiPolygon>

/// @brief Geometry pairs whose intersection is a list of points
#define INTERSECTION_LINESTRING_PAIRS(NS) \
  std::pair<NS::LineString, NS::LineString>

/// @brief Geometry pairs whose union is a list of polygons
#define UNION_POLYGON_PAIRS(NS)                                              \
  std::pair<NS::Ring, NS::Ring>, std::pair<NS::Ring, NS::Polygon>,           \
      std::pair<NS::Polygon, NS::Ring>, std::pair<NS::Polygon, NS::Polygon>, \
      std::pair<NS::MultiPolygon, NS::Ring>,                                 \
      std::pair<NS::MultiPolygon, NS::Polygon>,                              \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>,                         \
      std::pair<NS::Ring, NS::MultiPolygon>,                                 \
      std::pair<NS::Polygon, NS::MultiPolygon>

/// @brief Geometry pairs whose union is a list of linestrings
#define UNION_LINESTRING_PAIRS(NS)           \
  std::pair<NS::LineString, NS::LineString>, \
      std::pair<NS::MultiLineString, NS::MultiLineString>

namespace pyinterp::geometry::geographic::pybind {

/// @brief Overlay operations
enum class Overlay : std::uint8_t {
  kDifference,    ///< Difference of two geometries
  kIntersection,  ///< Intersection of two geometries
  kUnion          ///< Union of two geometries
};

/// @brief Computes an overlay operation using a given strategy
///
/// Only declared here: the specializations are explicitly instantiated, one
/// translation unit per strategy, from overlay_instances.hpp.
///
/// @tparam Operation Overlay operation to compute
/// @tparam Result Type of the geometries returned by the operation
/// @tparam Method Calculation strategy
/// @param[in] geometry1 First geometry
/// @param[in] geometry2 Second geometry
/// @param[in] spheroid Optional spheroid (WGS84 if not provided)
/// @return The geometries resulting from the operation
template <Overlay Operation, typename Result, StrategyMethod Method,
          typename Geometry1, typename Geometry2>
auto overlay_strategy(const Geometry1 &geometry1, const Geometry2 &geometry2,
                      const std::optional<Spheroid> &spheroid)
    -> std::vector<Result>;

/// @brief Computes an overlay operation using the strategy selected at runtime
/// @tparam Operation Overlay operation to compute
/// @tparam Result Type of the geometries returned by the operation
/// @param[in] geometry1 First geometry
/// @param[in] geometry2 Second geometry
/// @param[in] spheroid Optional spheroid (WGS84 if not provided)
/// @param[in] strategy Calculation strategy
/// @return The geometries resulting from the operation
template <Overlay Operation, typename Result, typename Geometry1,
          typename Geometry2>
[[nodiscard]] inline auto overlay(const Geometry1 &geometry1,
                                  const Geometry2 &geometry2,
                                  const std::optional<Spheroid> &spheroid,
                                  const StrategyMethod strategy)
    -> std::vector<Result> {
  using enum StrategyMethod;
  switch (strategy) {
    case kAndoyer:
      return overlay_strategy<Operation, Result, kAndoyer>(geometry1, geometry2,
                                                           spheroid);
    case kKarney:
      return overlay_strategy<Operation, Result, kKarney>(geometry1, geometry2,
                                                          spheroid);
    case kThomas:
      return overlay_strategy<Operation, Result, kThomas>(geometry1, geometry2,
                                                          spheroid);
    case kVincenty:
      return overlay_strategy<Operation, Result, kVincenty>(
          geometry1, geometry2, spheroid);
  }
  std::unreachable();
}

}  // namespace pyinterp::geometry::geographic::pybind

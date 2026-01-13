// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <cstdint>

// IWYU pragma: begin_keep
// Helper functions for binding geometry algorithms with optional arguments
// using nanobind. IWYU pragmas below preserve geometry type includes that are
// required by template fold expressions but may not be detected automatically.
#include "pyinterp/geometry/cartesian/box.hpp"
#include "pyinterp/geometry/cartesian/linestring.hpp"
#include "pyinterp/geometry/cartesian/multi_linestring.hpp"
#include "pyinterp/geometry/cartesian/multi_point.hpp"
#include "pyinterp/geometry/cartesian/multi_polygon.hpp"
#include "pyinterp/geometry/cartesian/point.hpp"
#include "pyinterp/geometry/cartesian/polygon.hpp"
#include "pyinterp/geometry/cartesian/ring.hpp"
#include "pyinterp/geometry/cartesian/segment.hpp"
#include "pyinterp/geometry/geographic/box.hpp"
#include "pyinterp/geometry/geographic/linestring.hpp"
#include "pyinterp/geometry/geographic/multi_linestring.hpp"
#include "pyinterp/geometry/geographic/multi_point.hpp"
#include "pyinterp/geometry/geographic/multi_polygon.hpp"
#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/geographic/polygon.hpp"
#include "pyinterp/geometry/geographic/ring.hpp"
#include "pyinterp/geometry/geographic/segment.hpp"
// IWYU pragma: end_keep

namespace pyinterp::geometry::pybind {

/// @brief Enumeration to identify geometry namespace
enum class GeometryNamespace : int8_t { kGeographic, kCartesian };

namespace nb = nanobind;
using nb::literals::operator""_a;

/// @brief Helper to define a unary algorithm for multiple geometry types
/// @tparam Algorithm Algorithm functor
/// @tparam Geometries Geometry types to bind
/// @param[in] m Python module
/// @param[in] name Function name
/// @param[in] doc Documentation string
/// @param[in] alg Algorithm functor that takes a geometry
template <typename Algorithm, typename... Geometries>
inline auto define_unary_predicate(nb::module_& m, const char* name,
                                   const char* doc, Algorithm&& alg) -> void {
  // Fold expression to define binding for each geometry type
  (...,
   m.def(
       name, [alg](const Geometries& g) { return alg(g); }, "geometry"_a, doc));
}

/// @brief Helper to define a unary algorithm for multiple geometry types
/// using a strategy
/// @tparam Algorithm Algorithm functor
/// @tparam Geometries Geometry types to bind
/// @param[in] m Python module
/// @param[in] name Function name
/// @param[in] doc Documentation string
/// @param[in] alg Algorithm functor that takes a geometry
template <typename Algorithm, typename Spheroid, typename Strategy,
          typename... Geometries>
inline auto define_unary_predicate_with_strategy(nb::module_& m,
                                                 const char* name,
                                                 const char* doc,
                                                 Algorithm&& alg) -> void {
  (...,
   m.def(
       name,
       [alg](const Geometries& g, const std::optional<Spheroid>& spheroid,
             const Strategy& strategy) { return alg(g, spheroid, strategy); },
       "geometry"_a, "spheroid"_a = std::nullopt, "strategy"_a = Strategy{},
       doc));
}

/// @brief Helper to define a mutable algorithm for multiple geometry types
/// @tparam Algorithm Algorithm functor that modifies geometry in place
/// @tparam Geometries Geometry types to bind
/// @param[in] m Python module
/// @param[in] name Function name
/// @param[in] doc Documentation string
/// @param[in] alg Algorithm functor that takes a mutable geometry reference
template <typename Algorithm, typename... Geometries>
inline auto define_mutable_unary_predicate(nb::module_& m, const char* name,
                                           const char* doc, Algorithm&& alg)
    -> void {
  (..., m.def(name, [alg](Geometries& g) { alg(g); }, "geometry"_a, doc));
}

/// @brief Helper to define a binary predicate for geometry pairs
/// @tparam Predicate Binary predicate functor
/// @tparam GeometryPairs Tuple of std::pair<G1, G2> for each combination
/// @param[in] m Python module
/// @param[in] name Function name
/// @param[in] doc Documentation string
/// @param[in] pred Predicate functor that takes two geometries
template <typename Predicate, typename... GeometryPairs>
inline auto define_binary_predicate(nb::module_& m, const char* name,
                                    const char* doc, Predicate&& pred) -> void {
  // Helper to unpack std::pair and define binding
  auto define_pair = [&]<typename Pair>(Pair*) {
    using G1 = typename Pair::first_type;
    using G2 = typename Pair::second_type;
    m.def(
        name, [pred](const G1& g1, const G2& g2) { return pred(g1, g2); },
        "geometry1"_a, "geometry2"_a, doc);
  };

  // Fold expression to define binding for each pair
  (..., define_pair(static_cast<GeometryPairs*>(nullptr)));
}

/// @brief Helper to define a binary predicate for geometry pairs
/// @tparam Predicate Binary predicate functor
/// @tparam GeometryPairs Tuple of std::pair<G1, G2> for each combination
/// @param[in] m Python module
/// @param[in] name Function name
/// @param[in] doc Documentation string
/// @param[in] pred Predicate functor that takes two geometries
template <typename Predicate, typename Spheroid, typename Strategy,
          typename... GeometryPairs>
inline auto define_binary_predicate_with_strategy(nb::module_& m,
                                                  const char* name,
                                                  const char* doc,
                                                  Predicate&& pred) -> void {
  // Helper to unpack std::pair and define binding
  auto define_pair = [&]<typename Pair>(Pair*) {
    using G1 = typename Pair::first_type;
    using G2 = typename Pair::second_type;
    m.def(
        name,
        [pred](const G1& g1, const G2& g2,
               const std::optional<Spheroid>& spheroid,
               const Strategy& strategy) {
          return pred(g1, g2, spheroid, strategy);
        },
        "geometry1"_a, "geometry2"_a, "spheroid"_a = std::nullopt,
        "strategy"_a = Strategy{}, doc);
  };

  // Fold expression to define binding for each pair
  (..., define_pair(static_cast<GeometryPairs*>(nullptr)));
}

/// @brief Macro to create a list of all standard geometry types for a
/// coordinate system
/// @param NS Namespace containing the geometry types (e.g., geographic,
/// cartesian)
#define GEOMETRY_TYPES(NS)                                                \
  NS::Point, NS::Segment, NS::Box, NS::LineString, NS::Ring, NS::Polygon, \
      NS::MultiPoint, NS::MultiLineString, NS::MultiPolygon

/// @brief Macro to create common binary geometry pairs
/// @param NS Namespace containing the geometry types
#define GEOMETRY_PAIRS(NS)                                                   \
  std::pair<NS::Point, NS::Point>, std::pair<NS::Point, NS::Box>,            \
      std::pair<NS::Point, NS::Polygon>,                                     \
      std::pair<NS::Point, NS::MultiPolygon>, std::pair<NS::Box, NS::Point>, \
      std::pair<NS::Box, NS::Box>, std::pair<NS::Box, NS::Polygon>,          \
      std::pair<NS::LineString, NS::LineString>,                             \
      std::pair<NS::LineString, NS::Polygon>,                                \
      std::pair<NS::LineString, NS::Box>, std::pair<NS::Polygon, NS::Point>, \
      std::pair<NS::Polygon, NS::Box>, std::pair<NS::Polygon, NS::Polygon>,  \
      std::pair<NS::Polygon, NS::MultiPolygon>,                              \
      std::pair<NS::MultiPolygon, NS::Point>,                                \
      std::pair<NS::MultiPolygon, NS::Polygon>,                              \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>

/// @brief Macro to create common binary geometry pairs for the "crosses"
/// predicate
#define CROSSES_PAIRS(NS)                                                \
  std::pair<NS::Point, NS::LineString>, std::pair<NS::Point, NS::Ring>,  \
      std::pair<NS::Point, NS::Polygon>,                                 \
      std::pair<NS::Point, NS::MultiLineString>,                         \
      std::pair<NS::Point, NS::MultiPolygon>,                            \
      std::pair<NS::LineString, NS::Point>,                              \
      std::pair<NS::LineString, NS::LineString>,                         \
      std::pair<NS::LineString, NS::Ring>,                               \
      std::pair<NS::LineString, NS::Polygon>,                            \
      std::pair<NS::LineString, NS::MultiLineString>,                    \
      std::pair<NS::LineString, NS::MultiPolygon>,                       \
      std::pair<NS::Ring, NS::Point>, std::pair<NS::Polygon, NS::Point>, \
      std::pair<NS::MultiLineString, NS::Point>,                         \
      std::pair<NS::MultiLineString, NS::LineString>,                    \
      std::pair<NS::MultiLineString, NS::Ring>,                          \
      std::pair<NS::MultiLineString, NS::Polygon>,                       \
      std::pair<NS::MultiLineString, NS::MultiLineString>,               \
      std::pair<NS::MultiLineString, NS::MultiPolygon>,                  \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>

/// @brief Macro to create container geometry types for for_each_point
/// algorithms. These are geometries that can contain points.
/// @param NS Namespace containing the geometry types
#define CONTAINER_TYPES(NS) NS::Box, NS::Ring, NS::Polygon, NS::MultiPolygon

/// @brief Macro to create source geometry types for for_each_point algorithms
/// These are geometries that have extractable points (vertices)
/// @param NS Namespace containing the geometry types
#define SOURCE_TYPES(NS) NS::MultiPoint, NS::LineString, NS::Ring

/// @brief Helper to define for_each_point style algorithms for single source
/// Applies an algorithm to each point in a source geometry against a container
/// @tparam Algorithm Algorithm functor
/// @tparam SourceGeometry Source geometry type (MultiPoint, LineString, Ring)
/// @tparam Containers Container geometry types
/// @param[in] m Python module
/// @param[in] name Function name
/// @param[in] doc Documentation string
/// @param[in] alg Algorithm functor that takes (SourceGeometry, Container)
template <typename Algorithm, typename SourceGeometry, typename... Containers>
inline auto define_for_each_point_single_source(nanobind::module_& m,
                                                const char* name,
                                                const char* doc,
                                                const Algorithm& alg) -> void {
  // Fold expression to define binding for each container type
  (..., m.def(
            name,
            [alg](const SourceGeometry& source, const Containers& container) {
              return alg(source, container);
            },
            "source"_a, "container"_a, doc));
}

/// @brief Helper to define for_each_point algorithms with strategy support
/// @tparam Algorithm Algorithm functor
/// @tparam SourceGeometry Source geometry type (MultiPoint, LineString, Ring)
/// @tparam Spheroid Spheroid type
/// @tparam Strategy Strategy type
/// @tparam Containers Container geometry types
/// @param[in] m Python module
/// @param[in] name Function name
/// @param[in] doc Documentation string
/// @param[in] alg Algorithm functor that takes (Source, Container, Spheroid,
/// Strategy)
template <typename Algorithm, typename SourceGeometry, typename Spheroid,
          typename Strategy, typename... Containers>
inline auto define_for_each_point_single_source_with_strategy(
    nanobind::module_& m, const char* name, const char* doc,
    const Algorithm& alg) -> void {
  // Fold expression to define binding for each container type
  (..., m.def(
            name,
            [alg](const SourceGeometry& source, const Containers& container,
                  const std::optional<Spheroid>& spheroid,
                  const Strategy& strategy) {
              return alg(source, container, spheroid, strategy);
            },
            "source"_a, "target"_a, "spheroid"_a = std::nullopt,
            "strategy"_a = Strategy{}, doc));
}

}  // namespace pyinterp::geometry::pybind

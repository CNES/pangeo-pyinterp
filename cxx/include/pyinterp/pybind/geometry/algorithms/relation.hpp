// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <boost/geometry.hpp>
#include <string>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kRelationDoc = R"doc(
Computes the DE-9IM (Dimensionally Extended nine-Intersection Model) matrix
for two geometries.

The DE-9IM is a topological model that describes the spatial relationship between
two geometries using a 3x3 matrix. Each cell in the matrix describes the dimension
of the intersection between interior/boundary/exterior of the two geometries.

Args:
    geometry1: First geometry.
    geometry2: Second geometry.

Returns:
    A string representing the 9-character DE-9IM matrix pattern (e.g., "FF2F11212").
    Each character can be:
    - 'F' (false): empty set (no intersection)
    - '0': point (0-dimensional) intersection
    - '1': line (1-dimensional) intersection
    - '2': area (2-dimensional) intersection

Examples:
    >>> point = Point(4.0, 1.0)
    >>> polygon = Polygon(...)
    >>> matrix = relation(point, polygon)
    >>> # Returns string like "0FFFFF212" indicating the topological
    >>> # relationship
)doc";

/// @brief Macro for geometry pairs that support relation
/// Note: Segment, Box pairs not supported by Boost.Geometry relation
#define RELATION_PAIRS(NS)                                                    \
  std::pair<NS::Point, NS::Point>, std::pair<NS::LineString, NS::LineString>, \
      std::pair<NS::Ring, NS::Ring>, std::pair<NS::Ring, NS::Polygon>,        \
      std::pair<NS::Polygon, NS::Ring>, std::pair<NS::Polygon, NS::Polygon>

/// @brief Helper to define relation for geometry pairs
/// @tparam GeometryPairs Pairs of geometry types
/// @param[in] m Python module
/// @param[in] doc Documentation string
template <typename... GeometryPairs>
inline auto define_relation_for_pairs(nanobind::module_& m, const char* doc)
    -> void {
  auto relation_impl = [](const auto& g1, const auto& g2) -> std::string {
    nanobind::gil_scoped_release release;
    auto de9im = boost::geometry::relation(g1, g2);
    // Convert DE-9IM matrix to string representation
    return de9im.str();
  };

  (..., m.def(
            "relation",
            [relation_impl](const typename GeometryPairs::first_type& g1,
                            const typename GeometryPairs::second_type& g2) {
              return relation_impl(g1, g2);
            },
            "geometry1"_a, "geometry2"_a, doc));
}

/// @brief Initialize the relation algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_relation(nanobind::module_& m) -> void {
  if constexpr (NS == GeometryNamespace::kCartesian) {
    define_relation_for_pairs<RELATION_PAIRS(cartesian)>(m, kRelationDoc);
  } else {
    define_relation_for_pairs<RELATION_PAIRS(geographic)>(m, kRelationDoc);
  }
}

}  // namespace pyinterp::geometry::pybind

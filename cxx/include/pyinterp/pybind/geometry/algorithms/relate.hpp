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

constexpr auto kRelateDoc = R"doc(
Checks if two geometries satisfy a DE-9IM (Dimensionally Extended nine-Intersection
Model) relationship mask.

The DE-9IM mask is a 9-character pattern that specifies the required relationship
between the interior/boundary/exterior of two geometries. This function returns
true if the actual relationship matches the mask pattern.

Args:
    geometry1: First geometry.
    geometry2: Second geometry.
    mask: DE-9IM mask pattern as a 9-character string (e.g., "T*F**F***" for within).
          Each character can be:
          - 'T' (true): non-empty intersection required
          - 'F' (false): empty intersection required
          - '*': any dimension (don't care)
          - '0', '1', '2': specific dimension required

Returns:
    Boolean indicating if the geometries satisfy the mask relationship.

Examples:
    >>> point = Point(4.0, 1.0)
    >>> polygon = Polygon(...)

    Check if point is within polygon

    >>> is_within = relate(point, polygon, "T*F**F***")
    >>> # Returns True if point is inside polygon
)doc";

/// @brief Macro for geometry pairs that support relate
/// Note: Segment, Box pairs not supported by Boost.Geometry relate
#define RELATE_PAIRS(NS)                                                      \
  std::pair<NS::Point, NS::Point>, std::pair<NS::LineString, NS::LineString>, \
      std::pair<NS::Ring, NS::Ring>, std::pair<NS::Ring, NS::Polygon>,        \
      std::pair<NS::Polygon, NS::Ring>, std::pair<NS::Polygon, NS::Polygon>

/// @brief Helper to define relate for geometry pairs
/// @tparam GeometryPairs Pairs of geometry types
/// @param[in] m Python module
/// @param[in] doc Documentation string
template <typename... GeometryPairs>
inline auto define_relate_for_pairs(nanobind::module_& m, const char* doc)
    -> void {
  auto relate_impl = [](const auto& g1, const auto& g2,
                        const std::string& mask) -> bool {
    nanobind::gil_scoped_release release;
    boost::geometry::de9im::mask de9im_mask(mask);
    return boost::geometry::relate(g1, g2, de9im_mask);
  };

  (..., m.def(
            "relate",
            [relate_impl](const typename GeometryPairs::first_type& g1,
                          const typename GeometryPairs::second_type& g2,
                          const std::string& mask) {
              return relate_impl(g1, g2, mask);
            },
            "geometry1"_a, "geometry2"_a, "mask"_a, doc));
}

/// @brief Initialize the relate algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_relate(nanobind::module_& m) -> void {
  if constexpr (NS == GeometryNamespace::kCartesian) {
    define_relate_for_pairs<RELATE_PAIRS(cartesian)>(m, kRelateDoc);
  } else {
    define_relate_for_pairs<RELATE_PAIRS(geographic)>(m, kRelateDoc);
  }
}

}  // namespace pyinterp::geometry::pybind

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kIsSimpleDoc = R"doc(
Check if a geometry is simple (has no self-intersections).

For linestrings and rings, a geometry is simple if it does not intersect itself
(except possibly at endpoints). For polygons, this checks both the outer ring
and inner rings.

Note: Points and boxes are always simple.

Args:
    geometry: Geometric object to check.

Returns:
    True if the geometry is simple, false otherwise.
)doc";

/// @brief Initialize the is_simple algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_is_simple(nanobind::module_& m) -> void {
  auto is_simple_impl = [](const auto& g) -> bool {
    nanobind::gil_scoped_release release;
    return boost::geometry::is_simple(g);
  };
  if constexpr (NS == GeometryNamespace::kCartesian) {
    geometry::pybind::define_unary_predicate<decltype(is_simple_impl),
                                             GEOMETRY_TYPES(cartesian)>(
        m, "is_simple", kIsSimpleDoc, std::move(is_simple_impl));
  } else {
    geometry::pybind::define_unary_predicate<decltype(is_simple_impl),
                                             GEOMETRY_TYPES(geographic)>(
        m, "is_simple", kIsSimpleDoc, std::move(is_simple_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

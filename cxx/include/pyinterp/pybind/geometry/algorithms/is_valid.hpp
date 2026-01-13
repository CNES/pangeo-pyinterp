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

constexpr auto kIsValidDoc = R"doc(
Check if a geometry is valid according to OGC standards.

Validity rules (OGC Simple Features specification):

- Points: Always valid (unless coordinates are NaN/Inf)
- LineStrings: Must have at least 2 points
- Rings: Must be closed and have at least 4 points
- Polygons: Outer ring must be counter-clockwise, inner rings clockwise,
  rings must not cross each other or touch except at single points
- MultiPolygons: All constituent polygons must be valid

Args:
    geometry: Geometric object to check.

Returns:
    If called without return_reason:
        bool: True if the geometry is valid, false otherwise.
    If called with return_reason=True:
        tuple: (is_valid, reason) where reason is a string describing why
               the geometry is invalid (empty string if valid).
)doc";

/// @brief Define is_valid algorithm for the specified geometry types.
/// @tparam NS Geometry namespace (cartesian or geographic).
/// @tparam Geometries Geometry types to bind.
/// @param[in,out] m Python module.
template <GeometryNamespace NS, typename... Geometries>
inline auto is_valid_impl(nanobind::module_& m) -> void {
  (..., m.def(
            "is_valid",
            [](const Geometries& geometry,
               const bool return_reason) -> nanobind::object {
              bool valid;
              if (return_reason) {
                std::string reason;
                {
                  nanobind::gil_scoped_release release;
                  valid = boost::geometry::is_valid(geometry, reason);
                }
                return nanobind::make_tuple(valid, reason);
              } else {
                {
                  nanobind::gil_scoped_release release;
                  valid = boost::geometry::is_valid(geometry);
                }
                return nanobind::cast(valid);
              }
            },
            "geometry"_a, "return_reason"_a = false, kIsValidDoc));
}

/// @brief Initialize is_valid algorithm bindings for the specified geometry
/// namespace.
/// @tparam NS Geometry namespace (cartesian or geographic).
/// @param[in,out] m Nanobind module.
template <GeometryNamespace NS>
inline auto init_is_valid(nanobind::module_& m) -> void {
  if constexpr (NS == GeometryNamespace::kCartesian) {
    is_valid_impl<NS, GEOMETRY_TYPES(cartesian)>(m);
  } else {
    is_valid_impl<NS, GEOMETRY_TYPES(geographic)>(m);
  }
}

}  // namespace pyinterp::geometry::pybind

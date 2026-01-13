// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kReverseDoc = R"doc(
Reverses the order of points in a geometry.

For linestrings and rings, this reverses the order of points.
For polygons, this reverses both the exterior ring and all interior rings.
For multi-geometries, this reverses each component geometry.

This operation modifies the geometry in-place.

Args:
    geometry: Geometric object to reverse.
)doc";

/// @brief Initialize the reverse algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_reverse(nanobind::module_& m) -> void {
  auto reverse_impl = [](auto& g) -> void {
    nanobind::gil_scoped_release release;
    boost::geometry::reverse(g);
  };
  if constexpr (NS == GeometryNamespace::kCartesian) {
    geometry::pybind::define_mutable_unary_predicate<decltype(reverse_impl),
                                                     GEOMETRY_TYPES(cartesian)>(
        m, "reverse", kReverseDoc, std::move(reverse_impl));
  } else {
    geometry::pybind::define_mutable_unary_predicate<
        decltype(reverse_impl), GEOMETRY_TYPES(geographic)>(
        m, "reverse", kReverseDoc, std::move(reverse_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

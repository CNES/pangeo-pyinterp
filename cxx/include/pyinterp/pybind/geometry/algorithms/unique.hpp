// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kUniqueDoc = R"doc(
Removes consecutive duplicate points from a geometry.

This function removes points that are equal to their immediate predecessor.
For closed geometries like rings, the closing point is preserved.

This operation modifies the geometry in-place.

Args:
    geometry: Geometric object to process.
)doc";

/// @brief Initialize the unique algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_unique(nanobind::module_& m) -> void {
  auto unique_impl = [](auto& g) -> void {
    nanobind::gil_scoped_release release;
    boost::geometry::unique(g);
  };
  if constexpr (NS == GeometryNamespace::kCartesian) {
    geometry::pybind::define_mutable_unary_predicate<decltype(unique_impl),
                                                     GEOMETRY_TYPES(cartesian)>(
        m, "unique", kUniqueDoc, std::move(unique_impl));
  } else {
    geometry::pybind::define_mutable_unary_predicate<
        decltype(unique_impl), GEOMETRY_TYPES(geographic)>(
        m, "unique", kUniqueDoc, std::move(unique_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

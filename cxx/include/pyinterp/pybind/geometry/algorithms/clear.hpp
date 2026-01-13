// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kClearDoc = R"doc(
Clears a geometry, removing all points.

After calling clear, the geometry will be empty (is_empty will return True).
This operation modifies the geometry in-place.

Args:
    geometry: Geometric object to clear.
)doc";

/// @brief Initialize the clear algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_clear(nanobind::module_& m) -> void {
  auto clear_impl = [](auto& g) -> void {
    nanobind::gil_scoped_release release;
    boost::geometry::clear(g);
  };
  if constexpr (NS == GeometryNamespace::kCartesian) {
    geometry::pybind::define_mutable_unary_predicate<decltype(clear_impl),
                                                     GEOMETRY_TYPES(cartesian)>(
        m, "clear", kClearDoc, std::move(clear_impl));
  } else {
    geometry::pybind::define_mutable_unary_predicate<
        decltype(clear_impl), GEOMETRY_TYPES(geographic)>(
        m, "clear", kClearDoc, std::move(clear_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

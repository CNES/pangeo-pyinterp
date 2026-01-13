// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kNumInteriorRingsDoc = R"doc(
Returns the number of interior rings (holes) in a polygon.

For polygons, returns the number of holes. For other geometries, returns 0.

Args:
    geometry: Geometric object to count interior rings in.

Returns:
    The number of interior rings.
)doc";

/// @brief Initialize the num_interior_rings algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_num_interior_rings(nanobind::module_& m) -> void {
  auto num_interior_rings_impl = [](const auto& g) -> std::size_t {
    nanobind::gil_scoped_release release;
    return boost::geometry::num_interior_rings(g);
  };
  if constexpr (NS == GeometryNamespace::kCartesian) {
    geometry::pybind::define_unary_predicate<decltype(num_interior_rings_impl),
                                             GEOMETRY_TYPES(cartesian)>(
        m, "num_interior_rings", kNumInteriorRingsDoc,
        std::move(num_interior_rings_impl));
  } else {
    geometry::pybind::define_unary_predicate<decltype(num_interior_rings_impl),
                                             GEOMETRY_TYPES(geographic)>(
        m, "num_interior_rings", kNumInteriorRingsDoc,
        std::move(num_interior_rings_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

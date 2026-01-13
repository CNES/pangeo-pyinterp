// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kNumGeometriesDoc = R"doc(
Returns the number of geometries in a multi-geometry or 1 for single geometries.

For multi-geometries (MultiPoint, MultiLineString, MultiPolygon), returns the
number of component geometries. For single geometries, returns 1.

Args:
    geometry: Geometric object to count geometries in.

Returns:
    The number of geometries.
)doc";

/// @brief Initialize the num_geometries algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_num_geometries(nanobind::module_& m) -> void {
  auto num_geometries_impl = [](const auto& g) -> std::size_t {
    nanobind::gil_scoped_release release;
    return boost::geometry::num_geometries(g);
  };
  if constexpr (NS == GeometryNamespace::kCartesian) {
    geometry::pybind::define_unary_predicate<decltype(num_geometries_impl),
                                             GEOMETRY_TYPES(cartesian)>(
        m, "num_geometries", kNumGeometriesDoc, std::move(num_geometries_impl));
  } else {
    geometry::pybind::define_unary_predicate<decltype(num_geometries_impl),
                                             GEOMETRY_TYPES(geographic)>(
        m, "num_geometries", kNumGeometriesDoc, std::move(num_geometries_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

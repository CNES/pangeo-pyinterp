// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kIsEmptyDoc = R"doc(
Check if a geometry is empty (contains no points).

Args:
    geometry: Geometric object to check.

Returns:
    True if the geometry is empty, false otherwise.
)doc";

/// @brief Initialize the is_empty algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_is_empty(nanobind::module_& m) -> void {
  auto is_empty_impl = [](const auto& g) -> bool {
    nanobind::gil_scoped_release release;
    return boost::geometry::is_empty(g);
  };
  if constexpr (NS == GeometryNamespace::kCartesian) {
    geometry::pybind::define_unary_predicate<decltype(is_empty_impl),
                                             GEOMETRY_TYPES(cartesian)>(
        m, "is_empty", kIsEmptyDoc, std::move(is_empty_impl));
  } else {
    geometry::pybind::define_unary_predicate<decltype(is_empty_impl),
                                             GEOMETRY_TYPES(geographic)>(
        m, "is_empty", kIsEmptyDoc, std::move(is_empty_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

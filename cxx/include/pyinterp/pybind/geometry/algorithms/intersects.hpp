// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kIntersectsDoc = R"doc(
Checks if two geometries intersect (have at least one point in common).

Two geometries intersect if they share at least one point. This is the
opposite of the disjoint predicate.

Args:
    geometry1: First geometric object.
    geometry2: Second geometric object.

Returns:
    True if the geometries intersect, False otherwise.
)doc";

/// @brief Initialize the intersects algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_intersects(nanobind::module_& m) -> void {
  auto intersects_impl = [](const auto& geometry1,
                            const auto& geometry2) -> bool {
    nanobind::gil_scoped_release release;
    return boost::geometry::intersects(geometry1, geometry2);
  };

  if constexpr (NS == GeometryNamespace::kCartesian) {
    geometry::pybind::define_binary_predicate<decltype(intersects_impl),
                                              GEOMETRY_PAIRS(cartesian)>(
        m, "intersects", kIntersectsDoc, std::move(intersects_impl));
  } else {
    geometry::pybind::define_binary_predicate<decltype(intersects_impl),
                                              GEOMETRY_PAIRS(geographic)>(
        m, "intersects", kIntersectsDoc, std::move(intersects_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

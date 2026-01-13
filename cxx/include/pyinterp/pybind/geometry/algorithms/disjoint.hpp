// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kDisjointDoc = R"doc(
Checks if two geometries are disjoint (do not intersect).

Two geometries are disjoint if they have no point in common.

Args:
    geometry1: First geometric object.
    geometry2: Second geometric object.

Returns:
    True if the geometries are disjoint, False otherwise.
)doc";

/// @brief Initialize the disjoint algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_disjoint(nanobind::module_& m) -> void {
  auto disjoint_impl = [](const auto& geometry1,
                          const auto& geometry2) -> bool {
    nanobind::gil_scoped_release release;
    return boost::geometry::disjoint(geometry1, geometry2);
  };

  if constexpr (NS == GeometryNamespace::kCartesian) {
    geometry::pybind::define_binary_predicate<decltype(disjoint_impl),
                                              GEOMETRY_PAIRS(cartesian)>(
        m, "disjoint", kDisjointDoc, std::move(disjoint_impl));
  } else {
    geometry::pybind::define_binary_predicate<decltype(disjoint_impl),
                                              GEOMETRY_PAIRS(geographic)>(
        m, "disjoint", kDisjointDoc, std::move(disjoint_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kCorrectDoc = R"doc(
Corrects a geometry to make it valid according to OGC rules.

This function applies several corrections:

- Ensures rings are closed (first point equals last point)
- Ensures correct orientation (exterior rings counter-clockwise, holes
  clockwise)
- Orders multi-geometries consistently

This operation modifies the geometry in-place.

Args:
    geometry: Geometric object to correct.
)doc";

/// @brief Initialize the correct algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_correct(nanobind::module_& m) -> void {
  auto correct_impl = [](auto& g) -> void {
    nanobind::gil_scoped_release release;
    boost::geometry::correct(g);
  };
  if constexpr (NS == GeometryNamespace::kCartesian) {
    geometry::pybind::define_mutable_unary_predicate<decltype(correct_impl),
                                                     GEOMETRY_TYPES(cartesian)>(
        m, "correct", kCorrectDoc, std::move(correct_impl));
  } else {
    geometry::pybind::define_mutable_unary_predicate<
        decltype(correct_impl), GEOMETRY_TYPES(geographic)>(
        m, "correct", kCorrectDoc, std::move(correct_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kOverlapsDoc = R"doc(
Checks if two geometries overlap.

Two geometries overlap if they have some but not all points in common,
they have the same dimension, and the intersection of their interiors
is not empty.

Args:
    geometry1: First geometric object.
    geometry2: Second geometric object.

Returns:
    True if the geometries overlap, False otherwise.
)doc";

/// @brief Macro to create geometry pairs for the overlaps predicate
/// overlaps requires same-dimension geometries
#define OVERLAPS_PAIRS(NS)                                 \
  std::pair<NS::LineString, NS::LineString>,               \
      std::pair<NS::LineString, NS::MultiLineString>,      \
      std::pair<NS::Polygon, NS::Polygon>,                 \
      std::pair<NS::Polygon, NS::MultiPolygon>,            \
      std::pair<NS::MultiLineString, NS::LineString>,      \
      std::pair<NS::MultiLineString, NS::MultiLineString>, \
      std::pair<NS::MultiPolygon, NS::Polygon>,            \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>

/// @brief Initialize the overlaps algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_overlaps(nanobind::module_& m) -> void {
  auto overlaps_impl = [](const auto& geometry1,
                          const auto& geometry2) -> bool {
    nanobind::gil_scoped_release release;
    return boost::geometry::overlaps(geometry1, geometry2);
  };

  if constexpr (NS == GeometryNamespace::kCartesian) {
    geometry::pybind::define_binary_predicate<decltype(overlaps_impl),
                                              OVERLAPS_PAIRS(cartesian)>(
        m, "overlaps", kOverlapsDoc, std::move(overlaps_impl));
  } else {
    geometry::pybind::define_binary_predicate<decltype(overlaps_impl),
                                              OVERLAPS_PAIRS(geographic)>(
        m, "overlaps", kOverlapsDoc, std::move(overlaps_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

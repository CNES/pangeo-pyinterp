// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kEqualsDoc = R"doc(
Checks if two geometries are spatially equal.

Two geometries are equal if they represent the same set of points,
regardless of the order of points or internal representation.

Args:
    geometry1: First geometric object.
    geometry2: Second geometric object.

Returns:
    True if the geometries are equal, False otherwise.
)doc";

/// @brief Macro to create geometry pairs for the equals predicate
/// equals works only for same-type comparisons in many cases
#define EQUALS_PAIRS(NS)                                                      \
  std::pair<NS::Point, NS::Point>, std::pair<NS::Segment, NS::Segment>,       \
      std::pair<NS::Box, NS::Box>, std::pair<NS::LineString, NS::LineString>, \
      std::pair<NS::Ring, NS::Ring>, std::pair<NS::Polygon, NS::Polygon>,     \
      std::pair<NS::MultiPoint, NS::MultiPoint>,                              \
      std::pair<NS::MultiLineString, NS::MultiLineString>,                    \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>

/// @brief Initialize the equals algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_equals(nanobind::module_& m) -> void {
  auto equals_impl = [](const auto& geometry1, const auto& geometry2) -> bool {
    nanobind::gil_scoped_release release;
    return boost::geometry::equals(geometry1, geometry2);
  };

  if constexpr (NS == GeometryNamespace::kCartesian) {
    geometry::pybind::define_binary_predicate<decltype(equals_impl),
                                              EQUALS_PAIRS(cartesian)>(
        m, "equals", kEqualsDoc, std::move(equals_impl));
  } else {
    geometry::pybind::define_binary_predicate<decltype(equals_impl),
                                              EQUALS_PAIRS(geographic)>(
        m, "equals", kEqualsDoc, std::move(equals_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

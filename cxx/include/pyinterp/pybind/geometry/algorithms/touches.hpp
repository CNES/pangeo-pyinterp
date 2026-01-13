// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kTouchesDoc = R"doc(
Checks if two geometries touch (have at least one boundary point in common,
but no interior points).

Two geometries touch if they have at least one point in common, but their
interiors do not intersect.

Args:
    geometry1: First geometric object.
    geometry2: Second geometric object.

Returns:
    True if the geometries touch, False otherwise.
)doc";

/// @brief Macro to create geometry pairs for the touches predicate
/// touches has restrictions on Box combinations
#define TOUCHES_PAIRS(NS)                                                    \
  std::pair<NS::Point, NS::Point>, std::pair<NS::Point, NS::LineString>,     \
      std::pair<NS::Point, NS::Ring>, std::pair<NS::Point, NS::Polygon>,     \
      std::pair<NS::Point, NS::MultiLineString>,                             \
      std::pair<NS::Point, NS::MultiPolygon>,                                \
      std::pair<NS::LineString, NS::LineString>,                             \
      std::pair<NS::LineString, NS::Ring>,                                   \
      std::pair<NS::LineString, NS::Polygon>,                                \
      std::pair<NS::LineString, NS::MultiLineString>,                        \
      std::pair<NS::LineString, NS::MultiPolygon>,                           \
      std::pair<NS::Ring, NS::LineString>, std::pair<NS::Ring, NS::Ring>,    \
      std::pair<NS::Ring, NS::Polygon>,                                      \
      std::pair<NS::Ring, NS::MultiLineString>,                              \
      std::pair<NS::Ring, NS::MultiPolygon>,                                 \
      std::pair<NS::Polygon, NS::Point>,                                     \
      std::pair<NS::Polygon, NS::LineString>,                                \
      std::pair<NS::Polygon, NS::Ring>, std::pair<NS::Polygon, NS::Polygon>, \
      std::pair<NS::Polygon, NS::MultiLineString>,                           \
      std::pair<NS::Polygon, NS::MultiPolygon>,                              \
      std::pair<NS::MultiLineString, NS::Point>,                             \
      std::pair<NS::MultiLineString, NS::LineString>,                        \
      std::pair<NS::MultiLineString, NS::Ring>,                              \
      std::pair<NS::MultiLineString, NS::Polygon>,                           \
      std::pair<NS::MultiLineString, NS::MultiLineString>,                   \
      std::pair<NS::MultiLineString, NS::MultiPolygon>,                      \
      std::pair<NS::MultiPolygon, NS::Point>,                                \
      std::pair<NS::MultiPolygon, NS::LineString>,                           \
      std::pair<NS::MultiPolygon, NS::Ring>,                                 \
      std::pair<NS::MultiPolygon, NS::Polygon>,                              \
      std::pair<NS::MultiPolygon, NS::MultiLineString>,                      \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>

/// @brief Initialize the touches algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_touches(nanobind::module_& m) -> void {
  auto touches_impl = [](const auto& geometry1, const auto& geometry2) -> bool {
    nanobind::gil_scoped_release release;
    return boost::geometry::touches(geometry1, geometry2);
  };

  if constexpr (NS == GeometryNamespace::kCartesian) {
    geometry::pybind::define_binary_predicate<decltype(touches_impl),
                                              TOUCHES_PAIRS(cartesian)>(
        m, "touches", kTouchesDoc, std::move(touches_impl));
  } else {
    geometry::pybind::define_binary_predicate<decltype(touches_impl),
                                              TOUCHES_PAIRS(geographic)>(
        m, "touches", kTouchesDoc, std::move(touches_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

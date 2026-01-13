// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kWithinDoc = R"doc(
Checks if the first geometry is completely within the second geometry.

A geometry is within another if all its points are in the interior or
boundary of the second geometry.

Args:
    geometry1: First geometric object (to be tested).
    geometry2: Second geometric object (container).

Returns:
    True if geometry1 is within geometry2, False otherwise.
)doc";

/// @brief Macro to create geometry pairs for the within predicate
/// within has similar restrictions as covered_by
#define WITHIN_PAIRS(NS)                                                       \
  std::pair<NS::Point, NS::Point>, std::pair<NS::Point, NS::Segment>,          \
      std::pair<NS::Point, NS::Box>, std::pair<NS::Point, NS::LineString>,     \
      std::pair<NS::Point, NS::Ring>, std::pair<NS::Point, NS::Polygon>,       \
      std::pair<NS::Point, NS::MultiPoint>,                                    \
      std::pair<NS::Point, NS::MultiLineString>,                               \
      std::pair<NS::Point, NS::MultiPolygon>, std::pair<NS::Box, NS::Box>,     \
      std::pair<NS::LineString, NS::LineString>,                               \
      std::pair<NS::LineString, NS::Ring>,                                     \
      std::pair<NS::LineString, NS::Polygon>,                                  \
      std::pair<NS::LineString, NS::MultiLineString>,                          \
      std::pair<NS::LineString, NS::MultiPolygon>,                             \
      std::pair<NS::Ring, NS::Ring>, std::pair<NS::Ring, NS::Polygon>,         \
      std::pair<NS::Ring, NS::MultiPolygon>, std::pair<NS::Polygon, NS::Ring>, \
      std::pair<NS::Polygon, NS::Polygon>,                                     \
      std::pair<NS::Polygon, NS::MultiPolygon>,                                \
      std::pair<NS::MultiLineString, NS::LineString>,                          \
      std::pair<NS::MultiLineString, NS::Ring>,                                \
      std::pair<NS::MultiLineString, NS::Polygon>,                             \
      std::pair<NS::MultiLineString, NS::MultiLineString>,                     \
      std::pair<NS::MultiLineString, NS::MultiPolygon>,                        \
      std::pair<NS::MultiPolygon, NS::Ring>,                                   \
      std::pair<NS::MultiPolygon, NS::Polygon>,                                \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>

/// @brief Initialize the within algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_within(nanobind::module_& m) -> void {
  auto within_impl = [](const auto& geometry1, const auto& geometry2) -> bool {
    nanobind::gil_scoped_release release;
    return boost::geometry::within(geometry1, geometry2);
  };

  if constexpr (NS == GeometryNamespace::kCartesian) {
    geometry::pybind::define_binary_predicate<decltype(within_impl),
                                              WITHIN_PAIRS(cartesian)>(
        m, "within", kWithinDoc, std::move(within_impl));
  } else {
    geometry::pybind::define_binary_predicate<decltype(within_impl),
                                              WITHIN_PAIRS(geographic)>(
        m, "within", kWithinDoc, std::move(within_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

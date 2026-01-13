// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "pyinterp/geometry/geographic/algorithms/intersection.hpp"
#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kIntersectionDoc = R"doc(
Computes the geometric intersection between two geometries.

The intersection operation returns the part of the geometries that overlap.
The result is returned as a vector of polygons, linestrings, or points,
depending on the input geometry types and their spatial relationship.

Args:
    geometry1: First geometry.
    geometry2: Second geometry.
    spheroid: Optional spheroid for geodetic calculations.
    strategy: Calculation strategy.

Returns:
    A list of geometry objects representing the intersection.
)doc";

// Macro to create geometry pairs for the intersection algorithm returning
// Polygons
#define INTERSECTION_POLYGON_PAIRS(NS)                                       \
  std::pair<NS::Ring, NS::Ring>, std::pair<NS::Ring, NS::Polygon>,           \
      std::pair<NS::Polygon, NS::Ring>, std::pair<NS::Polygon, NS::Polygon>, \
      std::pair<NS::MultiPolygon, NS::Ring>,                                 \
      std::pair<NS::MultiPolygon, NS::Polygon>,                              \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>,                         \
      std::pair<NS::Ring, NS::MultiPolygon>,                                 \
      std::pair<NS::Polygon, NS::MultiPolygon>

// Macro to create geometry pairs for the intersection algorithm returning
// LineStrings
#define INTERSECTION_LINESTRING_POLYGON_PAIRS(NS)                              \
  std::pair<NS::LineString, NS::Ring>, std::pair<NS::LineString, NS::Polygon>, \
      std::pair<NS::LineString, NS::MultiPolygon>

// Macro to create geometry pairs for the intersection algorithm returning
// Points
#define INTERSECTION_LINESTRING_PAIRS(NS) \
  std::pair<NS::LineString, NS::LineString>

auto init_intersection(nb::module_& m) -> void {
  auto intersection_polygon_impl = [](const auto& geometry1,
                                      const auto& geometry2,
                                      const std::optional<Spheroid>& spheroid,
                                      StrategyMethod strategy) -> auto {
    nb::gil_scoped_release release;
    return intersection_polygon(geometry1, geometry2, spheroid, strategy);
  };
  auto intersection_linestring_impl =
      [](const auto& geometry1, const auto& geometry2,
         const std::optional<Spheroid>& spheroid,
         StrategyMethod strategy) -> auto {
    nb::gil_scoped_release release;
    return intersection_linestring(geometry1, geometry2, spheroid, strategy);
  };
  auto intersection_point_impl = [](const auto& geometry1,
                                    const auto& geometry2,
                                    const std::optional<Spheroid>& spheroid,
                                    StrategyMethod strategy) -> auto {
    nb::gil_scoped_release release;
    return intersection_point(geometry1, geometry2, spheroid, strategy);
  };

  geometry::pybind::define_binary_predicate_with_strategy<
      decltype(intersection_polygon_impl), Spheroid, StrategyMethod,
      INTERSECTION_POLYGON_PAIRS(geographic)>(
      m, "intersection", kIntersectionDoc,
      std::move(intersection_polygon_impl));
  geometry::pybind::define_binary_predicate_with_strategy<
      decltype(intersection_linestring_impl), Spheroid, StrategyMethod,
      INTERSECTION_LINESTRING_POLYGON_PAIRS(geographic)>(
      m, "intersection", kIntersectionDoc,
      std::move(intersection_linestring_impl));
  geometry::pybind::define_binary_predicate_with_strategy<
      decltype(intersection_point_impl), Spheroid, StrategyMethod,
      INTERSECTION_LINESTRING_PAIRS(geographic)>(
      m, "intersection", kIntersectionDoc, std::move(intersection_point_impl));
}

}  // namespace pyinterp::geometry::geographic::pybind

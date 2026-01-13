// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "pyinterp/geometry/geographic/algorithms/union.hpp"
#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kUnionDoc = R"doc(
Computes the geometric union of two geometries.

Args:
    geometry1: First geometry.
    geometry2: Second geometry.
    spheroid: Optional spheroid for geodetic calculations. If not provided, uses
        WGS84 ellipsoid.
    strategy: Calculation strategy.

Returns:
    A list of geometry objects representing the union.
)doc";

// Macro to create geometry pairs for the union algorithm returning Polygons
#define UNION_POLYGON_PAIRS(NS)                                              \
  std::pair<NS::Ring, NS::Ring>, std::pair<NS::Ring, NS::Polygon>,           \
      std::pair<NS::Polygon, NS::Ring>, std::pair<NS::Polygon, NS::Polygon>, \
      std::pair<NS::MultiPolygon, NS::Ring>,                                 \
      std::pair<NS::MultiPolygon, NS::Polygon>,                              \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>,                         \
      std::pair<NS::Ring, NS::MultiPolygon>,                                 \
      std::pair<NS::Polygon, NS::MultiPolygon>

// Macro to create geometry pairs for the union algorithm returning LineStrings
#define UNION_LINESTRING_PAIRS(NS)           \
  std::pair<NS::LineString, NS::LineString>, \
      std::pair<NS::MultiLineString, NS::MultiLineString>

auto init_union(nb::module_& m) -> void {
  auto polygon_union_impl = [](const auto& geometry1, const auto& geometry2,
                               const std::optional<Spheroid>& spheroid,
                               StrategyMethod strategy) -> auto {
    nb::gil_scoped_release release;
    return union_polygon(geometry1, geometry2, spheroid, strategy);
  };
  auto linestring_union_impl = [](const auto& geometry1, const auto& geometry2,
                                  const std::optional<Spheroid>& spheroid,
                                  StrategyMethod strategy) -> auto {
    nb::gil_scoped_release release;
    return union_linestring(geometry1, geometry2, spheroid, strategy);
  };

  geometry::pybind::define_binary_predicate_with_strategy<
      decltype(polygon_union_impl), Spheroid, StrategyMethod,
      UNION_POLYGON_PAIRS(geographic)>(m, "union", kUnionDoc,
                                       std::move(polygon_union_impl));
  geometry::pybind::define_binary_predicate_with_strategy<
      decltype(linestring_union_impl), Spheroid, StrategyMethod,
      UNION_LINESTRING_PAIRS(geographic)>(m, "union", kUnionDoc,
                                          std::move(linestring_union_impl));
}

}  // namespace pyinterp::geometry::geographic::pybind

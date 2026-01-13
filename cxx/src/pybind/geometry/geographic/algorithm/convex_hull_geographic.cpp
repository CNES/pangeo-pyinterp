// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/geometry/geographic/algorithms/convex_hull.hpp"
#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/geographic/polygon.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"
#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;
using pyinterp::geometry::pybind::GeometryNamespace;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kConvexHullDoc = R"doc(
Calculates the convex hull of a geometry.

The convex hull is the smallest convex geometry that contains all points
of the input geometry. It is always returned as a Polygon.

Args:
    geometry: Geometric object.
    spheroid: Optional spheroid for geodetic calculations. If not provided, uses
        WGS84 ellipsoid.
    strategy: Calculation strategy.

Returns:
    A Polygon representing the convex hull.
)doc";

auto init_convex_hull(nb::module_& m) -> void {
  auto convex_hull_impl = [](const auto& geometry,
                             const std::optional<Spheroid>& spheroid,
                             StrategyMethod strategy) -> Polygon {
    nb::gil_scoped_release release;
    return convex_hull(geometry, spheroid, strategy);
  };

  geometry::pybind::define_unary_predicate_with_strategy<
      decltype(convex_hull_impl), Spheroid, StrategyMethod,
      GEOMETRY_TYPES(geographic)>(m, "convex_hull", kConvexHullDoc,
                                  std::move(convex_hull_impl));
}

}  // namespace pyinterp::geometry::geographic::pybind

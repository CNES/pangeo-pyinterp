// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "pyinterp/geometry/geographic/algorithms/densify.hpp"
#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kDensifyDoc = R"doc(
Densifies a geometry by adding points along segments.

Args:
    geometry: Geometric object to densify.
    max_distance: Maximum allowed distance between consecutive points.
    spheroid: Optional spheroid for geodetic calculations.
    strategy: Calculation strategy.

Returns:
    Densified version of the input geometry (same type).
)doc";

template <typename... Geometries>
inline auto define_densify(nb::module_& m, const char* doc) -> void {
  auto densify_impl = [](const auto& g, double max_distance,
                         const std::optional<Spheroid>& wgs,
                         StrategyMethod strategy) -> auto {
    using GeometryType = std::decay_t<decltype(g)>;
    nb::gil_scoped_release release;
    return densify<GeometryType>(g, max_distance, wgs, strategy);
  };

  (..., m.def(
            "densify",
            [densify_impl](const Geometries& g, double max_distance,
                           const std::optional<Spheroid>& spheroid,
                           StrategyMethod strategy) -> auto {
              return densify_impl(g, max_distance, spheroid, strategy);
            },
            "geometry"_a, "max_distance"_a, "spheroid"_a = std::nullopt,
            "strategy"_a = StrategyMethod{}, doc));
}

auto init_densify(nb::module_& m) -> void {
  define_densify<LineString, Ring, Polygon, MultiLineString, MultiPolygon>(
      m, kDensifyDoc);
}

}  // namespace pyinterp::geometry::geographic::pybind

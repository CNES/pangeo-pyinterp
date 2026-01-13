// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "pyinterp/geometry/geographic/algorithms/simplify.hpp"
#include "pyinterp/geometry/geographic/linestring.hpp"
#include "pyinterp/geometry/geographic/multi_linestring.hpp"
#include "pyinterp/geometry/geographic/multi_polygon.hpp"
#include "pyinterp/geometry/geographic/polygon.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kSimplifyDoc = R"doc(
Simplifies a geometry by removing points based on distance tolerance.

Args:
    geometry: Geometric object to simplify.
    max_distance: Maximum allowed distance for simplification.
    spheroid: Optional spheroid for geodetic calculations. If not provided, uses
        WGS84 ellipsoid.
    strategy: Calculation strategy.

Returns:
    Simplified version of the input geometry (same type).
)doc";

template <typename... Geometries>
inline auto define_simplify(nb::module_& m, const char* doc) -> void {
  auto simplify_impl = [](const auto& g, double max_distance,
                          const std::optional<Spheroid>& wgs,
                          StrategyMethod strategy) -> auto {
    using GeometryType = std::decay_t<decltype(g)>;
    nb::gil_scoped_release release;
    return simplify<GeometryType>(g, max_distance, wgs, strategy);
  };

  (..., m.def(
            "simplify",
            [simplify_impl](const Geometries& g, double max_distance,
                            const std::optional<Spheroid>& spheroid,
                            StrategyMethod strategy) -> auto {
              return simplify_impl(g, max_distance, spheroid, strategy);
            },
            "geometry"_a, "max_distance"_a, "spheroid"_a = std::nullopt,
            "strategy"_a = StrategyMethod{}, doc));
}

auto init_simplify(nb::module_& m) -> void {
  define_simplify<LineString, Ring, Polygon, MultiLineString, MultiPolygon>(
      m, kSimplifyDoc);
}

}  // namespace pyinterp::geometry::geographic::pybind

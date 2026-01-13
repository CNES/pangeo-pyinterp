// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "pyinterp/geometry/geographic/algorithms/line_interpolate.hpp"
#include "pyinterp/geometry/geographic/linestring.hpp"
#include "pyinterp/geometry/geographic/segment.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kLineInterpolateDoc = R"doc(
Returns a point at the specified distance along a linestring.

The function interpolates a point at the given distance from the start
of the linestring using geodetic calculations on the spheroid.

Args:
    geometry: LineString or Segment to interpolate along.
    distance: Distance from the start of the linestring (in meters).
    spheroid: Optional spheroid for geodetic calculations.
    strategy: Calculation strategy.

Returns:
    A Point at the specified distance along the linestring.
)doc";

// Define line_interpolate for multiple geometry types
template <typename... Geometries>
inline auto define_line_interpolate(nb::module_& m, const char* doc) -> void {
  auto line_interpolate_impl = [](const auto& g, double distance,
                                  const std::optional<Spheroid>& wgs,
                                  StrategyMethod strategy) -> auto {
    using GeometryType = std::decay_t<decltype(g)>;
    nb::gil_scoped_release release;
    return line_interpolate<GeometryType>(g, distance, wgs, strategy);
  };

  (..., m.def(
            "line_interpolate",
            [line_interpolate_impl](const Geometries& g, double distance,
                                    const std::optional<Spheroid>& spheroid,
                                    StrategyMethod strategy) -> auto {
              return line_interpolate_impl(g, distance, spheroid, strategy);
            },
            "geometry"_a, "distance"_a, "spheroid"_a = std::nullopt,
            "strategy"_a = StrategyMethod{}, doc));
}

auto init_line_interpolate(nb::module_& m) -> void {
  define_line_interpolate<LineString, Segment>(m, kLineInterpolateDoc);
}

}  // namespace pyinterp::geometry::geographic::pybind

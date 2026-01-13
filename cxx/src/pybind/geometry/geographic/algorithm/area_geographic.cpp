// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "pyinterp/geometry/geographic/algorithms/area.hpp"
#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kAreaDoc = R"doc(
Calculate the area of a geometric object.

Args:
    geometry: Geometric object (Box, Ring, Polygon, or MultiPolygon).
    spheroid: Optional spheroid for geodetic calculations. If not provided, uses
        WGS84 ellipsoid.
    strategy: Calculation strategy.

Returns:
    Area in square meters.

Note:
    For Point, Segment, LineString, MultiPoint, and MultiLineString,
    the area is always 0.0.
)doc";

auto init_area(nb::module_& m) -> void {
  auto area_impl = [](const auto& geometry, const std::optional<Spheroid>& wgs,
                      StrategyMethod strategy) -> double {
    using GeometryType = std::decay_t<decltype(geometry)>;

    if constexpr (std::is_same_v<GeometryType, Point> ||
                  std::is_same_v<GeometryType, Segment> ||
                  std::is_same_v<GeometryType, LineString> ||
                  std::is_same_v<GeometryType, MultiPoint> ||
                  std::is_same_v<GeometryType, MultiLineString>) {
      return 0.0;  // 0D/1D geometries
    } else {
      nb::gil_scoped_release release;
      return area(geometry, wgs, strategy);
    }
  };

  geometry::pybind::define_unary_predicate_with_strategy<
      decltype(area_impl), Spheroid, StrategyMethod,
      GEOMETRY_TYPES(geographic)>(m, "area", kAreaDoc, std::move(area_impl));
}

}  // namespace pyinterp::geometry::geographic::pybind

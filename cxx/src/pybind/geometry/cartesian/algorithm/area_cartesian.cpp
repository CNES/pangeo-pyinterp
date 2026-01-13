// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;

namespace pyinterp::geometry::cartesian::pybind {

constexpr auto kAreaDoc = R"doc(
Calculate the area of a geometric object in Cartesian coordinates.

The area is calculated on a flat 2D plane using standard Euclidean geometry.

Args:
    geometry: Geometric object (Box, Ring, Polygon, or MultiPolygon).

Returns:
    Area in square units.

Note:
    For Point, Segment, LineString, MultiPoint, and MultiLineString,
    the area is always 0.0.
)doc";

auto init_area(nb::module_& m) -> void {
  auto area_impl = [](const auto& geometry) -> double {
    using GeometryType = std::decay_t<decltype(geometry)>;

    if constexpr (std::is_same_v<GeometryType, Point> ||
                  std::is_same_v<GeometryType, Segment> ||
                  std::is_same_v<GeometryType, LineString> ||
                  std::is_same_v<GeometryType, MultiPoint> ||
                  std::is_same_v<GeometryType, MultiLineString>) {
      return 0.0;  // 0D/1D geometries
    } else {
      nb::gil_scoped_release release;
      return boost::geometry::area(geometry);
    }
  };

  geometry::pybind::define_unary_predicate<decltype(area_impl),
                                           GEOMETRY_TYPES(cartesian)>(
      m, "area", kAreaDoc, std::move(area_impl));
}

}  // namespace pyinterp::geometry::cartesian::pybind

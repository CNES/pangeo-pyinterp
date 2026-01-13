// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;

using nb::literals::operator""_a;
using pyinterp::geometry::pybind::GeometryNamespace;

namespace pyinterp::geometry::cartesian::pybind {

constexpr auto kLineInterpolateDoc = R"doc(
Returns a point at the specified distance along a linestring.

The function interpolates a point at the given distance from the start
of the linestring. If the distance exceeds the linestring length, the
endpoint is returned.

Args:
    geometry: LineString or Segment to interpolate along.
    distance: Distance from the start of the linestring.

Returns:
    A Point at the specified distance along the linestring.
)doc";

// Macro for geometries that support line_interpolate line_interpolate works on
// linear geometries
#define LINE_INTERPOLATE_TYPES(NS) NS::LineString, NS::Segment

// Helper to define line_interpolate for multiple geometry types
template <typename Point, typename... Geometries>
inline auto define_line_interpolate_for_geometries(nanobind::module_& m,
                                                   const char* doc) -> void {
  auto line_interpolate_impl = [](const auto& g, double distance) -> Point {
    nanobind::gil_scoped_release release;
    Point result;
    boost::geometry::line_interpolate(g, distance, result);
    return result;
  };
  (...,
   m.def(
       "line_interpolate",
       [line_interpolate_impl](const Geometries& g, double distance) -> auto {
         return line_interpolate_impl(g, distance);
       },
       "geometry"_a, "distance"_a, doc));
}

auto init_line_interpolate(nanobind::module_& m) -> void {
  define_line_interpolate_for_geometries<cartesian::Point,
                                         LINE_INTERPOLATE_TYPES(cartesian)>(
      m, kLineInterpolateDoc);
}

}  // namespace pyinterp::geometry::cartesian::pybind

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/geometry/cartesian/multi_linestring.hpp"
#include "pyinterp/geometry/cartesian/multi_polygon.hpp"
#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::cartesian::pybind {

constexpr auto kClosestPointsDoc = R"doc(
Calculate the closest points between two geometries.

The closest points are the pair of points, one on each geometry, that are
closest to each other.

Args:
    geometry1: First geometry.
    geometry2: Second geometry.

Returns:
    Closest points as a Segment.
)doc";

// Define all valid pairs of geometry types for closest points calculation
#define PAIRS(NS)                                          \
  std::pair<NS::LineString, NS::LineString>,               \
      std::pair<NS::LineString, NS::MultiLineString>,      \
      std::pair<NS::LineString, NS::MultiPoint>,           \
      std::pair<NS::LineString, NS::MultiPolygon>,         \
      std::pair<NS::LineString, NS::Polygon>,              \
      std::pair<NS::MultiPoint, NS::LineString>,           \
      std::pair<NS::MultiPoint, NS::MultiLineString>,      \
      std::pair<NS::MultiPoint, NS::MultiPoint>,           \
      std::pair<NS::MultiPoint, NS::MultiPolygon>,         \
      std::pair<NS::MultiPoint, NS::Polygon>,              \
      std::pair<NS::MultiLineString, NS::LineString>,      \
      std::pair<NS::MultiLineString, NS::MultiLineString>, \
      std::pair<NS::MultiLineString, NS::MultiPoint>,      \
      std::pair<NS::MultiLineString, NS::MultiPolygon>,    \
      std::pair<NS::MultiLineString, NS::Polygon>,         \
      std::pair<NS::MultiPolygon, NS::LineString>,         \
      std::pair<NS::MultiPolygon, NS::MultiLineString>,    \
      std::pair<NS::MultiPolygon, NS::MultiPoint>,         \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>,       \
      std::pair<NS::MultiPolygon, NS::Polygon>,            \
      std::pair<NS::Polygon, NS::LineString>,              \
      std::pair<NS::Polygon, NS::MultiLineString>,         \
      std::pair<NS::Polygon, NS::MultiPoint>,              \
      std::pair<NS::Polygon, NS::MultiPolygon>,            \
      std::pair<NS::Polygon, NS::Polygon>

auto init_closest_points(nb::module_& m) -> void {
  auto closest_points_impl = [](const auto& geometry1,
                                const auto& geometry2) -> Segment {
    nb::gil_scoped_release release;
    Segment segment;
    boost::geometry::closest_points(geometry1, geometry2, segment);
    return segment;
  };

  geometry::pybind::define_binary_predicate<decltype(closest_points_impl),
                                            PAIRS(cartesian)>(
      m, "closest_points", kClosestPointsDoc, std::move(closest_points_impl));
}

}  // namespace pyinterp::geometry::cartesian::pybind

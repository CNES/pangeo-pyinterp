// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;
using pyinterp::geometry::pybind::GeometryNamespace;

namespace pyinterp::geometry::cartesian::pybind {

constexpr auto kUnionDoc = R"doc(
Computes the geometric union of two geometries.

The union operation returns the combined area covered by both geometries. The
result is returned as a vector of polygons or linestrings, depending on the
input geometry types.

Args:
    geometry1: First geometry.
    geometry2: Second geometry.

Returns:
    A list of geometries representing the union. For polygon inputs, returns
    a list of Polygon objects (typically one polygon, but may be multiple if
    geometries are disjoint). For linestring inputs, returns a list of
    LineString objects.
)doc";

// Macro for polygon geometry pairs that support union
#define UNION_POLYGON_PAIRS(NS)                                              \
  std::pair<NS::Ring, NS::Ring>, std::pair<NS::Ring, NS::Polygon>,           \
      std::pair<NS::Polygon, NS::Ring>, std::pair<NS::Polygon, NS::Polygon>, \
      std::pair<NS::MultiPolygon, NS::Ring>,                                 \
      std::pair<NS::MultiPolygon, NS::Polygon>,                              \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>,                         \
      std::pair<NS::Ring, NS::MultiPolygon>,                                 \
      std::pair<NS::Polygon, NS::MultiPolygon>

// Macro for linestring geometry pairs that support union
#define UNION_LINESTRING_PAIRS(NS)           \
  std::pair<NS::LineString, NS::LineString>, \
      std::pair<NS::MultiLineString, NS::MultiLineString>

auto init_union(nb::module_& m) -> void {
  auto _union_for_polygon_pairs_impl = [](const auto& g1,
                                          const auto& g2) -> auto {
    nb::gil_scoped_release release;
    std::vector<cartesian::Polygon> result;
    boost::geometry::union_(g1, g2, result);
    return result;
  };
  geometry::pybind::define_binary_predicate<
      decltype(_union_for_polygon_pairs_impl), UNION_POLYGON_PAIRS(cartesian)>(
      m, "union", kUnionDoc, std::move(_union_for_polygon_pairs_impl));

  auto _union_for_linestring_pairs_impl = [](const auto& g1,
                                             const auto& g2) -> auto {
    nb::gil_scoped_release release;
    std::vector<cartesian::LineString> result;
    boost::geometry::union_(g1, g2, result);
    return result;
  };
  geometry::pybind::define_binary_predicate<
      decltype(_union_for_linestring_pairs_impl),
      UNION_LINESTRING_PAIRS(cartesian)>(
      m, "union", kUnionDoc, std::move(_union_for_linestring_pairs_impl));
}

}  // namespace pyinterp::geometry::cartesian::pybind

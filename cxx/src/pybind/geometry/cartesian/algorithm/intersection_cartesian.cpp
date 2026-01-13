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

namespace pyinterp::geometry::cartesian::pybind {

constexpr auto kIntersectionDoc = R"doc(
Computes the geometric intersection between two geometries.

The intersection operation returns the part of the geometries that overlap.
The result is returned as a vector of polygons, linestrings, or points,
depending on the input geometry types and their spatial relationship.

Args:
    geometry1: First geometry.
    geometry2: Second geometry.

Returns:
    A list of geometry objects representing the intersection.
)doc";

// Macro for polygon-polygon intersection pairs
#define INTERSECTION_POLYGON_PAIRS(NS)                                       \
  std::pair<NS::Ring, NS::Ring>, std::pair<NS::Ring, NS::Polygon>,           \
      std::pair<NS::Polygon, NS::Ring>, std::pair<NS::Polygon, NS::Polygon>, \
      std::pair<NS::MultiPolygon, NS::Ring>,                                 \
      std::pair<NS::MultiPolygon, NS::Polygon>,                              \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>,                         \
      std::pair<NS::Ring, NS::MultiPolygon>,                                 \
      std::pair<NS::Polygon, NS::MultiPolygon>

// Macro for linestring-polygon intersection pairs (returns linestrings)
#define INTERSECTION_LINESTRING_POLYGON_PAIRS(NS)                              \
  std::pair<NS::LineString, NS::Ring>, std::pair<NS::LineString, NS::Polygon>, \
      std::pair<NS::Ring, NS::LineString>,                                     \
      std::pair<NS::Polygon, NS::LineString>

// Macro for linestring-linestring intersection pairs (returns points)
#define INTERSECTION_LINESTRING_PAIRS(NS) \
  std::pair<NS::LineString, NS::LineString>

auto init_intersection(nanobind::module_& m) -> void {
  // Polygon-polygon intersection
  auto intersection_for_polygon_pairs_impl =
      [](const auto& g1, const auto& g2) -> std::vector<Polygon> {
    nanobind::gil_scoped_release release;
    std::vector<Polygon> result;
    boost::geometry::intersection(g1, g2, result);
    return result;
  };
  geometry::pybind::define_binary_predicate<
      decltype(intersection_for_polygon_pairs_impl),
      INTERSECTION_POLYGON_PAIRS(cartesian)>(
      m, "intersection", kIntersectionDoc,
      std::move(intersection_for_polygon_pairs_impl));

  // Linestring-polygon intersection
  auto intersection_for_linestring_polygon_pairs_impl =
      [](const auto& g1, const auto& g2) -> std::vector<LineString> {
    nanobind::gil_scoped_release release;
    std::vector<LineString> result;
    boost::geometry::intersection(g1, g2, result);
    return result;
  };
  geometry::pybind::define_binary_predicate<
      decltype(intersection_for_linestring_polygon_pairs_impl),
      INTERSECTION_LINESTRING_POLYGON_PAIRS(cartesian)>(
      m, "intersection", kIntersectionDoc,
      std::move(intersection_for_linestring_polygon_pairs_impl));

  // Linestring-linestring intersection
  auto intersection_for_linestring_pairs_impl =
      [](const auto& g1, const auto& g2) -> std::vector<Point> {
    nanobind::gil_scoped_release release;
    std::vector<Point> result;
    boost::geometry::intersection(g1, g2, result);
    return result;
  };
  geometry::pybind::define_binary_predicate<
      decltype(intersection_for_linestring_pairs_impl),
      INTERSECTION_LINESTRING_PAIRS(cartesian)>(
      m, "intersection", kIntersectionDoc,
      std::move(intersection_for_linestring_pairs_impl));
}

}  // namespace pyinterp::geometry::cartesian::pybind

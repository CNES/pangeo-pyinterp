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

constexpr auto kDifferenceDoc = R"doc(
Computes the geometric difference between two geometries.

The difference operation returns the part of the first geometry that does not
intersect with the second geometry. The result is returned as a list
of polygons or linestrings, depending on the input geometry types.

Args:
    geometry1: First geometry (minuend).
    geometry2: Second geometry (subtrahend).

Returns:
    A list of geometries representing the difference. For polygon inputs,
    returns a list of Polygon objects. For linestring inputs, returns a list of
    LineString objects.
)doc";

// Macro for polygon geometry pairs that support difference
#define DIFFERENCE_POLYGON_PAIRS(NS)                                         \
  std::pair<NS::Ring, NS::Ring>, std::pair<NS::Ring, NS::Polygon>,           \
      std::pair<NS::Polygon, NS::Ring>, std::pair<NS::Polygon, NS::Polygon>, \
      std::pair<NS::MultiPolygon, NS::Ring>,                                 \
      std::pair<NS::MultiPolygon, NS::Polygon>,                              \
      std::pair<NS::MultiPolygon, NS::MultiPolygon>,                         \
      std::pair<NS::Ring, NS::MultiPolygon>,                                 \
      std::pair<NS::Polygon, NS::MultiPolygon>

auto init_difference(nanobind::module_& m) -> void {
  auto difference_impl = [](const auto& g1, const auto& g2) -> auto {
    nanobind::gil_scoped_release release;
    std::vector<Polygon> result;
    boost::geometry::difference(g1, g2, result);
    return result;
  };
  geometry::pybind::define_binary_predicate<
      decltype(difference_impl), DIFFERENCE_POLYGON_PAIRS(cartesian)>(
      m, "difference", kDifferenceDoc, std::move(difference_impl));
}

}  // namespace pyinterp::geometry::cartesian::pybind

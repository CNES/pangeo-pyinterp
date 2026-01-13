// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/geometry/cartesian/polygon.hpp"
#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::cartesian::pybind {

constexpr auto kConvexHullDoc = R"doc(
Calculates the convex hull of a geometry.

The convex hull is the smallest convex geometry that contains all points
of the input geometry. It is always returned as a Polygon.

Args:
    geometry: Geometric object to compute convex hull for.

Returns:
    A Polygon representing the convex hull.
)doc";

auto init_convex_hull(nb::module_& m) -> void {
  auto convex_hull_impl = [](const auto& g) -> Polygon {
    nanobind::gil_scoped_release release;
    Polygon result;
    boost::geometry::convex_hull(g, result);
    return result;
  };
  geometry::pybind::define_unary_predicate<decltype(convex_hull_impl),
                                           GEOMETRY_TYPES(cartesian)>(
      m, "convex_hull", kConvexHullDoc, std::move(convex_hull_impl));
  ;
}

}  // namespace pyinterp::geometry::cartesian::pybind

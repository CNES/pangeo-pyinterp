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

constexpr auto kPerimeterDoc = R"doc(
Calculates the perimeter of an areal geometry.

For polygons and rings, this is the sum of the lengths of all rings
(exterior and holes). For other geometries, returns 0.

Args:
    geometry: Geometric object to compute perimeter for.

Returns:
    The perimeter of the geometry (in coordinate units for Cartesian,
    meters for geographic).
)doc";

auto init_perimeter(nanobind::module_& m) -> void {
  auto perimeter_impl = [](const auto& g) -> double {
    nanobind::gil_scoped_release release;
    return boost::geometry::perimeter(g);
  };
  geometry::pybind::define_unary_predicate<decltype(perimeter_impl),
                                           GEOMETRY_TYPES(cartesian)>(
      m, "perimeter", kPerimeterDoc, std::move(perimeter_impl));
}

}  // namespace pyinterp::geometry::cartesian::pybind

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>
#include <optional>

#include "pyinterp/geometry/cartesian/point.hpp"
#include "pyinterp/geometry/geographic/algorithms/azimuth.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::cartesian::pybind {

constexpr auto kAzimuthDoc = R"doc(
Calculate the azimuth from point1 to point2.

The azimuth is the angle measured clockwise from north to the direction
from point1 to point2.

Args:
    point1: First point.
    point2: Second point.

Returns:
    Azimuth in radians (0 to 2π).
)doc";

auto init_azimuth(nb::module_& m) -> void {
  m.def(
      "azimuth",
      [](const cartesian::Point& point1,
         const cartesian::Point& point2) -> double {
        nb::gil_scoped_release release;
        return boost::geometry::azimuth(point1, point2);
      },
      "point1"_a, "point2"_a, kAzimuthDoc);
}

}  // namespace pyinterp::geometry::cartesian::pybind

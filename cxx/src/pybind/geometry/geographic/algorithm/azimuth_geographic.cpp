// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <boost/geometry.hpp>
#include <optional>

#include "pyinterp/geometry/geographic/algorithms/azimuth.hpp"
#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kAzimuthDoc = R"doc(
Calculate the azimuth from point1 to point2.

The azimuth is the angle measured clockwise from north to the direction
from point1 to point2.

Args:
    point1: First point.
    point2: Second point.
    spheroid: Optional spheroid for geodetic calculations. If not provided,
        uses WGS84 ellipsoid.
    strategy: Calculation strategy.

Returns:
    Azimuth in radians (0 to 2π).
)doc";

auto init_azimuth(nb::module_& m) -> void {
  m.def(
      "azimuth",
      [](const geographic::Point& point1, const geographic::Point& point2,
         const std::optional<geographic::Spheroid>& spheroid,
         const geographic::StrategyMethod& strategy) -> double {
        nb::gil_scoped_release release;
        return azimuth(point1, point2, spheroid, strategy);
      },
      "point1"_a, "point2"_a, "spheroid"_a = std::nullopt,
      "strategy"_a = StrategyMethod::kVincenty, kAzimuthDoc);
}

}  // namespace pyinterp::geometry::geographic::pybind

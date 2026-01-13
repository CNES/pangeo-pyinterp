// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "pyinterp/geometry/geographic/algorithms/perimeter.hpp"
#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kPerimeterDoc = R"doc(
Calculates the perimeter of an areal geometry.

For polygons and rings, this is the sum of the lengths of all rings (exterior
and holes) computed using geodetic calculations on the spheroid.

Args:
    geometry: Geometric object to compute perimeter for.
    spheroid: Optional spheroid for geodetic calculations.
    strategy: Calculation strategy.

Returns:
    The perimeter of the geometry in meters.
)doc";

auto init_perimeter(nb::module_& m) -> void {
  auto perimeter_impl = [](const auto& g,
                           const std::optional<Spheroid>& spheroid,
                           const StrategyMethod& strategy) -> auto {
    using GeometryType = std::decay_t<decltype(g)>;
    nb::gil_scoped_release release;
    return perimeter<GeometryType>(g, spheroid, strategy);
  };
  geometry::pybind::define_unary_predicate_with_strategy<
      decltype(perimeter_impl), Spheroid, StrategyMethod, Ring, Polygon,
      MultiPolygon>(m, "perimeter", kPerimeterDoc, std::move(perimeter_impl));
}

}  // namespace pyinterp::geometry::geographic::pybind

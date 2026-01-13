// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "pyinterp/geometry/geographic/algorithms/length.hpp"
#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kLengthDoc = R"doc(
Calculate the length of a geometric object.

The length is calculated on the surface of a spheroid (default: WGS84).
Different geodetic calculation strategies are available for accuracy/performance
trade-offs.

Args:
    geometry: Geometric object.
    spheroid: Optional spheroid for geodetic calculations. If not provided, uses
        WGS84 ellipsoid.
    strategy: Calculation strategy.
Returns:
    Length in meters.
)doc";

auto init_length(nb::module_& m) -> void {
  auto length_impl = [](const auto& geometry,
                        const std::optional<Spheroid>& wgs,
                        StrategyMethod strategy) -> double {
    nb::gil_scoped_release release;
    return length(geometry, wgs, strategy);
  };

  geometry::pybind::define_unary_predicate_with_strategy<
      decltype(length_impl), Spheroid, StrategyMethod,
      GEOMETRY_TYPES(geographic)>(m, "length", kLengthDoc,
                                  std::move(length_impl));
}

}  // namespace pyinterp::geometry::geographic::pybind

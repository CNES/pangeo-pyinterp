// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"
#include "pyinterp/pybind/geometry/geographic/overlay.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::geometry::geographic::pybind {

constexpr auto kDifferenceDoc = R"doc(
Computes the geometric difference between two geometries.

Args:
    geometry1: First geometry (minuend).
    geometry2: Second geometry (subtrahend).
    spheroid: Optional spheroid for geodetic calculations.
    strategy: Calculation strategy.

Returns:
    A list of Polygon objects representing the difference.
)doc";

auto init_difference(nb::module_& m) -> void {
  auto difference_impl = [](const auto& geometry1, const auto& geometry2,
                            const std::optional<Spheroid>& spheroid,
                            StrategyMethod strategy) -> auto {
    nb::gil_scoped_release release;
    return overlay<Overlay::kDifference, Polygon>(geometry1, geometry2,
                                                  spheroid, strategy);
  };
  geometry::pybind::define_binary_predicate_with_strategy<
      decltype(difference_impl), Spheroid, StrategyMethod,
      DIFFERENCE_POLYGON_PAIRS(geographic)>(m, "difference", kDifferenceDoc,
                                            std::move(difference_impl));
}

}  // namespace pyinterp::geometry::geographic::pybind

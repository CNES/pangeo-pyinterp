// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;

namespace pyinterp::geometry::cartesian::pybind {

constexpr auto kDistanceDoc = R"doc(
Calculates the distance between two geometries.

Args:
    geometry1: First geometric object.
    geometry2: Second geometric object.

Returns:
    The distance between the geometries in cartesian coordinate units.
)doc";

/// @brief Initialize the distance algorithm in the given module
/// @param[in,out] m Nanobind module
auto init_distance(nanobind::module_& m) -> void {
  auto distance_impl = [](const auto& geometry1,
                          const auto& geometry2) -> double {
    nanobind::gil_scoped_release release;
    return boost::geometry::distance(geometry1, geometry2);
  };
  geometry::pybind::define_binary_predicate<decltype(distance_impl),
                                            GEOMETRY_PAIRS(cartesian)>(
      m, "distance", kDistanceDoc, std::move(distance_impl));
}

}  // namespace pyinterp::geometry::cartesian::pybind

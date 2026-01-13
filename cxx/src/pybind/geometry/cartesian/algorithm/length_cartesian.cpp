// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;

namespace pyinterp::geometry::cartesian::pybind {

constexpr auto kLengthDoc = R"doc(
Calculates the length of a geometry.

Args:
    geometry: Geometric object to compute length for.

Returns:
    The length of the geometry in cartesian coordinate units.
)doc";

/// @brief Initialize the length algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
auto init_length(nb::module_& m) -> void {
  auto length_impl = [](const auto& g) -> double {
    nb::gil_scoped_release release;
    return boost::geometry::length(g);
  };
  geometry::pybind::define_unary_predicate<decltype(length_impl),
                                           GEOMETRY_TYPES(cartesian)>(
      m, "length", kLengthDoc, std::move(length_impl));
}

}  // namespace pyinterp::geometry::cartesian::pybind

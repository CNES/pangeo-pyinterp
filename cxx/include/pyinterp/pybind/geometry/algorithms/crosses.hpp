// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include <nanobind/nanobind.h>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace nb = nanobind;

namespace pyinterp::geometry::pybind {

constexpr auto kCrossesDoc = R"doc(
Checks if the first geometry crosses the second geometry.

Args:
    geometry1: First geometric object.
    geometry2: Second geometric object.

Returns:
    True if geometry1 crosses geometry2, False otherwise.
)doc";

/// @brief Initialize the crosses algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_crosses(nanobind::module_& m) -> void {
  auto crosses_impl = [](const auto& geometry1, const auto& geometry2) -> bool {
    nanobind::gil_scoped_release release;
    return boost::geometry::crosses(geometry1, geometry2);
  };
  if constexpr (NS == GeometryNamespace::kCartesian) {
    geometry::pybind::define_binary_predicate<decltype(crosses_impl),
                                              CROSSES_PAIRS(cartesian)>(
        m, "crosses", kCrossesDoc, std::move(crosses_impl));
  } else {
    geometry::pybind::define_binary_predicate<decltype(crosses_impl),
                                              CROSSES_PAIRS(geographic)>(
        m, "crosses", kCrossesDoc, std::move(crosses_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

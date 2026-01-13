// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kEnvelopeDoc = R"doc(
Calculates the envelope (bounding box) of a geometry.

The envelope is the smallest axis-aligned box that contains the entire geometry.

Args:
    geometry: Geometric object to compute envelope for.

Returns:
    A Box representing the minimum bounding rectangle.
)doc";

/// @brief Initialize the envelope algorithm in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_envelope(nanobind::module_& m) -> void {
  if constexpr (NS == GeometryNamespace::kCartesian) {
    auto envelope_impl = [](const auto& g) -> cartesian::Box {
      nanobind::gil_scoped_release release;
      cartesian::Box result;
      boost::geometry::envelope(g, result);
      return result;
    };
    geometry::pybind::define_unary_predicate<decltype(envelope_impl),
                                             GEOMETRY_TYPES(cartesian)>(
        m, "envelope", kEnvelopeDoc, std::move(envelope_impl));
  } else {
    auto envelope_impl = [](const auto& g) -> geographic::Box {
      nanobind::gil_scoped_release release;
      geographic::Box result;
      boost::geometry::envelope(g, result);
      return result;
    };
    geometry::pybind::define_unary_predicate<decltype(envelope_impl),
                                             GEOMETRY_TYPES(geographic)>(
        m, "envelope", kEnvelopeDoc, std::move(envelope_impl));
  }
}

}  // namespace pyinterp::geometry::pybind

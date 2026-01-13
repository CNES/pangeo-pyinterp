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

constexpr auto kDensifyDoc = R"doc(
Densifies a geometry by adding points along segments.

The algorithm adds points along each segment so that no segment is longer
than the specified maximum distance. This is useful for better approximating
curves or for preparing geometries for projection transformations.

Args:
    geometry: Geometric object to densify.
    max_distance: Maximum allowed distance between consecutive points.

Returns:
    A densified version of the input geometry (same type).
)doc";

/// @brief Helper to define densify for multiple geometry types
/// @tparam Geometries Geometry types to bind
/// @param[in] m Python module
/// @param[in] doc Documentation string
template <typename... Geometries>
inline auto define_densify_for_geometries(nanobind::module_& m, const char* doc)
    -> void {
  auto densify_impl = [](const auto& g, double max_distance) -> auto {
    using GeometryType = std::decay_t<decltype(g)>;
    nanobind::gil_scoped_release release;
    GeometryType result;
    boost::geometry::densify(g, result, max_distance);
    return result;
  };
  (..., m.def(
            "densify",
            [densify_impl](const Geometries& g, double max_distance) -> auto {
              return densify_impl(g, max_distance);
            },
            "geometry"_a, "max_distance"_a, doc));
}

// Macro for geometries that support densify densify requires container-like
// geometries
#define DENSIFY_TYPES(NS) \
  NS::LineString, NS::Ring, NS::Polygon, NS::MultiLineString, NS::MultiPolygon

auto init_densify(nanobind::module_& m) -> void {
  define_densify_for_geometries<DENSIFY_TYPES(cartesian)>(m, kDensifyDoc);
}

}  // namespace pyinterp::geometry::cartesian::pybind

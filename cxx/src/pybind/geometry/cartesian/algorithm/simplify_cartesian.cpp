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

constexpr auto kSimplifyDoc = R"doc(
Simplifies a geometry using the Douglas-Peucker algorithm.

The algorithm removes points from the geometry while preserving its general shape.
Points are removed if they are within the specified distance tolerance from
the simplified line.

Args:
    geometry: Geometric object to simplify.
    distance: Maximum distance tolerance for simplification.

Returns:
    A simplified version of the input geometry (same type).

Examples:
    >>> # LineString with many points
    >>> ls = LineString(np.array([0.0, 0.1, 0.2, 0.3, 1.0]),
    ...                 np.array([0.0, 0.05, 0.0, 0.05, 0.0]))
    >>> simplified = simplify(ls, distance=0.1)
    >>> # simplified has fewer points but similar shape
)doc";

/// @brief Helper to define simplify for multiple geometry types
/// @tparam Geometries Geometry types to bind
/// @param[in] m Python module
/// @param[in] doc Documentation string
template <typename... Geometries>
inline auto define_simplify_for_geometries(nanobind::module_& m,
                                           const char* doc) -> void {
  auto simplify_impl = [](const auto& g, double distance) -> auto {
    using GeometryType = std::decay_t<decltype(g)>;

    nanobind::gil_scoped_release release;
    GeometryType result;
    boost::geometry::simplify(g, result, distance);
    return result;
  };
  (..., m.def(
            "simplify",
            [simplify_impl](const Geometries& g, double distance) -> auto {
              return simplify_impl(g, distance);
            },
            "geometry"_a, "distance"_a, doc));
}

// Macro for geometries that support simplify
#define SIMPLIFY_TYPES(NS) \
  NS::LineString, NS::Ring, NS::Polygon, NS::MultiLineString, NS::MultiPolygon

auto init_simplify(nanobind::module_& m) -> void {
  define_simplify_for_geometries<SIMPLIFY_TYPES(cartesian)>(m, kSimplifyDoc);
}

}  // namespace pyinterp::geometry::cartesian::pybind

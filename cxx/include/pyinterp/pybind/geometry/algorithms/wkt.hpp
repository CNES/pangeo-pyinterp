// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <algorithm>
#include <boost/geometry.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <variant>

#include "pyinterp/pybind/geometry/algorithm_binding_helpers.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kToWktDoc = R"doc(
Converts a geometry to Well-Known Text (WKT) representation.

WKT is a text markup language for representing vector geometry objects. It provides
a human-readable format for geometry data interchange.

Args:
    geometry: The geometry to convert.

Returns:
    A string containing the WKT representation (e.g., "POINT(1 2)").

Examples:
    >>> point = Point(1.0, 2.0)
    >>> wkt = to_wkt(point)
    >>> # Returns "POINT(1 2)"

    >>> polygon = Polygon(...)
    >>> wkt = to_wkt(polygon)
    >>> # Returns "POLYGON((...))"
)doc";

constexpr auto kFromWktDoc = R"doc(
Creates a geometry from Well-Known Text (WKT) representation.

Parses a WKT string and constructs the corresponding geometry object.
The function automatically detects the geometry type from the WKT string
and returns the appropriate geometry object.

Args:
    wkt: WKT string representation of the geometry.

Returns:
    A variant containing one of: Point, LineString, Ring, Polygon, MultiPoint,
    MultiLineString, or MultiPolygon depending on the WKT content.

Examples:
    >>> point = from_wkt("POINT(1 2)")
    >>> # Returns Point(1.0, 2.0)

    >>> polygon = from_wkt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    >>> # Returns Polygon object

    >>> linestring = from_wkt("LINESTRING(0 0, 1 1, 2 2)")
    >>> # Returns LineString object
)doc";

/// @brief Macro for all geometry types supporting WKT
#define WKT_TYPES(NS)                                               \
  NS::Point, NS::LineString, NS::Ring, NS::Polygon, NS::MultiPoint, \
      NS::MultiLineString, NS::MultiPolygon

/// @brief Helper to define to_wkt for geometry types
/// @tparam Geometries Geometry types
/// @param[in] m Python module
/// @param[in] doc Documentation string
template <typename... Geometries>
inline auto define_to_wkt(nanobind::module_& m, const char* doc) -> void {
  auto to_wkt_impl = [](const auto& g) -> std::string {
    nanobind::gil_scoped_release release;
    std::ostringstream oss;
    oss << boost::geometry::wkt(g);
    return oss.str();
  };

  (...,
   m.def(
       "to_wkt", [to_wkt_impl](const Geometries& g) { return to_wkt_impl(g); },
       "geometry"_a, doc));
}

/// @brief Helper to define unified from_wkt using std::variant
/// @tparam Geometries All geometry types to support
/// @param[in] m Python module
/// @param[in] doc Documentation string
template <typename... Geometries>
inline auto define_from_wkt(nanobind::module_& m, const char* doc) -> void {
  using GeometryVariant = std::variant<Geometries...>;

  m.def(
      "from_wkt",
      [](const std::string& wkt) -> GeometryVariant {
        nanobind::gil_scoped_release release;

        // Try to parse as each geometry type in order
        // We detect the type by looking at the WKT prefix
        std::string wkt_upper = wkt;
        std::ranges::transform(wkt_upper, wkt_upper.begin(), ::toupper);

        // Helper to try parsing as a specific geometry type
        auto try_parse =
            []<typename Geometry>(
                const std::string& wkt_str) -> std::optional<Geometry> {
          try {
            Geometry geometry;
            boost::geometry::read_wkt(wkt_str, geometry);
            return geometry;
          } catch (...) {
            return std::nullopt;
          }
        };

        // Try each geometry type based on WKT prefix
        if (wkt_upper.starts_with("POINT")) {
          using Point = std::tuple_element_t<0, std::tuple<Geometries...>>;
          if (auto geom = try_parse.template operator()<Point>(wkt)) {
            return *geom;
          }
        } else if (wkt_upper.starts_with("LINESTRING")) {
          using LineString = std::tuple_element_t<1, std::tuple<Geometries...>>;
          if (auto geom = try_parse.template operator()<LineString>(wkt)) {
            return *geom;
          }
        } else if (wkt_upper.starts_with("POLYGON")) {
          // Try Polygon first (more common than Ring)
          using Polygon = std::tuple_element_t<3, std::tuple<Geometries...>>;
          if (auto geom = try_parse.template operator()<Polygon>(wkt)) {
            return *geom;
          }
        } else if (wkt_upper.starts_with("MULTIPOINT")) {
          using MultiPoint = std::tuple_element_t<4, std::tuple<Geometries...>>;
          if (auto geom = try_parse.template operator()<MultiPoint>(wkt)) {
            return *geom;
          }
        } else if (wkt_upper.starts_with("MULTILINESTRING")) {
          using MultiLineString =
              std::tuple_element_t<5, std::tuple<Geometries...>>;
          if (auto geom = try_parse.template operator()<MultiLineString>(wkt)) {
            return *geom;
          }
        } else if (wkt_upper.starts_with("MULTIPOLYGON")) {
          using MultiPolygon =
              std::tuple_element_t<6, std::tuple<Geometries...>>;
          if (auto geom = try_parse.template operator()<MultiPolygon>(wkt)) {
            return *geom;
          }
        }

        throw std::runtime_error("Unable to parse WKT string: " + wkt);
      },
      "wkt"_a, doc);
}

/// @brief Initialize WKT algorithms in the given module
/// @tparam NS Namespace of the geometries (cartesian or geographic)
/// @param[in,out] m Nanobind module
template <GeometryNamespace NS>
inline auto init_wkt(nanobind::module_& m) -> void {
  if constexpr (NS == GeometryNamespace::kCartesian) {
    // to_wkt for all geometry types
    define_to_wkt<WKT_TYPES(cartesian)>(m, kToWktDoc);

    // Unified from_wkt using variant
    define_from_wkt<WKT_TYPES(cartesian)>(m, kFromWktDoc);
  } else {
    // to_wkt for all geometry types
    define_to_wkt<WKT_TYPES(geographic)>(m, kToWktDoc);

    // Unified from_wkt using variant
    define_from_wkt<WKT_TYPES(geographic)>(m, kFromWktDoc);
  }
}

}  // namespace pyinterp::geometry::pybind

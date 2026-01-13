// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <nanobind/nanobind.h>

#include <boost/geometry.hpp>

#include "pyinterp/geometry/cartesian/box.hpp"
#include "pyinterp/geometry/cartesian/linestring.hpp"
#include "pyinterp/geometry/cartesian/multi_linestring.hpp"
#include "pyinterp/geometry/cartesian/multi_point.hpp"
#include "pyinterp/geometry/cartesian/multi_polygon.hpp"
#include "pyinterp/geometry/cartesian/point.hpp"
#include "pyinterp/geometry/cartesian/polygon.hpp"
#include "pyinterp/geometry/cartesian/ring.hpp"
#include "pyinterp/geometry/cartesian/segment.hpp"
#include "pyinterp/geometry/geographic/box.hpp"
#include "pyinterp/geometry/geographic/linestring.hpp"
#include "pyinterp/geometry/geographic/multi_linestring.hpp"
#include "pyinterp/geometry/geographic/multi_point.hpp"
#include "pyinterp/geometry/geographic/multi_polygon.hpp"
#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/geographic/polygon.hpp"
#include "pyinterp/geometry/geographic/ring.hpp"
#include "pyinterp/geometry/geographic/segment.hpp"

namespace pyinterp::geometry::pybind {

constexpr auto kConvertCartesianToGeographicDoc = R"doc(
Converts a Cartesian geometry to Geographic coordinates.

This function converts geometries from Cartesian (x, y) coordinates to
Geographic (longitude, latitude) coordinates. The conversion assumes
the Cartesian coordinates are already in appropriate units (typically degrees).

Args:
    geometry: Cartesian geometric object to convert.

Returns:
    Equivalent geometry in Geographic coordinates.
)doc";

constexpr auto kConvertGeographicToCartesianDoc = R"doc(
Converts a Geographic geometry to Cartesian coordinates.

This function converts geometries from Geographic (longitude, latitude)
coordinates to Cartesian (x, y) coordinates. The conversion simply copies
the coordinate values.

Args:
    geometry: Geographic geometric object to convert.

Returns:
    Equivalent geometry in Cartesian coordinates.
)doc";

/// @brief Helper to define convert from Cartesian to Geographic
/// @tparam CartesianGeometries Cartesian geometry types
/// @tparam GeographicGeometry Corresponding geographic geometry type
template <typename GeographicGeometry, typename... CartesianGeometries>
inline auto add_geographic_converter(nanobind::module_& m, const char* doc)
    -> void {
  auto convert_impl = [](const auto& g) -> GeographicGeometry {
    nanobind::gil_scoped_release release;
    GeographicGeometry result;
    boost::geometry::convert(g, result);
    return result;
  };
  (...,
   m.def(
       "convert_to_geographic",
       [convert_impl](const CartesianGeometries& g) { return convert_impl(g); },
       nanobind::arg("geometry"), doc));
}

/// @brief Helper to define convert from Geographic to Cartesian
/// @tparam GeographicGeometries Geographic geometry types
/// @tparam CartesianGeometry Corresponding cartesian geometry type
template <typename CartesianGeometry, typename... GeographicGeometries>
inline auto add_cartesian_converter(nanobind::module_& m, const char* doc)
    -> void {
  auto convert_impl = [](const auto& g) -> CartesianGeometry {
    nanobind::gil_scoped_release release;
    CartesianGeometry result;
    boost::geometry::convert(g, result);
    return result;
  };
  (..., m.def(
            "convert_to_cartesian",
            [convert_impl](const GeographicGeometries& g) {
              return convert_impl(g);
            },
            nanobind::arg("geometry"), doc));
}

/// @brief Initialize convert operations for Cartesian namespace
/// @param[in,out] m Nanobind module
inline auto init_convert_cartesian(nanobind::module_& m) -> void {
  // Cartesian to Geographic conversions
  add_geographic_converter<geographic::Point, cartesian::Point>(
      m, kConvertCartesianToGeographicDoc);
  add_geographic_converter<geographic::Box, cartesian::Box>(
      m, kConvertCartesianToGeographicDoc);
  add_geographic_converter<geographic::Segment, cartesian::Segment>(
      m, kConvertCartesianToGeographicDoc);
  add_geographic_converter<geographic::LineString, cartesian::LineString>(
      m, kConvertCartesianToGeographicDoc);
  add_geographic_converter<geographic::Ring, cartesian::Ring>(
      m, kConvertCartesianToGeographicDoc);
  add_geographic_converter<geographic::Polygon, cartesian::Polygon>(
      m, kConvertCartesianToGeographicDoc);
  add_geographic_converter<geographic::MultiPoint, cartesian::MultiPoint>(
      m, kConvertCartesianToGeographicDoc);
  add_geographic_converter<geographic::MultiLineString,
                           cartesian::MultiLineString>(
      m, kConvertCartesianToGeographicDoc);
  add_geographic_converter<geographic::MultiPolygon, cartesian::MultiPolygon>(
      m, kConvertCartesianToGeographicDoc);
}

/// @brief Initialize convert operations for Geographic namespace
/// @param[in,out] m Nanobind module
inline auto init_convert_geographic(nanobind::module_& m) -> void {
  // Geographic to Cartesian conversions
  add_cartesian_converter<cartesian::Point, geographic::Point>(
      m, kConvertGeographicToCartesianDoc);
  add_cartesian_converter<cartesian::Box, geographic::Box>(
      m, kConvertGeographicToCartesianDoc);
  add_cartesian_converter<cartesian::Segment, geographic::Segment>(
      m, kConvertGeographicToCartesianDoc);
  add_cartesian_converter<cartesian::LineString, geographic::LineString>(
      m, kConvertGeographicToCartesianDoc);
  add_cartesian_converter<cartesian::Ring, geographic::Ring>(
      m, kConvertGeographicToCartesianDoc);
  add_cartesian_converter<cartesian::Polygon, geographic::Polygon>(
      m, kConvertGeographicToCartesianDoc);
  add_cartesian_converter<cartesian::MultiPoint, geographic::MultiPoint>(
      m, kConvertGeographicToCartesianDoc);
  add_cartesian_converter<cartesian::MultiLineString,
                          geographic::MultiLineString>(
      m, kConvertGeographicToCartesianDoc);
  add_cartesian_converter<cartesian::MultiPolygon, geographic::MultiPolygon>(
      m, kConvertGeographicToCartesianDoc);
}

}  // namespace pyinterp::geometry::pybind

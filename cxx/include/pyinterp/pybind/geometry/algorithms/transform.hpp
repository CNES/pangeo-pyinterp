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

constexpr auto kTransformDoc = R"doc(
Transform a geometry to a different geometry type.

Args:
    geometry: Geometric object to transform.

Returns:
    Transformed geometry in the requested type.
)doc";

/// @brief Transform using boost::geometry::convert
/// @tparam SourceGeometry Source geometry type
/// @tparam TargetGeometry Target geometry type
/// @param[in] source Source geometry
/// @return Transformed geometry
template <typename SourceGeometry, typename TargetGeometry>
inline auto do_transform(const SourceGeometry& source) -> TargetGeometry {
  TargetGeometry target;
  boost::geometry::convert(source, target);
  return target;
}

/// @brief Initialize transform operations for Cartesian namespace
/// @param[in,out] m Nanobind module
template <typename Point, typename Segment, typename LineString, typename Ring,
          typename Box, typename Polygon, typename MultiPoint,
          typename MultiLineString, typename MultiPolygon>
constexpr auto init_transform(nanobind::module_& m) -> void {
  // Identity transforms (same type to same type)
  m.def(
      "transform_to_point",
      [](const Point& g) { return do_transform<Point, Point>(g); },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_box",
      [](const Box& g) { return do_transform<Box, Box>(g); },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_segment",
      [](const Segment& g) { return do_transform<Segment, Segment>(g); },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_linestring",
      [](const LineString& g) {
        return do_transform<LineString, LineString>(g);
      },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_ring",
      [](const Ring& g) { return do_transform<Ring, Ring>(g); },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_polygon",
      [](const Polygon& g) { return do_transform<Polygon, Polygon>(g); },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_multipoint",
      [](const MultiPoint& g) {
        return do_transform<MultiPoint, MultiPoint>(g);
      },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_multilinestring",
      [](const MultiLineString& g) {
        return do_transform<MultiLineString, MultiLineString>(g);
      },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_multipolygon",
      [](const MultiPolygon& g) {
        return do_transform<MultiPolygon, MultiPolygon>(g);
      },
      nanobind::arg("geometry"), kTransformDoc);

  // Segment -> LineString (supported: convert.hpp line 389-392)
  m.def(
      "transform_to_linestring",
      [](const Segment& g) { return do_transform<Segment, LineString>(g); },
      nanobind::arg("geometry"), kTransformDoc);

  // Box conversions (supported: convert.hpp lines 410-442)
  m.def(
      "transform_to_ring",
      [](const Box& g) { return do_transform<Box, Ring>(g); },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_multipoint",
      [](const Box& g) { return do_transform<Box, MultiPoint>(g); },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_polygon",
      [](const Box& g) { return do_transform<Box, Polygon>(g); },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_multipolygon",
      [](const Box& g) { return do_transform<Box, MultiPolygon>(g); },
      nanobind::arg("geometry"), kTransformDoc);

  // Point -> Box (supported: convert.hpp line 445-459)
  m.def(
      "transform_to_box",
      [](const Point& g) { return do_transform<Point, Box>(g); },
      nanobind::arg("geometry"), kTransformDoc);

  // Ring <-> Polygon
  m.def(
      "transform_to_polygon",
      [](const Ring& g) { return do_transform<Ring, Polygon>(g); },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_ring",
      [](const Polygon& g) { return do_transform<Polygon, Ring>(g); },
      nanobind::arg("geometry"), kTransformDoc);

  // Range -> MultiPoint
  m.def(
      "transform_to_multipoint",
      [](const LineString& g) {
        return do_transform<LineString, MultiPoint>(g);
      },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_multipoint",
      [](const Ring& g) { return do_transform<Ring, MultiPoint>(g); },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_multipoint",
      [](const Polygon& g) { return do_transform<Polygon, MultiPoint>(g); },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_multipoint",
      [](const MultiPolygon& g) {
        return do_transform<MultiPolygon, MultiPoint>(g);
      },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_multipoint",
      [](const MultiLineString& g) {
        return do_transform<MultiLineString, MultiPoint>(g);
      },
      nanobind::arg("geometry"), kTransformDoc);

  // Ring/Polygon/MultiPolygon -> MultiLineString
  m.def(
      "transform_to_multilinestring",
      [](const Ring& g) { return do_transform<Ring, MultiLineString>(g); },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_multilinestring",
      [](const Polygon& g) {
        return do_transform<Polygon, MultiLineString>(g);
      },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_multilinestring",
      [](const MultiPolygon& g) {
        return do_transform<MultiPolygon, MultiLineString>(g);
      },
      nanobind::arg("geometry"), kTransformDoc);

  // Single -> Multi conversions
  m.def(
      "transform_to_multipoint",
      [](const Point& g) { return do_transform<Point, MultiPoint>(g); },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_multilinestring",
      [](const Segment& g) {
        return do_transform<Segment, MultiLineString>(g);
      },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_multilinestring",
      [](const LineString& g) {
        return do_transform<LineString, MultiLineString>(g);
      },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_multipolygon",
      [](const Ring& g) { return do_transform<Ring, MultiPolygon>(g); },
      nanobind::arg("geometry"), kTransformDoc);
  m.def(
      "transform_to_multipolygon",
      [](const Polygon& g) { return do_transform<Polygon, MultiPolygon>(g); },
      nanobind::arg("geometry"), kTransformDoc);
}

/// @brief Initialize transform operations for Cartesian namespace
/// @param[in,out] m Nanobind module
inline auto init_transform_cartesian(nanobind::module_& m) -> void {
  init_transform<cartesian::Point, cartesian::Segment, cartesian::LineString,
                 cartesian::Ring, cartesian::Box, cartesian::Polygon,
                 cartesian::MultiPoint, cartesian::MultiLineString,
                 cartesian::MultiPolygon>(m);
}

/// @brief Initialize transform operations for Geographic namespace
/// @param[in,out] m Nanobind module
inline auto init_transform_geographic(nanobind::module_& m) -> void {
  init_transform<geographic::Point, geographic::Segment, geographic::LineString,
                 geographic::Ring, geographic::Box, geographic::Polygon,
                 geographic::MultiPoint, geographic::MultiLineString,
                 geographic::MultiPolygon>(m);
}

}  // namespace pyinterp::geometry::pybind

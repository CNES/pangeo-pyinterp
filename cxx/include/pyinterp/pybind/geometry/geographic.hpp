// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <nanobind/nanobind.h>

namespace pyinterp::geometry::geographic::pybind {

/// @brief Initialize Strategy enum bindings
/// @param[in,out] m Nanobind module
auto init_strategy(nanobind::module_& m) -> void;

/// @brief Initialize Point bindings
/// @param[in,out] m Python module
auto init_point(nanobind::module_& m) -> void;

/// @brief Initialize Box bindings
/// @param[in,out] m Python module
auto init_box(nanobind::module_& m) -> void;

/// @brief Initialize Ring bindings
/// @param[in,out] m Python module
auto init_ring(nanobind::module_& m) -> void;

/// @brief Initialize LineString bindings
/// @param[in,out] m Python module
auto init_linestring(nanobind::module_& m) -> void;

/// @brief Initialize Segment bindings
/// @param[in,out] m Python module
auto init_segment(nanobind::module_& m) -> void;

/// @brief Initialize Polygon bindings
/// @param[in,out] m Python module
auto init_polygon(nanobind::module_& m) -> void;

/// @brief Initialize MultiPoint bindings
/// @param[in,out] m Python module
auto init_multipoint(nanobind::module_& m) -> void;

/// @brief Initialize MultiPolygon bindings
/// @param[in,out] m Python module
auto init_multipolygon(nanobind::module_& m) -> void;

/// @brief Initialize MultiLineString bindings
/// @param[in,out] m Python module
auto init_multilinestring(nanobind::module_& m) -> void;

/// @brief Initialize Spheroid bindings
/// @param[in,out] m Python module
auto init_spheroid(nanobind::module_& m) -> void;

/// @brief Initialize coordinate transformation bindings
/// @param[in,out] m Python module
auto init_coordinates(nanobind::module_& m) -> void;

/// @brief Initialize geographic RTree bindings
/// @param[in,out] m Python module
auto init_rtree(nanobind::module_& m) -> void;

/// @brief Initialize area algorithm bindings
/// @param[in,out] m Python module
auto init_area(nanobind::module_& m) -> void;

/// @brief Initialize azimuth algorithm bindings
/// @param[in,out] m Python module
auto init_azimuth(nanobind::module_& m) -> void;

/// @brief Initialize centroid algorithm bindings
/// @param[in,out] m Python module
auto init_centroid(nanobind::module_& m) -> void;

/// @brief Initialize convert algorithm binding
/// @param[in,out] m Python module
auto init_convert(nanobind::module_& m) -> void;

/// @brief Initialize transform algorithm binding
/// @param[in,out] m Python module
auto init_transform(nanobind::module_& m) -> void;

/// @brief Initialize closest_points algorithm bindings
/// @param[in,out] m Python module
auto init_closest_points(nanobind::module_& m) -> void;

/// @brief Initialize covered_by algorithm binding
/// @param[in,out] m Python module
auto init_covered_by(nanobind::module_& m) -> void;

/// @brief Initialize crosses algorithm binding
/// @param[in,out] m Python module
auto init_crosses(nanobind::module_& m) -> void;

/// @brief Initialize is_empty algorithm binding
/// @param[in,out] m Python module
auto init_is_empty(nanobind::module_& m) -> void;

/// @brief Initialize is_simple algorithm binding
/// @param[in,out] m Python module
auto init_is_simple(nanobind::module_& m) -> void;

/// @brief Initialize is_valid algorithm binding
/// @param[in,out] m Python module
auto init_is_valid(nanobind::module_& m) -> void;

/// @brief Initialize clear algorithm binding
/// @param[in,out] m Python module
auto init_clear(nanobind::module_& m) -> void;

/// @brief Initialize correct algorithm binding
/// @param[in,out] m Python module
auto init_correct(nanobind::module_& m) -> void;

/// @brief Initialize convex_hull algorithm binding
/// @param[in,out] m Python module
auto init_convex_hull(nanobind::module_& m) -> void;

/// @brief Initialize densify algorithm binding
/// @param[in,out] m Python module
auto init_densify(nanobind::module_& m) -> void;

/// @brief Initialize difference algorithm binding
/// @param[in,out] m Python module
auto init_difference(nanobind::module_& m) -> void;

/// @brief Initialize disjoint algorithm binding
/// @param[in,out] m Python module
auto init_disjoint(nanobind::module_& m) -> void;

/// @brief Initialize distance algorithm binding
/// @param[in,out] m Python module
auto init_distance(nanobind::module_& m) -> void;

/// @brief Initialize envelope algorithm binding
/// @param[in,out] m Python module
auto init_envelope(nanobind::module_& m) -> void;

/// @brief Initialize equals algorithm binding
/// @param[in,out] m Python module
auto init_equals(nanobind::module_& m) -> void;

/// @brief Initialize intersection algorithm binding
/// @param[in,out] m Python module
auto init_intersection(nanobind::module_& m) -> void;

/// @brief Initialize intersects algorithm binding
/// @param[in,out] m Python module
auto init_intersects(nanobind::module_& m) -> void;

/// @brief Initialize length algorithm binding
/// @param[in,out] m Python module
auto init_length(nanobind::module_& m) -> void;

/// @brief Initialize line_interpolate algorithm binding
/// @param[in,out] m Python module
auto init_line_interpolate(nanobind::module_& m) -> void;

/// @brief Initialize num_geometries algorithm binding
/// @param[in,out] m Python module
auto init_num_geometries(nanobind::module_& m) -> void;

/// @brief Initialize num_interior_rings algorithm binding
/// @param[in,out] m Python module
auto init_num_interior_rings(nanobind::module_& m) -> void;

/// @brief Initialize num_points algorithm binding
/// @param[in,out] m Python module
auto init_num_points(nanobind::module_& m) -> void;

/// @brief Initialize num_segments algorithm binding
/// @param[in,out] m Python module
auto init_num_segments(nanobind::module_& m) -> void;

/// @brief Initialize overlaps algorithm binding
/// @param[in,out] m Python module
auto init_overlaps(nanobind::module_& m) -> void;

/// @brief Initialize perimeter algorithm binding
/// @param[in,out] m Python module
auto init_perimeter(nanobind::module_& m) -> void;

/// @brief Initialize relate algorithm binding
/// @param[in,out] m Python module
auto init_relate(nanobind::module_& m) -> void;

/// @brief Initialize relation algorithm binding
/// @param[in,out] m Python module
auto init_relation(nanobind::module_& m) -> void;

/// @brief Initialize reverse algorithm binding
/// @param[in,out] m Python module
auto init_reverse(nanobind::module_& m) -> void;

/// @brief Initialize simplify algorithm binding
/// @param[in,out] m Python module
auto init_simplify(nanobind::module_& m) -> void;

/// @brief Initialize touches algorithm binding
/// @param[in,out] m Python module
auto init_touches(nanobind::module_& m) -> void;

/// @brief Initialize union algorithm binding
/// @param[in,out] m Python module
auto init_union(nanobind::module_& m) -> void;

/// @brief Initialize unique algorithm binding
/// @param[in,out] m Python module
auto init_unique(nanobind::module_& m) -> void;

/// @brief Initialize WKT algorithm binding
/// @param[in,out] m Python module
auto init_wkt(nanobind::module_& m) -> void;

/// @brief Initialize GeoJSON algorithm binding
/// @param[in,out] m Python module
auto init_geojson(nanobind::module_& m) -> void;

/// @brief Initialize within algorithm binding
/// @param[in,out] m Python module
auto init_within(nanobind::module_& m) -> void;

/// @brief Initialize for_each_point_covered_by algorithm binding
/// @param[in,out] m Python module
auto init_for_each_point_covered_by(nanobind::module_& m) -> void;

/// @brief Initialize for_each_point_distance algorithm binding
/// @param[in,out] m Python module
auto init_for_each_point_distance(nanobind::module_& m) -> void;

/// @brief Initialize for_each_point_within algorithm binding
/// @param[in,out] m Python module
auto init_for_each_point_within(nanobind::module_& m) -> void;

/// @brief Initialize all algorithm bindings
/// @param[in,out] m Python module
inline void init_algorithms(nanobind::module_& m) {
  init_strategy(m);
  init_area(m);
  init_azimuth(m);
  init_centroid(m);
  init_convert(m);
  init_transform(m);

  init_clear(m);
  init_closest_points(m);
  init_convex_hull(m);
  init_correct(m);
  init_covered_by(m);
  init_crosses(m);
  init_densify(m);
  init_difference(m);
  init_disjoint(m);
  init_distance(m);
  init_envelope(m);
  init_equals(m);
  init_intersection(m);
  init_intersects(m);
  init_is_empty(m);
  init_is_simple(m);
  init_is_valid(m);
  init_length(m);
  init_line_interpolate(m);

  init_for_each_point_covered_by(m);
  init_for_each_point_distance(m);
  init_for_each_point_within(m);
  init_geojson(m);
  init_num_geometries(m);
  init_num_interior_rings(m);
  init_num_points(m);
  init_num_segments(m);
  init_overlaps(m);
  init_perimeter(m);
  init_relate(m);
  init_relation(m);
  init_reverse(m);
  init_simplify(m);
  init_touches(m);
  init_union(m);
  init_unique(m);
  init_within(m);
  init_wkt(m);
}

}  // namespace pyinterp::geometry::geographic::pybind

namespace pyinterp::geometry::pybind {

/// @brief Initialize geographic coordinate bindings
/// @param[in,out] m Python module
void init_geographic(nanobind::module_& m);

/// @brief Initialize cartesian coordinate bindings
/// @param[in,out] m Python module
void init_cartesian(nanobind::module_& m);

}  // namespace pyinterp::geometry::pybind

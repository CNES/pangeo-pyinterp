# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for measurement algorithms (geographic)."""

from __future__ import annotations

import numpy as np

from .......core.geometry.geographic import (
    Box,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    Ring,
)
from .......core.geometry.geographic.algorithms import (
    curvilinear_distance,
    envelope,
    length,
    num_geometries,
    num_interior_rings,
    num_points,
    num_segments,
    perimeter,
)


# Length tests (geographic uses geodesic calculations)
def test_length_linestring() -> None:
    """Test length of a linestring."""
    lon = np.array([0.0, 1.0])
    lat = np.array([0.0, 0.0])
    linestring = LineString(lon, lat)

    result = length(linestring)
    # Should return non-zero distance in meters
    assert result > 0.0


def test_length_polygon() -> None:
    """Test length of polygon (should be 0)."""
    lon = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    lat = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    polygon = Polygon(Ring(lon, lat))

    result = length(polygon)
    assert result == 0.0


def test_length_multilinestring() -> None:
    """Test length of multilinestring."""
    lon1 = np.array([0.0, 1.0])
    lat1 = np.array([0.0, 0.0])
    line1 = LineString(lon1, lat1)

    lon2 = np.array([0.0, 0.0])
    lat2 = np.array([0.0, 1.0])
    line2 = LineString(lon2, lat2)

    multilinestring = MultiLineString([line1, line2])

    result = length(multilinestring)
    assert result > 0.0


def test_length_empty_linestring() -> None:
    """Test length of empty linestring."""
    linestring = LineString()

    result = length(linestring)
    assert result == 0.0


# Curvilinear distance tests
def test_curvilinear_distance_linestring() -> None:
    """Test curvilinear distance along a linestring."""
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([0.0, 0.0, 0.0])
    linestring = LineString(lon, lat)

    result = curvilinear_distance(linestring)

    # Should return an array of cumulative distances
    assert isinstance(result, np.ndarray)
    assert len(result) == len(lon)
    # First point should be at distance 0
    assert result[0] == 0.0
    # Subsequent distances should be monotonically increasing
    assert np.all(np.diff(result) >= 0.0)
    # Last distance should equal the total length
    assert np.isclose(result[-1], length(linestring))


def test_curvilinear_distance_ring() -> None:
    """Test curvilinear distance along a ring."""
    lon = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    lat = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    ring = Ring(lon, lat)

    result = curvilinear_distance(ring)

    # Should return an array of cumulative distances
    assert isinstance(result, np.ndarray)
    assert len(result) == len(lon)
    # First point should be at distance 0
    assert result[0] == 0.0
    # Subsequent distances should be monotonically increasing
    assert np.all(np.diff(result) >= 0.0)
    # Last distance should equal the perimeter
    assert np.isclose(result[-1], perimeter(ring))


def test_curvilinear_distance_empty_linestring() -> None:
    """Test curvilinear distance on empty linestring."""
    linestring = LineString()

    result = curvilinear_distance(linestring)

    # Should return an empty array
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


# Perimeter tests
def test_perimeter_polygon(polygon_1x1: Polygon) -> None:
    """Test perimeter of a polygon."""
    result = perimeter(polygon_1x1)

    # Should return non-zero perimeter in meters
    assert result > 0.0


def test_perimeter_polygon_10x10(polygon_10x10: Polygon) -> None:
    """Test perimeter of a larger polygon."""
    result = perimeter(polygon_10x10)

    assert result > 0.0


def test_perimeter_polygon_with_hole(polygon_with_hole: Polygon) -> None:
    """Test perimeter includes inner rings."""
    result = perimeter(polygon_with_hole)

    assert result > 0.0


def test_perimeter_multipolygon(multipolygon_simple: MultiPolygon) -> None:
    """Test perimeter of multipolygon."""
    result = perimeter(multipolygon_simple)

    assert result > 0.0


# Envelope tests
def test_envelope_point() -> None:
    """Test envelope of a point."""
    point = Point(5.0, 7.0)

    box = envelope(point)

    assert isinstance(box, Box)
    min_corner = box.min_corner
    max_corner = box.max_corner
    assert min_corner.lon == 5.0
    assert min_corner.lat == 7.0
    assert max_corner.lon == 5.0
    assert max_corner.lat == 7.0


def test_envelope_linestring() -> None:
    """Test envelope of a linestring."""
    lon = np.array([0.0, 5.0, 10.0])
    lat = np.array([0.0, 10.0, 0.0])
    linestring = LineString(lon, lat)

    box = envelope(linestring)

    assert isinstance(box, Box)
    min_corner = box.min_corner
    max_corner = box.max_corner
    assert min_corner.lon == 0.0
    assert min_corner.lat == 0.0
    assert max_corner.lon == 10.0
    assert max_corner.lat == 10.0


def test_envelope_polygon(polygon_10x10: Polygon) -> None:
    """Test envelope of a polygon."""
    box = envelope(polygon_10x10)

    assert isinstance(box, Box)


def test_envelope_multipolygon(multipolygon_simple: MultiPolygon) -> None:
    """Test envelope of multipolygon."""
    box = envelope(multipolygon_simple)

    assert isinstance(box, Box)


# Num_points tests
def test_num_points_point() -> None:
    """Test num_points for a single point."""
    point = Point(5.0, 5.0)

    count = num_points(point)
    assert count == 1


def test_num_points_linestring(linestring_basic: LineString) -> None:
    """Test num_points for a linestring."""
    count = num_points(linestring_basic)

    assert count == len(linestring_basic)


def test_num_points_polygon(polygon_1x1: Polygon) -> None:
    """Test num_points for a polygon."""
    count = num_points(polygon_1x1)

    assert count == len(polygon_1x1.outer)


def test_num_points_polygon_with_hole(polygon_with_hole: Polygon) -> None:
    """Test num_points for polygon with hole."""
    count = num_points(polygon_with_hole)

    expected = len(polygon_with_hole.outer) + sum(
        len(inner) for inner in polygon_with_hole.inners
    )
    assert count == expected


# Num_segments tests
def test_num_segments_linestring() -> None:
    """Test num_segments for linestring."""
    lon = np.array([0.0, 5.0, 10.0])
    lat = np.array([0.0, 5.0, 0.0])
    linestring = LineString(lon, lat)

    count = num_segments(linestring)
    assert count == 2


def test_num_segments_ring() -> None:
    """Test num_segments for ring."""
    lon = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    lat = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    ring = Ring(lon, lat)

    count = num_segments(ring)
    assert count == 4


def test_num_segments_polygon(polygon_1x1: Polygon) -> None:
    """Test num_segments for polygon."""
    count = num_segments(polygon_1x1)

    assert count >= 4


# Num_interior_rings tests
def test_num_interior_rings_simple_polygon(polygon_1x1: Polygon) -> None:
    """Test num_interior_rings for simple polygon."""
    count = num_interior_rings(polygon_1x1)

    assert count == 0


def test_num_interior_rings_polygon_with_hole(
    polygon_with_hole: Polygon,
) -> None:
    """Test num_interior_rings for polygon with hole."""
    count = num_interior_rings(polygon_with_hole)

    assert count == 1


def test_num_interior_rings_multipolygon(
    multipolygon_simple: MultiPolygon,
) -> None:
    """Test num_interior_rings for multipolygon."""
    count = num_interior_rings(multipolygon_simple)

    assert count == 0


# Num_geometries tests
def test_num_geometries_multipolygon(
    multipolygon_simple: MultiPolygon,
) -> None:
    """Test num_geometries for multipolygon."""
    count = num_geometries(multipolygon_simple)

    assert count == 2


def test_num_geometries_multilinestring(
    multilinestring_basic: MultiLineString,
) -> None:
    """Test num_geometries for multilinestring."""
    count = num_geometries(multilinestring_basic)

    assert count == 2


def test_num_geometries_simple_geometry(polygon_1x1: Polygon) -> None:
    """Test num_geometries for simple geometry."""
    count = num_geometries(polygon_1x1)

    assert count == 1


def test_num_geometries_empty_multipolygon() -> None:
    """Test num_geometries for empty multipolygon."""
    multipolygon = MultiPolygon()

    count = num_geometries(multipolygon)
    assert count == 0

# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for measurement algorithms."""

from __future__ import annotations

import numpy as np

from .......core.geometry.cartesian import (
    Box,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    Ring,
)
from .......core.geometry.cartesian.algorithms import (
    envelope,
    length,
    num_geometries,
    num_interior_rings,
    num_points,
    num_segments,
    perimeter,
)


# Length tests
def test_length_linestring() -> None:
    """Test length of a linestring."""
    # Horizontal line from 0 to 10
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 0.0])
    linestring = LineString(x, y)

    result = length(linestring)
    assert abs(result - 10.0) < 1e-10


def test_length_diagonal_linestring() -> None:
    """Test length of diagonal linestring."""
    # 3-4-5 triangle
    x = np.array([0.0, 3.0])
    y = np.array([0.0, 4.0])
    linestring = LineString(x, y)

    result = length(linestring)
    assert abs(result - 5.0) < 1e-10


def test_length_polygon() -> None:
    """Test length of polygon (should be 0)."""
    x = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    y = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    polygon = Polygon(Ring(x, y))

    result = length(polygon)
    # Polygons have zero length (area != 0)
    assert result == 0.0


def test_length_multilinestring() -> None:
    """Test length of multilinestring."""
    x1 = np.array([0.0, 10.0])
    y1 = np.array([0.0, 0.0])
    line1 = LineString(x1, y1)

    x2 = np.array([0.0, 0.0])
    y2 = np.array([0.0, 10.0])
    line2 = LineString(x2, y2)

    multilinestring = MultiLineString([line1, line2])

    result = length(multilinestring)
    # Total length should be 20
    assert abs(result - 20.0) < 1e-10


def test_length_empty_linestring() -> None:
    """Test length of empty linestring."""
    linestring = LineString()

    result = length(linestring)
    assert result == 0.0


# Perimeter tests
def test_perimeter_polygon(polygon_1x1: Polygon) -> None:
    """Test perimeter of a 1x1 square."""
    result = perimeter(polygon_1x1)

    # Perimeter of 1x1 square is 4
    assert abs(result - 4.0) < 1e-10


def test_perimeter_polygon_10x10(polygon_10x10: Polygon) -> None:
    """Test perimeter of a 10x10 square."""
    result = perimeter(polygon_10x10)

    # Perimeter of 10x10 square is 40
    assert abs(result - 40.0) < 1e-10


def test_perimeter_polygon_with_hole(polygon_with_hole: Polygon) -> None:
    """Test perimeter includes inner rings."""
    result = perimeter(polygon_with_hole)

    # Outer: 40, Inner: 24, Total: 64
    assert abs(result - 64.0) < 1e-10


def test_perimeter_multipolygon(multipolygon_simple: MultiPolygon) -> None:
    """Test perimeter of multipolygon."""
    result = perimeter(multipolygon_simple)

    # Two 1x1 squares, total perimeter = 8
    assert abs(result - 8.0) < 1e-10


def test_perimeter_point() -> None:
    """Test perimeter of point (should be 0)."""
    point = Point(5.0, 5.0)

    result = perimeter(point)
    assert result == 0.0


def test_perimeter_linestring() -> None:
    """Test perimeter of linestring (should be 0)."""
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 0.0])
    linestring = LineString(x, y)

    result = perimeter(linestring)
    # LineStrings have zero perimeter
    assert result == 0.0


# Envelope tests
def test_envelope_point() -> None:
    """Test envelope of a point."""
    point = Point(5.0, 7.0)

    box = envelope(point)

    assert isinstance(box, Box)
    # Envelope of a point is a zero-size box at that point
    min_corner = box.min_corner
    max_corner = box.max_corner
    assert min_corner.x == 5.0
    assert min_corner.y == 7.0
    assert max_corner.x == 5.0
    assert max_corner.y == 7.0


def test_envelope_linestring() -> None:
    """Test envelope of a linestring."""
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([0.0, 10.0, 0.0])
    linestring = LineString(x, y)

    box = envelope(linestring)

    assert isinstance(box, Box)
    min_corner = box.min_corner
    max_corner = box.max_corner
    assert min_corner.x == 0.0
    assert min_corner.y == 0.0
    assert max_corner.x == 10.0
    assert max_corner.y == 10.0


def test_envelope_polygon(polygon_10x10: Polygon) -> None:
    """Test envelope of a polygon."""
    box = envelope(polygon_10x10)

    assert isinstance(box, Box)
    min_corner = box.min_corner
    max_corner = box.max_corner
    assert min_corner.x == 0.0
    assert min_corner.y == 0.0
    assert max_corner.x == 10.0
    assert max_corner.y == 10.0


def test_envelope_multipolygon(multipolygon_simple: MultiPolygon) -> None:
    """Test envelope of multipolygon."""
    box = envelope(multipolygon_simple)

    assert isinstance(box, Box)
    # Should encompass all polygons


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

    # Should count outer ring points
    assert count == len(polygon_1x1.outer)


def test_num_points_polygon_with_hole(polygon_with_hole: Polygon) -> None:
    """Test num_points for polygon with hole."""
    count = num_points(polygon_with_hole)

    # Should count outer + inner ring points
    expected = len(polygon_with_hole.outer) + sum(
        len(inner) for inner in polygon_with_hole.inners
    )
    assert count == expected


def test_num_points_empty_linestring() -> None:
    """Test num_points for empty linestring."""
    linestring = LineString()

    count = num_points(linestring)
    assert count == 0


# Num_segments tests
def test_num_segments_linestring() -> None:
    """Test num_segments for linestring."""
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([0.0, 5.0, 0.0])
    linestring = LineString(x, y)

    count = num_segments(linestring)
    # 3 points = 2 segments
    assert count == 2


def test_num_segments_ring() -> None:
    """Test num_segments for ring."""
    x = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    y = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    ring = Ring(x, y)

    count = num_segments(ring)
    # 5 points forming a closed ring = 4 segments
    assert count == 4


def test_num_segments_polygon(polygon_1x1: Polygon) -> None:
    """Test num_segments for polygon."""
    count = num_segments(polygon_1x1)

    # Should count segments in outer ring
    assert count >= 4


def test_num_segments_empty() -> None:
    """Test num_segments for empty geometry."""
    linestring = LineString()

    count = num_segments(linestring)
    assert count == 0


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

    # Two simple polygons, no holes
    assert count == 0


def test_num_interior_rings_point() -> None:
    """Test num_interior_rings for point."""
    point = Point(5.0, 5.0)

    count = num_interior_rings(point)
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

    # Simple geometries return 1
    assert count == 1


def test_num_geometries_empty_multipolygon() -> None:
    """Test num_geometries for empty multipolygon."""
    multipolygon = MultiPolygon()

    count = num_geometries(multipolygon)
    assert count == 0

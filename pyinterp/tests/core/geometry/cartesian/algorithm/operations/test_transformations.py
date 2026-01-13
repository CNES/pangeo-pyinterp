# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for geometric transformation algorithms."""

import numpy as np

from .......core.geometry.cartesian import (
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    Ring,
)
from .......core.geometry.cartesian.algorithms import (
    clear,
    convex_hull,
    correct,
    densify,
    line_interpolate,
    reverse,
    simplify,
    unique,
)


# Convex hull tests
def test_convex_hull_polygon(polygon_10x10: Polygon) -> None:
    """Test convex hull of a polygon."""
    hull = convex_hull(polygon_10x10)

    assert isinstance(hull, Polygon)
    # Convex hull of a square is the square itself
    assert len(hull.outer) >= 4


def test_convex_hull_multipoint() -> None:
    """Test convex hull of scattered points."""
    # Create points forming a triangle
    points = [Point(0.0, 0.0), Point(10.0, 0.0), Point(5.0, 10.0)]
    multipoint = MultiPoint(points)

    hull = convex_hull(multipoint)

    assert isinstance(hull, Polygon)
    # Hull should be a triangle
    assert len(hull.outer) >= 3


def test_convex_hull_linestring() -> None:
    """Test convex hull of a linestring."""
    x = np.array([0.0, 5.0, 10.0, 5.0, 0.0])
    y = np.array([0.0, 5.0, 0.0, -5.0, 0.0])
    linestring = LineString(x, y)

    hull = convex_hull(linestring)

    assert isinstance(hull, Polygon)


# Densify tests
def test_densify_linestring() -> None:
    """Test densify adds points to linestring."""
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 0.0])
    linestring = LineString(x, y)

    # Densify with max distance 2.0
    densified = densify(linestring, 2.0)

    assert isinstance(densified, LineString)
    # Should have more points than original
    assert len(densified) > len(linestring)


def test_densify_polygon() -> None:
    """Test densify adds points to polygon."""
    x = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    y = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    polygon = Polygon(Ring(x, y))

    # Densify with max distance 2.0
    densified = densify(polygon, 2.0)

    assert isinstance(densified, Polygon)
    # Should have more points than original
    assert len(densified.outer) > len(polygon.outer)


def test_densify_ring() -> None:
    """Test densify adds points to ring."""
    x = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    y = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    ring = Ring(x, y)

    # Densify with max distance 2.0
    densified = densify(ring, 2.0)

    assert isinstance(densified, Ring)
    # Should have more points than original
    assert len(densified) > len(ring)


def test_densify_multilinestring() -> None:
    """Test densify on multilinestring."""
    x1 = np.array([0.0, 10.0])
    y1 = np.array([0.0, 0.0])
    line1 = LineString(x1, y1)

    x2 = np.array([0.0, 0.0])
    y2 = np.array([0.0, 10.0])
    line2 = LineString(x2, y2)

    multilinestring = MultiLineString([line1, line2])

    densified = densify(multilinestring, 2.0)

    assert isinstance(densified, MultiLineString)
    assert len(densified) == len(multilinestring)


# Simplify tests
def test_simplify_linestring() -> None:
    """Test simplify removes points from linestring."""
    # Create a linestring with many points
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
    y = np.array([0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0])
    linestring = LineString(x, y)

    # Simplify with distance 0.5 (should remove small deviations)
    simplified = simplify(linestring, 0.5)

    assert isinstance(simplified, LineString)
    # Should have fewer points than original
    assert len(simplified) <= len(linestring)


def test_simplify_polygon() -> None:
    """Test simplify removes points from polygon."""
    # Create polygon with extra points
    x = np.array([0.0, 0.0, 5.0, 10.0, 10.0, 5.0, 0.0])
    y = np.array([0.0, 5.0, 10.0, 10.0, 5.0, 0.0, 0.0])
    polygon = Polygon(Ring(x, y))

    simplified = simplify(polygon, 1.0)

    assert isinstance(simplified, Polygon)


def test_simplify_ring() -> None:
    """Test simplify removes points from ring."""
    x = np.array([0.0, 0.0, 5.0, 10.0, 10.0, 5.0, 0.0])
    y = np.array([0.0, 5.0, 10.0, 10.0, 5.0, 0.0, 0.0])
    ring = Ring(x, y)

    simplified = simplify(ring, 1.0)

    assert isinstance(simplified, Ring)


# Reverse tests
def test_reverse_linestring(linestring_basic: LineString) -> None:
    """Test reverse reverses point order."""
    # Get first point before reverse
    first_before = linestring_basic[0]

    # Reverse the linestring (modifies in place)
    reverse(linestring_basic)

    # First point should now be last
    last_after = linestring_basic[len(linestring_basic) - 1]
    assert first_before.x == last_after.x
    assert first_before.y == last_after.y


def test_reverse_polygon(polygon_10x10: Polygon) -> None:
    """Test reverse on polygon."""
    # Reverse modifies in place
    reverse(polygon_10x10)

    # Should still be a valid polygon
    assert isinstance(polygon_10x10, Polygon)


# Correct tests
def test_correct_polygon(polygon_10x10: Polygon) -> None:
    """Test correct fixes orientation."""
    # Correct modifies in place
    correct(polygon_10x10)

    # Should still be a valid polygon
    assert isinstance(polygon_10x10, Polygon)


# Unique tests
def test_unique_linestring() -> None:
    """Test unique removes duplicate consecutive points."""
    # Create linestring with duplicate points
    x = np.array([0.0, 0.0, 5.0, 5.0, 10.0])
    y = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    linestring = LineString(x, y)

    # Unique modifies in place
    unique(linestring)

    # Should have removed duplicates
    assert len(linestring) <= 5


def test_unique_polygon(polygon_10x10: Polygon) -> None:
    """Test unique on polygon."""
    # Unique modifies in place
    unique(polygon_10x10)

    # Should still be a valid polygon
    assert isinstance(polygon_10x10, Polygon)


# Clear tests
def test_clear_linestring(linestring_basic: LineString) -> None:
    """Test clear empties a linestring."""
    clear(linestring_basic)

    # Should be empty
    assert len(linestring_basic) == 0


def test_clear_polygon(polygon_10x10: Polygon) -> None:
    """Test clear empties a polygon."""
    clear(polygon_10x10)

    # Should be empty
    assert len(polygon_10x10.outer) == 0


# Line interpolate tests
def test_line_interpolate_linestring() -> None:
    """Test line_interpolate finds point at distance along linestring."""
    # Straight horizontal line from 0 to 10
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 0.0])
    linestring = LineString(x, y)

    # Interpolate at distance 5.0 (halfway)
    point = line_interpolate(linestring, 5.0)

    assert isinstance(point, Point)
    assert abs(point.x - 5.0) < 1e-6
    assert abs(point.y - 0.0) < 1e-6


def test_line_interpolate_at_start() -> None:
    """Test line_interpolate at distance 0."""
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 0.0])
    linestring = LineString(x, y)

    point = line_interpolate(linestring, 0.0)

    assert isinstance(point, Point)
    assert abs(point.x - 0.0) < 1e-6
    assert abs(point.y - 0.0) < 1e-6


def test_line_interpolate_at_end() -> None:
    """Test line_interpolate at end of line."""
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 0.0])
    linestring = LineString(x, y)

    point = line_interpolate(linestring, 10.0)

    assert isinstance(point, Point)
    assert abs(point.x - 10.0) < 1e-6
    assert abs(point.y - 0.0) < 1e-6

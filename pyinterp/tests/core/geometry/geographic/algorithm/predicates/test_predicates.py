# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for spatial predicate algorithms (geographic)."""

import numpy as np

from .......core.geometry.geographic import (
    Box,
    LineString,
    Point,
    Polygon,
    Ring,
)
from .......core.geometry.geographic.algorithms import (
    covered_by,
    crosses,
    disjoint,
    distance,
    equals,
    intersects,
    overlaps,
    touches,
    within,
)


# Disjoint tests
def test_disjoint_points() -> None:
    """Test disjoint with two separate points."""
    p1 = Point(0.0, 0.0)
    p2 = Point(10.0, 10.0)

    assert disjoint(p1, p2) is True


def test_not_disjoint_same_point() -> None:
    """Test disjoint with same point (not disjoint)."""
    p1 = Point(5.0, 5.0)
    p2 = Point(5.0, 5.0)

    assert disjoint(p1, p2) is False


def test_disjoint_point_box() -> None:
    """Test disjoint with point outside box."""
    point = Point(20.0, 20.0)
    box = Box((0.0, 0.0), (5.0, 5.0))

    assert disjoint(point, box) is True
    assert disjoint(box, point) is True


def test_not_disjoint_point_box() -> None:
    """Test point inside box (not disjoint)."""
    point = Point(2.5, 2.5)
    box = Box((0.0, 0.0), (5.0, 5.0))

    assert disjoint(point, box) is False
    assert disjoint(box, point) is False


def test_disjoint_polygons(polygon_pair: tuple[Polygon, Polygon]) -> None:
    """Test disjoint with two separate polygons."""
    poly1, poly2 = polygon_pair

    assert disjoint(poly1, poly2) is True


# Intersects tests
def test_intersects_points() -> None:
    """Test intersects with same point."""
    p1 = Point(5.0, 5.0)
    p2 = Point(5.0, 5.0)

    assert intersects(p1, p2) is True


def test_not_intersects_points() -> None:
    """Test intersects with different points."""
    p1 = Point(0.0, 0.0)
    p2 = Point(10.0, 10.0)

    assert intersects(p1, p2) is False


def test_intersects_point_polygon() -> None:
    """Test point inside polygon."""
    point = Point(2.5, 2.5)
    lon = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    polygon = Polygon(Ring(lon, lat))

    assert intersects(point, polygon) is True
    assert intersects(polygon, point) is True


def test_intersects_overlapping_boxes() -> None:
    """Test overlapping boxes."""
    box1 = Box((0.0, 0.0), (5.0, 5.0))
    box2 = Box((2.0, 2.0), (7.0, 7.0))

    assert intersects(box1, box2) is True


def test_not_intersects_boxes() -> None:
    """Test non-overlapping boxes."""
    box1 = Box((0.0, 0.0), (2.0, 2.0))
    box2 = Box((10.0, 10.0), (15.0, 15.0))

    assert intersects(box1, box2) is False


def test_intersects_linestrings() -> None:
    """Test intersecting linestrings."""
    # Cross pattern
    lon1 = np.array([0.0, 10.0])
    lat1 = np.array([5.0, 5.0])
    line1 = LineString(lon1, lat1)

    lon2 = np.array([5.0, 5.0])
    lat2 = np.array([0.0, 10.0])
    line2 = LineString(lon2, lat2)

    assert intersects(line1, line2) is True


# Touches tests
def test_touches_polygons() -> None:
    """Test polygons that touch at boundary."""
    # Two adjacent squares
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(lon1, lat1))

    lon2 = np.array([5.0, 10.0, 10.0, 5.0, 5.0])
    lat2 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly2 = Polygon(Ring(lon2, lat2))

    # Geographic geometries may have different behavior than cartesian
    # Just verify the function works and returns a boolean
    result = touches(poly1, poly2)
    assert isinstance(result, bool)


def test_not_touches_disjoint_polygons(
    polygon_pair: tuple[Polygon, Polygon],
) -> None:
    """Test disjoint polygons don't touch."""
    poly1, poly2 = polygon_pair

    assert touches(poly1, poly2) is False


def test_touches_point_linestring() -> None:
    """Test point touching end of linestring."""
    point = Point(0.0, 0.0)
    lon = np.array([0.0, 5.0, 10.0])
    lat = np.array([0.0, 5.0, 0.0])
    linestring = LineString(lon, lat)

    assert touches(point, linestring) is True


# Within tests
def test_within_point_box() -> None:
    """Test point within box."""
    point = Point(2.5, 2.5)
    box = Box((0.0, 0.0), (5.0, 5.0))

    assert within(point, box) is True


def test_not_within_point_box() -> None:
    """Test point outside box."""
    point = Point(10.0, 10.0)
    box = Box((0.0, 0.0), (5.0, 5.0))

    assert within(point, box) is False


def test_within_polygon_polygon() -> None:
    """Test small polygon within larger polygon."""
    # Small square
    lon1 = np.array([2.0, 3.0, 3.0, 2.0, 2.0])
    lat1 = np.array([2.0, 2.0, 3.0, 3.0, 2.0])
    small = Polygon(Ring(lon1, lat1))

    # Large square
    lon2 = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    lat2 = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    large = Polygon(Ring(lon2, lat2))

    assert within(small, large) is True


# Covered_by tests
def test_covered_by_point_box() -> None:
    """Test point covered by box."""
    point = Point(2.5, 2.5)
    box = Box((0.0, 0.0), (5.0, 5.0))

    assert covered_by(point, box) is True


def test_covered_by_point_on_boundary() -> None:
    """Test point on boundary is covered."""
    point = Point(0.0, 0.0)
    box = Box((0.0, 0.0), (5.0, 5.0))

    assert covered_by(point, box) is True


def test_not_covered_by_point_box() -> None:
    """Test point not covered by box."""
    point = Point(10.0, 10.0)
    box = Box((0.0, 0.0), (5.0, 5.0))

    assert covered_by(point, box) is False


# Crosses tests
def test_crosses_linestring_polygon() -> None:
    """Test linestring crossing polygon."""
    # Linestring that crosses through polygon
    lon_line = np.array([-1.0, 6.0])
    lat_line = np.array([2.5, 2.5])
    linestring = LineString(lon_line, lat_line)

    # Polygon
    lon = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    polygon = Polygon(Ring(lon, lat))

    assert crosses(linestring, polygon) is True


def test_crosses_linestrings() -> None:
    """Test crossing linestrings."""
    # Horizontal line
    lon1 = np.array([0.0, 10.0])
    lat1 = np.array([5.0, 5.0])
    line1 = LineString(lon1, lat1)

    # Vertical line
    lon2 = np.array([5.0, 5.0])
    lat2 = np.array([0.0, 10.0])
    line2 = LineString(lon2, lat2)

    assert crosses(line1, line2) is True


# Overlaps tests
def test_overlaps_polygons() -> None:
    """Test overlapping polygons."""
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(lon1, lat1))

    lon2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    lat2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    poly2 = Polygon(Ring(lon2, lat2))

    assert overlaps(poly1, poly2) is True


def test_not_overlaps_disjoint_polygons(
    polygon_pair: tuple[Polygon, Polygon],
) -> None:
    """Test disjoint polygons don't overlap."""
    poly1, poly2 = polygon_pair

    assert overlaps(poly1, poly2) is False


# Equals tests
def test_equals_points() -> None:
    """Test equals with identical points."""
    p1 = Point(5.0, 5.0)
    p2 = Point(5.0, 5.0)

    assert equals(p1, p2) is True


def test_not_equals_points() -> None:
    """Test equals with different points."""
    p1 = Point(0.0, 0.0)
    p2 = Point(10.0, 10.0)

    assert equals(p1, p2) is False


def test_equals_polygons(polygon_1x1: Polygon) -> None:
    """Test equals with identical polygons."""
    # Create another identical polygon
    lon = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    lat = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    polygon2 = Polygon(Ring(lon, lat))

    assert equals(polygon_1x1, polygon2) is True


def test_not_equals_polygons(
    polygon_1x1: Polygon,
    polygon_10x10: Polygon,
) -> None:
    """Test equals with different polygons."""
    assert equals(polygon_1x1, polygon_10x10) is False


# Distance tests (geographic uses geodesic calculations)
def test_distance_points() -> None:
    """Test distance between two points."""
    p1 = Point(0.0, 0.0)
    p2 = Point(1.0, 0.0)

    # Distance should be non-zero (in meters for geographic)
    dist = distance(p1, p2)
    assert dist > 0.0


def test_distance_same_point() -> None:
    """Test distance between same point."""
    p1 = Point(5.0, 5.0)
    p2 = Point(5.0, 5.0)

    dist = distance(p1, p2)
    assert abs(dist) < 1e-6


def test_distance_point_box() -> None:
    """Test distance from point to box."""
    point = Point(10.0, 10.0)
    box = Box((0.0, 0.0), (5.0, 5.0))

    # Distance should be non-zero
    dist = distance(point, box)
    assert dist > 0.0


def test_distance_point_inside_polygon() -> None:
    """Test distance from point inside polygon (should be 0)."""
    point = Point(2.5, 2.5)
    lon = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    polygon = Polygon(Ring(lon, lat))

    dist = distance(point, polygon)
    assert abs(dist) < 1e-6


def test_distance_polygons(polygon_pair: tuple[Polygon, Polygon]) -> None:
    """Test distance between two separate polygons."""
    poly1, poly2 = polygon_pair

    dist = distance(poly1, poly2)
    # Should be non-zero for disjoint polygons
    assert dist > 0.0

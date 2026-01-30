# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for spatial predicate algorithms."""

from __future__ import annotations

import numpy as np

from .......core.geometry.cartesian import (
    Box,
    LineString,
    Point,
    Polygon,
    Ring,
)
from .......core.geometry.cartesian.algorithms import (
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
    point = Point(10.0, 10.0)
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


def test_not_disjoint_overlapping_polygons() -> None:
    """Test overlapping polygons (not disjoint)."""
    x1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    y1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    x2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    y2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    poly2 = Polygon(Ring(x2, y2))

    assert disjoint(poly1, poly2) is False


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
    x = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    y = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    polygon = Polygon(Ring(x, y))

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
    box2 = Box((5.0, 5.0), (7.0, 7.0))

    assert intersects(box1, box2) is False


def test_intersects_linestrings() -> None:
    """Test intersecting linestrings."""
    # Cross pattern
    x1 = np.array([0.0, 10.0])
    y1 = np.array([5.0, 5.0])
    line1 = LineString(x1, y1)

    x2 = np.array([5.0, 5.0])
    y2 = np.array([0.0, 10.0])
    line2 = LineString(x2, y2)

    assert intersects(line1, line2) is True


# Touches tests
def test_touches_polygons() -> None:
    """Test polygons that touch at boundary."""
    # Two adjacent squares
    x1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    y1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    x2 = np.array([5.0, 5.0, 10.0, 10.0, 5.0])
    y2 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    poly2 = Polygon(Ring(x2, y2))

    assert touches(poly1, poly2) is True


def test_not_touches_disjoint_polygons(
    polygon_pair: tuple[Polygon, Polygon],
) -> None:
    """Test disjoint polygons don't touch."""
    poly1, poly2 = polygon_pair

    assert touches(poly1, poly2) is False


def test_touches_point_linestring() -> None:
    """Test point touching end of linestring."""
    point = Point(0.0, 0.0)
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([0.0, 5.0, 0.0])
    linestring = LineString(x, y)

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
    x1 = np.array([2.0, 2.0, 3.0, 3.0, 2.0])
    y1 = np.array([2.0, 3.0, 3.0, 2.0, 2.0])
    small = Polygon(Ring(x1, y1))

    # Large square
    x2 = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    y2 = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    large = Polygon(Ring(x2, y2))

    assert within(small, large) is True


def test_not_within_polygon_polygon() -> None:
    """Test polygon not within another."""
    x1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    y1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    x2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    y2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    poly2 = Polygon(Ring(x2, y2))

    assert within(poly1, poly2) is False


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
    x_line = np.array([-1.0, 6.0])
    y_line = np.array([2.5, 2.5])
    linestring = LineString(x_line, y_line)

    # Polygon
    x = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    y = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    polygon = Polygon(Ring(x, y))

    assert crosses(linestring, polygon) is True


def test_crosses_linestrings() -> None:
    """Test crossing linestrings."""
    # Horizontal line
    x1 = np.array([0.0, 10.0])
    y1 = np.array([5.0, 5.0])
    line1 = LineString(x1, y1)

    # Vertical line
    x2 = np.array([5.0, 5.0])
    y2 = np.array([0.0, 10.0])
    line2 = LineString(x2, y2)

    assert crosses(line1, line2) is True


# Overlaps tests
def test_overlaps_polygons() -> None:
    """Test overlapping polygons."""
    x1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    y1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    x2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    y2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    poly2 = Polygon(Ring(x2, y2))

    assert overlaps(poly1, poly2) is True


def test_not_overlaps_disjoint_polygons(
    polygon_pair: tuple[Polygon, Polygon],
) -> None:
    """Test disjoint polygons don't overlap."""
    poly1, poly2 = polygon_pair

    assert overlaps(poly1, poly2) is False


def test_not_overlaps_contained_polygon() -> None:
    """Test contained polygon doesn't overlap (one within other)."""
    # Small square
    x1 = np.array([2.0, 2.0, 3.0, 3.0, 2.0])
    y1 = np.array([2.0, 3.0, 3.0, 2.0, 2.0])
    small = Polygon(Ring(x1, y1))

    # Large square
    x2 = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    y2 = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    large = Polygon(Ring(x2, y2))

    # Within doesn't count as overlap
    assert overlaps(small, large) is False


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
    x = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    polygon2 = Polygon(Ring(x, y))

    assert equals(polygon_1x1, polygon2) is True


def test_not_equals_polygons(
    polygon_1x1: Polygon,
    polygon_10x10: Polygon,
) -> None:
    """Test equals with different polygons."""
    assert equals(polygon_1x1, polygon_10x10) is False


# Distance tests
def test_distance_points() -> None:
    """Test distance between two points."""
    p1 = Point(0.0, 0.0)
    p2 = Point(3.0, 4.0)

    # Distance should be 5 (3-4-5 triangle)
    dist = distance(p1, p2)
    assert abs(dist - 5.0) < 1e-10


def test_distance_same_point() -> None:
    """Test distance between same point."""
    p1 = Point(5.0, 5.0)
    p2 = Point(5.0, 5.0)

    dist = distance(p1, p2)
    assert dist == 0.0


def test_distance_point_box() -> None:
    """Test distance from point to box."""
    point = Point(10.0, 10.0)
    box = Box((0.0, 0.0), (5.0, 5.0))

    # Distance to nearest corner (5, 5)
    dist = distance(point, box)
    expected = np.sqrt(25.0 + 25.0)  # sqrt((10-5)^2 + (10-5)^2)
    assert abs(dist - expected) < 1e-10


def test_distance_point_inside_polygon() -> None:
    """Test distance from point inside polygon (should be 0)."""
    point = Point(2.5, 2.5)
    x = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    y = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    polygon = Polygon(Ring(x, y))

    dist = distance(point, polygon)
    assert dist == 0.0


def test_distance_polygons(polygon_pair: tuple[Polygon, Polygon]) -> None:
    """Test distance between two separate polygons."""
    poly1, poly2 = polygon_pair

    dist = distance(poly1, poly2)
    # Should be non-zero for disjoint polygons
    assert dist > 0.0

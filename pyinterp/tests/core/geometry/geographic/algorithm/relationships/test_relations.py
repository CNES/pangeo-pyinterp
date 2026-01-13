# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for DE-9IM relation functions (geographic)."""

import numpy as np

from .......core.geometry.geographic import LineString, Point, Polygon, Ring
from .......core.geometry.geographic.algorithms import relate, relation


# Relate tests (DE-9IM pattern matching)
def test_relate_identical_points() -> None:
    """Test relate with identical points using DE-9IM pattern."""
    p1 = Point(5.0, 5.0)
    p2 = Point(5.0, 5.0)

    # Pattern for equality: "0FFFFFFF2"
    result = relate(p1, p2, "0FFFFFFF2")

    assert isinstance(result, bool)
    assert result is True


def test_relate_different_points() -> None:
    """Test relate with different points."""
    p1 = Point(0.0, 0.0)
    p2 = Point(10.0, 10.0)

    # Pattern for disjoint: "FF*FF****"
    result = relate(p1, p2, "FF*FF****")

    assert isinstance(result, bool)
    assert result is True


def test_relate_linestrings_cross() -> None:
    """Test relate with crossing linestrings."""
    # Horizontal line
    lon1 = np.array([0.0, 10.0])
    lat1 = np.array([5.0, 5.0])
    line1 = LineString(lon1, lat1)

    # Vertical line
    lon2 = np.array([5.0, 5.0])
    lat2 = np.array([0.0, 10.0])
    line2 = LineString(lon2, lat2)

    # Pattern for crosses: "0********"
    result = relate(line1, line2, "0********")

    assert isinstance(result, bool)


def test_relate_overlapping_polygons() -> None:
    """Test relate with overlapping polygons."""
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(lon1, lat1))

    lon2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    lat2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    poly2 = Polygon(Ring(lon2, lat2))

    # Pattern for overlaps: "T*T***T**"
    result = relate(poly1, poly2, "T*T***T**")

    assert isinstance(result, bool)


def test_relate_ring_polygon() -> None:
    """Test relate with ring and polygon."""
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    ring = Ring(lon1, lat1)

    lon2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    lat2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    polygon = Polygon(Ring(lon2, lat2))

    # Use wildcard pattern
    result = relate(ring, polygon, "*********")

    assert isinstance(result, bool)
    # Any geometry should match the wildcard pattern
    assert result is True


# Relation tests (get DE-9IM matrix string)
def test_relation_identical_points() -> None:
    """Test relation returns DE-9IM matrix for identical points."""
    p1 = Point(5.0, 5.0)
    p2 = Point(5.0, 5.0)

    matrix = relation(p1, p2)

    assert isinstance(matrix, str)
    # DE-9IM matrix should be 9 characters
    assert len(matrix) == 9


def test_relation_different_points() -> None:
    """Test relation for different points."""
    p1 = Point(0.0, 0.0)
    p2 = Point(10.0, 10.0)

    matrix = relation(p1, p2)

    assert isinstance(matrix, str)
    assert len(matrix) == 9


def test_relation_linestrings() -> None:
    """Test relation for linestrings."""
    lon1 = np.array([0.0, 10.0])
    lat1 = np.array([0.0, 0.0])
    line1 = LineString(lon1, lat1)

    lon2 = np.array([5.0, 5.0])
    lat2 = np.array([0.0, 10.0])
    line2 = LineString(lon2, lat2)

    matrix = relation(line1, line2)

    assert isinstance(matrix, str)
    assert len(matrix) == 9


def test_relation_polygons() -> None:
    """Test relation for polygons."""
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(lon1, lat1))

    lon2 = np.array([5.0, 10.0, 10.0, 5.0, 5.0])
    lat2 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly2 = Polygon(Ring(lon2, lat2))

    matrix = relation(poly1, poly2)

    assert isinstance(matrix, str)
    assert len(matrix) == 9


def test_relation_overlapping_polygons() -> None:
    """Test relation for overlapping polygons."""
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(lon1, lat1))

    lon2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    lat2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    poly2 = Polygon(Ring(lon2, lat2))

    matrix = relation(poly1, poly2)

    assert isinstance(matrix, str)
    assert len(matrix) == 9


def test_relation_ring_ring() -> None:
    """Test relation for two rings."""
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    ring1 = Ring(lon1, lat1)

    lon2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    lat2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    ring2 = Ring(lon2, lat2)

    matrix = relation(ring1, ring2)

    assert isinstance(matrix, str)
    assert len(matrix) == 9


def test_relation_ring_polygon() -> None:
    """Test relation for ring and polygon."""
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    ring = Ring(lon1, lat1)

    lon2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    lat2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    polygon = Polygon(Ring(lon2, lat2))

    matrix = relation(ring, polygon)

    assert isinstance(matrix, str)
    assert len(matrix) == 9


def test_relation_polygon_ring() -> None:
    """Test relation for polygon and ring."""
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    polygon = Polygon(Ring(lon1, lat1))

    lon2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    lat2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    ring = Ring(lon2, lat2)

    matrix = relation(polygon, ring)

    assert isinstance(matrix, str)
    assert len(matrix) == 9

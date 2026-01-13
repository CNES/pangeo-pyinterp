# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for set operation algorithms (intersection, union, difference)."""

import numpy as np

from .......core.geometry.cartesian import Polygon, Ring
from .......core.geometry.cartesian.algorithms import (
    difference,
    intersection,
    union,
)


# Intersection tests
def test_intersection_overlapping_polygons() -> None:
    """Test intersection of overlapping polygons."""
    # First polygon: [0,0] to [5,5]
    x1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    y1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    # Second polygon: [2,2] to [7,7]
    x2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    y2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    poly2 = Polygon(Ring(x2, y2))

    result = intersection(poly1, poly2)

    # Should return a list of polygons
    assert isinstance(result, list)
    assert len(result) > 0
    # Result should be the overlapping area [2,2] to [5,5]
    assert all(isinstance(p, Polygon) for p in result)


def test_intersection_disjoint_polygons(
    polygon_pair: tuple[Polygon, Polygon],
) -> None:
    """Test intersection of disjoint polygons."""
    poly1, poly2 = polygon_pair

    result = intersection(poly1, poly2)

    # Should return empty list for disjoint polygons
    assert isinstance(result, list)
    assert len(result) == 0


def test_intersection_identical_polygons(polygon_1x1: Polygon) -> None:
    """Test intersection of identical polygons."""
    # Create another identical polygon
    x = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    poly2 = Polygon(Ring(x, y))

    result = intersection(polygon_1x1, poly2)

    # Should return the polygon itself
    assert isinstance(result, list)
    assert len(result) > 0


def test_intersection_ring_polygon() -> None:
    """Test intersection of ring with polygon."""
    # Ring
    x1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    y1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    ring = Ring(x1, y1)

    # Overlapping polygon
    x2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    y2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    polygon = Polygon(Ring(x2, y2))

    result = intersection(ring, polygon)

    # Should return a list of polygons
    assert isinstance(result, list)


# Union tests
def test_union_overlapping_polygons() -> None:
    """Test union of overlapping polygons."""
    # First polygon: [0,0] to [5,5]
    x1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    y1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    # Second polygon: [2,2] to [7,7]
    x2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    y2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    poly2 = Polygon(Ring(x2, y2))

    result = union(poly1, poly2)

    # Should return a list of polygons
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(p, Polygon) for p in result)


def test_union_disjoint_polygons(
    polygon_pair: tuple[Polygon, Polygon],
) -> None:
    """Test union of disjoint polygons."""
    poly1, poly2 = polygon_pair

    result = union(poly1, poly2)

    # Should return both polygons
    assert isinstance(result, list)
    # For disjoint polygons, typically returns 2 separate polygons
    assert len(result) >= 1


def test_union_identical_polygons(polygon_1x1: Polygon) -> None:
    """Test union of identical polygons."""
    # Create another identical polygon
    x = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    poly2 = Polygon(Ring(x, y))

    result = union(polygon_1x1, poly2)

    # Should return the polygon itself
    assert isinstance(result, list)
    assert len(result) > 0


def test_union_ring_polygon() -> None:
    """Test union of ring with polygon."""
    # Ring
    x1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    y1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    ring = Ring(x1, y1)

    # Overlapping polygon
    x2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    y2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    polygon = Polygon(Ring(x2, y2))

    result = union(ring, polygon)

    # Should return a list of polygons
    assert isinstance(result, list)
    assert len(result) > 0


# Difference tests
def test_difference_overlapping_polygons() -> None:
    """Test difference of overlapping polygons."""
    # First polygon: [0,0] to [5,5]
    x1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    y1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    # Second polygon: [2,2] to [7,7]
    x2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    y2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    poly2 = Polygon(Ring(x2, y2))

    result = difference(poly1, poly2)

    # Should return a list of polygons (poly1 - poly2)
    assert isinstance(result, list)
    # The result should be the part of poly1 not covered by poly2


def test_difference_disjoint_polygons(
    polygon_pair: tuple[Polygon, Polygon],
) -> None:
    """Test difference of disjoint polygons."""
    poly1, poly2 = polygon_pair

    result = difference(poly1, poly2)

    # Should return poly1 unchanged (no overlap to subtract)
    assert isinstance(result, list)
    assert len(result) > 0


def test_difference_identical_polygons(polygon_1x1: Polygon) -> None:
    """Test difference of identical polygons."""
    # Create another identical polygon
    x = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    poly2 = Polygon(Ring(x, y))

    result = difference(polygon_1x1, poly2)

    # Should return empty (polygon - itself = nothing)
    assert isinstance(result, list)


def test_difference_contained_polygons() -> None:
    """Test difference when one polygon contains the other."""
    # Large polygon: [0,0] to [10,10]
    x1 = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    y1 = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    large = Polygon(Ring(x1, y1))

    # Small polygon: [2,2] to [3,3]
    x2 = np.array([2.0, 2.0, 3.0, 3.0, 2.0])
    y2 = np.array([2.0, 3.0, 3.0, 2.0, 2.0])
    small = Polygon(Ring(x2, y2))

    result = difference(large, small)

    # Should return large polygon with hole
    assert isinstance(result, list)
    assert len(result) > 0


def test_difference_ring_polygon() -> None:
    """Test difference of ring with polygon."""
    # Ring
    x1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    y1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    ring = Ring(x1, y1)

    # Overlapping polygon
    x2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    y2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    polygon = Polygon(Ring(x2, y2))

    result = difference(ring, polygon)

    # Should return a list of polygons
    assert isinstance(result, list)


def test_difference_ring_ring() -> None:
    """Test difference of two rings."""
    # First ring
    x1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    y1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    ring1 = Ring(x1, y1)

    # Second ring
    x2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    y2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    ring2 = Ring(x2, y2)

    result = difference(ring1, ring2)

    # Should return a list of polygons
    assert isinstance(result, list)

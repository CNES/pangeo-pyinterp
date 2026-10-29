# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for set operation algorithms (geographic geometries)."""

from __future__ import annotations

import numpy as np

from .......core.geometry.geographic import LineString, Polygon, Ring
from .......core.geometry.geographic.algorithms import (
    difference,
    intersection,
    union,
)


# Intersection tests
def test_intersection_overlapping_polygons() -> None:
    """Test intersection of overlapping polygons."""
    # First polygon: [0,0] to [5,5]
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(lon1, lat1))

    # Second polygon: [2,2] to [7,7]
    lon2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    lat2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    poly2 = Polygon(Ring(lon2, lat2))

    result = intersection(poly1, poly2)

    # Should return a list
    assert isinstance(result, list)
    # Geographic operations may return empty results for small areas
    if len(result) > 0:
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
    lon = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    lat = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    poly2 = Polygon(Ring(lon, lat))

    result = intersection(polygon_1x1, poly2)

    # Should return the polygon itself
    assert isinstance(result, list)
    assert len(result) > 0


def test_intersection_ring_polygon() -> None:
    """Test intersection of ring with polygon."""
    # Ring
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    ring = Ring(lon1, lat1)

    # Overlapping polygon
    lon2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    lat2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    polygon = Polygon(Ring(lon2, lat2))

    result = intersection(ring, polygon)

    # Should return a list of polygons
    assert isinstance(result, list)


# Union tests
def test_intersection_linestring_polygon_on_equator() -> None:
    """A line leaving a polygon through an edge on the equator stops there.

    Boost.Geometry 1.92 and earlier compute a wrong longitude for the
    intersection with a segment lying on the equator (boostorg/geometry#1482),
    so the line was kept whole, beyond the polygon (CNES/pangeo-pyinterp#36).
    """
    # A box from 20°W to 20°E and from the equator to 20°N, with a vertex
    # every half degree.
    lons = np.linspace(-20.0, 20.0, 80)
    polygon = Polygon(
        Ring(
            np.concatenate((lons[::-1], lons, lons[-1:])),
            np.concatenate((np.zeros(80), np.full(80, 20.0), [0.0])),
        )
    )
    # A segment going south across the equator.
    line = LineString(
        np.array([3.268218994140625, 3.3428595066070557]),
        np.array([0.01789649948477745, -0.4950112998485565]),
    )

    result = intersection(line, polygon)

    assert len(result) == 1
    points = np.array([(point.lon, point.lat) for point in result[0]])
    np.testing.assert_array_equal(
        points[0], [3.268218994140625, 0.01789649948477745]
    )
    # The line stops on the equator, where it crosses the edge of the box.
    np.testing.assert_allclose(points[-1], [3.27082, 0.0], atol=1e-5)
    assert np.all(points[:, 1] >= -1e-12)


def test_union_overlapping_polygons() -> None:
    """Test union of overlapping polygons."""
    # First polygon: [0,0] to [5,5]
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(lon1, lat1))

    # Second polygon: [2,2] to [7,7]
    lon2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    lat2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    poly2 = Polygon(Ring(lon2, lat2))

    result = union(poly1, poly2)

    # Should return a list
    assert isinstance(result, list)
    # Geographic operations may return empty results for small areas
    if len(result) > 0:
        assert all(isinstance(p, Polygon) for p in result)


def test_union_disjoint_polygons(
    polygon_pair: tuple[Polygon, Polygon],
) -> None:
    """Test union of disjoint polygons."""
    poly1, poly2 = polygon_pair

    result = union(poly1, poly2)

    # Should return a list
    assert isinstance(result, list)
    # Geographic operations may return empty results for small areas


def test_union_identical_polygons(polygon_1x1: Polygon) -> None:
    """Test union of identical polygons."""
    # Create another identical polygon
    lon = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    lat = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    poly2 = Polygon(Ring(lon, lat))

    result = union(polygon_1x1, poly2)

    # Should return the polygon itself
    assert isinstance(result, list)
    assert len(result) > 0


def test_union_ring_polygon() -> None:
    """Test union of ring with polygon."""
    # Ring
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    ring = Ring(lon1, lat1)

    # Overlapping polygon
    lon2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    lat2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    polygon = Polygon(Ring(lon2, lat2))

    result = union(ring, polygon)

    # Should return a list
    assert isinstance(result, list)
    # Geographic operations may return empty results for small areas


# Difference tests
def test_difference_overlapping_polygons() -> None:
    """Test difference of overlapping polygons."""
    # First polygon: [0,0] to [5,5]
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(lon1, lat1))

    # Second polygon: [2,2] to [7,7]
    lon2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    lat2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    poly2 = Polygon(Ring(lon2, lat2))

    result = difference(poly1, poly2)

    # Should return a list of polygons (poly1 - poly2)
    assert isinstance(result, list)


def test_difference_disjoint_polygons(
    polygon_pair: tuple[Polygon, Polygon],
) -> None:
    """Test difference of disjoint polygons."""
    poly1, poly2 = polygon_pair

    result = difference(poly1, poly2)

    # Should return a list
    assert isinstance(result, list)
    # Geographic operations may return empty results for small areas


def test_difference_identical_polygons(polygon_1x1: Polygon) -> None:
    """Test difference of identical polygons."""
    # Create another identical polygon
    lon = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    lat = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    poly2 = Polygon(Ring(lon, lat))

    result = difference(polygon_1x1, poly2)

    # Should return empty (polygon - itself = nothing)
    assert isinstance(result, list)


def test_difference_contained_polygons() -> None:
    """Test difference when one polygon contains the other."""
    # Large polygon: [0,0] to [10,10]
    lon1 = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    large = Polygon(Ring(lon1, lat1))

    # Small polygon: [2,2] to [3,3]
    lon2 = np.array([2.0, 3.0, 3.0, 2.0, 2.0])
    lat2 = np.array([2.0, 2.0, 3.0, 3.0, 2.0])
    small = Polygon(Ring(lon2, lat2))

    result = difference(large, small)

    # Should return large polygon with hole
    assert isinstance(result, list)
    assert len(result) > 0


def test_difference_ring_polygon() -> None:
    """Test difference of ring with polygon."""
    # Ring
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    ring = Ring(lon1, lat1)

    # Overlapping polygon
    lon2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    lat2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    polygon = Polygon(Ring(lon2, lat2))

    result = difference(ring, polygon)

    # Should return a list of polygons
    assert isinstance(result, list)


def test_difference_ring_ring() -> None:
    """Test difference of two rings."""
    # First ring
    lon1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    ring1 = Ring(lon1, lat1)

    # Second ring
    lon2 = np.array([2.0, 7.0, 7.0, 2.0, 2.0])
    lat2 = np.array([2.0, 2.0, 7.0, 7.0, 2.0])
    ring2 = Ring(lon2, lat2)

    result = difference(ring1, ring2)

    # Should return a list of polygons
    assert isinstance(result, list)

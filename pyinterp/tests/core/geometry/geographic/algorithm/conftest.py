# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Pytest fixtures for geodetic geometry tests."""

import numpy as np
import pytest

from ......core.geometry.geographic import (
    Box,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    Ring,
    Segment,
)


# Basic geometries
@pytest.fixture
def point_basic() -> Point:
    """Create a basic point at (1.0, 2.0)."""
    return Point(1.0, 2.0)


@pytest.fixture
def point_origin() -> Point:
    """Create a point at the origin (0.0, 0.0)."""
    return Point(0.0, 0.0)


@pytest.fixture
def point_far() -> Point:
    """Create a point at (10.0, 10.0)."""
    return Point(10.0, 10.0)


# Boxes
@pytest.fixture
def box_1x1() -> Box:
    """Create a 1x1 degree box at the origin."""
    return Box((0.0, 0.0), (1.0, 1.0))


@pytest.fixture
def box_10x10() -> Box:
    """Create a 10x10 degree box at the origin."""
    return Box((0.0, 0.0), (10.0, 10.0))


@pytest.fixture
def box_zero() -> Box:
    """Create a zero-size box (same min and max corners)."""
    return Box((0.0, 0.0), (0.0, 0.0))


@pytest.fixture
def box_polar() -> Box:
    """Create a box near the north pole."""
    return Box((0.0, 85.0), (10.0, 89.0))


@pytest.fixture
def box_equator() -> Box:
    """Create a box at the equator."""
    return Box((0.0, 0.0), (10.0, 4.0))


@pytest.fixture
def box_antimeridian() -> Box:
    """Create a box crossing the anti-meridian."""
    return Box((175.0, 0.0), (185.0, 10.0))


# Rings
@pytest.fixture
def ring_square_1x1() -> Ring:
    """Create a 1x1 degree square ring (counter-clockwise)."""
    lon = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    lat = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    return Ring(lon, lat)


@pytest.fixture
def ring_square_10x10() -> Ring:
    """Create a 10x10 degree square ring (counter-clockwise)."""
    lon = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    lat = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    return Ring(lon, lat)


@pytest.fixture
def ring_square_inner() -> Ring:
    """Create a 2-8 degree square ring for use as a hole."""
    lon = np.array([2.0, 8.0, 8.0, 2.0, 2.0])
    lat = np.array([2.0, 2.0, 8.0, 8.0, 2.0])
    return Ring(lon, lat)


@pytest.fixture
def ring_empty() -> Ring:
    """Create an empty ring."""
    return Ring(np.array([]), np.array([]))


# LineStrings
@pytest.fixture
def linestring_basic() -> LineString:
    """Create a basic linestring with 3 points."""
    lon = np.array([0.0, 5.0, 10.0])
    lat = np.array([0.0, 5.0, 0.0])
    return LineString(lon, lat)


@pytest.fixture
def linestring_simple() -> LineString:
    """Create a simple linestring."""
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([0.0, 1.0, 0.0])
    return LineString(lon, lat)


@pytest.fixture
def linestring_empty() -> LineString:
    """Create an empty linestring."""
    return LineString()


# Segments
@pytest.fixture
def segment_basic() -> Segment:
    """Create a segment from (0, 0) to (10, 10)."""
    return Segment((0.0, 0.0), (10.0, 10.0))


@pytest.fixture
def segment_simple() -> Segment:
    """Create a simple segment from (0, 0) to (1, 1)."""
    return Segment((0.0, 0.0), (1.0, 1.0))


# Polygons
@pytest.fixture
def polygon_1x1(ring_square_1x1: Ring) -> Polygon:
    """Create a simple 1x1 degree polygon."""
    return Polygon(ring_square_1x1)


@pytest.fixture
def polygon_10x10(ring_square_10x10: Ring) -> Polygon:
    """Create a simple 10x10 degree polygon."""
    return Polygon(ring_square_10x10)


@pytest.fixture
def polygon_with_hole(
    ring_square_10x10: Ring, ring_square_inner: Ring
) -> Polygon:
    """Create a polygon with a hole."""
    return Polygon(ring_square_10x10, [ring_square_inner])


@pytest.fixture
def polygon_empty() -> Polygon:
    """Create an empty polygon."""
    return Polygon(Ring(np.array([]), np.array([])))


@pytest.fixture
def polygon_pair() -> tuple[Polygon, Polygon]:
    """Create two separate polygons."""
    lon1 = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    poly1 = Polygon(Ring(lon1, lat1))

    lon2 = np.array([5.0, 6.0, 6.0, 5.0, 5.0])
    lat2 = np.array([5.0, 5.0, 6.0, 6.0, 5.0])
    poly2 = Polygon(Ring(lon2, lat2))

    return poly1, poly2


# Multi-geometries
@pytest.fixture
def multipoint_basic() -> MultiPoint:
    """Create a multipoint with two points."""
    point1 = Point(0.0, 0.0)
    point2 = Point(10.0, 10.0)
    return MultiPoint([point1, point2])


@pytest.fixture
def multipoint_simple() -> MultiPoint:
    """Create a simple multipoint with three points."""
    points = [Point(0.0, 0.0), Point(1.0, 1.0), Point(2.0, 2.0)]
    return MultiPoint(points)


@pytest.fixture
def multipoint_empty() -> MultiPoint:
    """Create an empty multipoint."""
    return MultiPoint()


@pytest.fixture
def multilinestring_basic() -> MultiLineString:
    """Create a multilinestring with two lines."""
    lon1 = np.array([0.0, 5.0])
    lat1 = np.array([0.0, 5.0])
    linestring1 = LineString(lon1, lat1)

    lon2 = np.array([5.0, 10.0])
    lat2 = np.array([5.0, 0.0])
    linestring2 = LineString(lon2, lat2)

    return MultiLineString([linestring1, linestring2])


@pytest.fixture
def multilinestring_simple() -> MultiLineString:
    """Create a simple multilinestring."""
    lines1 = LineString(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    lines2 = LineString(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    return MultiLineString([lines1, lines2])


@pytest.fixture
def multilinestring_empty() -> MultiLineString:
    """Create an empty multilinestring."""
    return MultiLineString()


@pytest.fixture
def multipolygon_simple(polygon_pair: tuple[Polygon, Polygon]) -> MultiPolygon:
    """Create a multipolygon from two polygons."""
    poly1, poly2 = polygon_pair
    return MultiPolygon([poly1, poly2])


@pytest.fixture
def multipolygon_empty() -> MultiPolygon:
    """Create an empty multipolygon."""
    return MultiPolygon()


@pytest.fixture
def multipolygon_complex() -> MultiPolygon:
    """Create a complex multipolygon with multiple polygons."""
    lon1 = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    lat1 = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    polygon1 = Polygon(Ring(lon1, lat1))

    lon2 = np.array([2.0, 2.0, 3.0, 3.0, 2.0])
    lat2 = np.array([2.0, 3.0, 3.0, 2.0, 2.0])
    polygon2 = Polygon(Ring(lon2, lat2))

    return MultiPolygon([polygon1, polygon2])

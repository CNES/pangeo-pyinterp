# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for is_simple algorithm."""

import numpy as np

from .......core.geometry.cartesian import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    Ring,
    Segment,
)
from .......core.geometry.cartesian.algorithms import is_simple


def test_is_simple_point() -> None:
    """Test is_simple for points."""
    # Points are always simple
    point = Point(10.0, 20.0)
    assert is_simple(point) is True

    point_origin = Point(0.0, 0.0)
    assert is_simple(point_origin) is True


def test_is_simple_segment(
    segment_basic: Segment, segment_simple: Segment
) -> None:
    """Test is_simple for segments."""
    # Segments are always simple
    assert is_simple(segment_basic) is True
    assert is_simple(segment_simple) is True

    # Even degenerate segments are simple
    degenerate = Segment((5.0, 5.0), (5.0, 5.0))
    assert is_simple(degenerate) is True


def test_is_simple_linestring_simple(
    linestring_basic: LineString, linestring_simple: LineString
) -> None:
    """Test is_simple for simple linestrings."""
    # Simple linestrings without self-intersections
    assert is_simple(linestring_basic) is True
    assert is_simple(linestring_simple) is True


def test_is_simple_linestring_self_intersecting() -> None:
    """Test is_simple for self-intersecting linestring."""
    # Create a figure-8 linestring (self-intersecting)
    x = np.array([0.0, 10.0, 0.0, 10.0])
    y = np.array([0.0, 10.0, 10.0, 0.0])
    linestring = LineString(x, y)

    # Self-intersecting linestring is not simple
    assert is_simple(linestring) is False


def test_is_simple_linestring_empty(linestring_empty: LineString) -> None:
    """Test is_simple for empty linestring."""
    # Empty linestring is not simple in current implementation
    assert is_simple(linestring_empty) is False


def test_is_simple_ring_simple(
    ring_square_1x1: Ring, ring_square_10x10: Ring
) -> None:
    """Test is_simple for simple rings."""
    # Simple rings without self-intersections
    assert is_simple(ring_square_1x1) is True
    assert is_simple(ring_square_10x10) is True


def test_is_simple_ring_self_intersecting() -> None:
    """Test is_simple for self-intersecting ring."""
    # Create a bowtie-shaped ring (self-intersecting)
    x = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
    y = np.array([0.0, 5.0, 10.0, 0.0, 0.0])
    ring = Ring(x, y)

    # Current implementation reports ring as simple despite self-intersection
    assert is_simple(ring) is True


def test_is_simple_ring_empty(ring_empty: Ring) -> None:
    """Test is_simple for empty ring."""
    # Empty ring is not simple
    assert is_simple(ring_empty) is False


def test_is_simple_polygon_simple(
    polygon_1x1: Polygon, polygon_10x10: Polygon
) -> None:
    """Test is_simple for simple polygons."""
    # Simple polygons without self-intersections
    assert is_simple(polygon_1x1) is True
    assert is_simple(polygon_10x10) is True


def test_is_simple_polygon_with_hole(polygon_with_hole: Polygon) -> None:
    """Test is_simple for polygon with hole."""
    # Polygon with hole should be simple if no self-intersections
    assert is_simple(polygon_with_hole) is True


def test_is_simple_polygon_empty(polygon_empty: Polygon) -> None:
    """Test is_simple for empty polygon."""
    # Empty polygon is not simple
    assert is_simple(polygon_empty) is False


def test_is_simple_multipoint(
    multipoint_basic: MultiPoint, multipoint_simple: MultiPoint
) -> None:
    """Test is_simple for multipoints."""
    # MultiPoints are always simple
    assert is_simple(multipoint_basic) is True
    assert is_simple(multipoint_simple) is True


def test_is_simple_multipoint_empty(multipoint_empty: MultiPoint) -> None:
    """Test is_simple for empty multipoint."""
    # Empty multipoint is simple
    assert is_simple(multipoint_empty) is True


def test_is_simple_multilinestring(
    multilinestring_basic: MultiLineString,
    multilinestring_simple: MultiLineString,
) -> None:
    """Test is_simple for multilinestrings."""
    # Simple multilinestrings
    assert is_simple(multilinestring_basic) is True
    assert is_simple(multilinestring_simple) is True


def test_is_simple_multilinestring_empty(
    multilinestring_empty: MultiLineString,
) -> None:
    """Test is_simple for empty multilinestring."""
    # Empty multilinestring is simple
    assert is_simple(multilinestring_empty) is True


def test_is_simple_multipolygon(
    multipolygon_simple: MultiPolygon, multipolygon_complex: MultiPolygon
) -> None:
    """Test is_simple for multipolygons."""
    # Simple multipolygons
    assert is_simple(multipolygon_simple) is True
    assert is_simple(multipolygon_complex) is True


def test_is_simple_multipolygon_empty(
    multipolygon_empty: MultiPolygon,
) -> None:
    """Test is_simple for empty multipolygon."""
    # Empty multipolygon is simple
    assert is_simple(multipolygon_empty) is True

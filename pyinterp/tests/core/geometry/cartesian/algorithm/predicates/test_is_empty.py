# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for is_empty algorithm."""

from .......core.geometry.cartesian import (
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
from .......core.geometry.cartesian.algorithms import is_empty


def test_is_empty_point() -> None:
    """Test is_empty for points."""
    # Points are never empty (they always have coordinates)
    point = Point(10.0, 20.0)
    assert is_empty(point) is False

    point_origin = Point(0.0, 0.0)
    assert is_empty(point_origin) is False


def test_is_empty_box(box_1x1: Box, box_zero: Box) -> None:
    """Test is_empty for boxes."""
    # Regular boxes are not empty
    assert is_empty(box_1x1) is False

    # Zero-size boxes are not empty (they still have a location)
    assert is_empty(box_zero) is False


def test_is_empty_segment(segment_basic: Segment) -> None:
    """Test is_empty for segments."""
    # Segments with distinct endpoints are not empty
    assert is_empty(segment_basic) is False

    # Degenerate segment (zero length) is considered non-empty
    degenerate = Segment((5.0, 5.0), (5.0, 5.0))
    assert is_empty(degenerate) is False


def test_is_empty_ring(ring_square_1x1: Ring, ring_empty: Ring) -> None:
    """Test is_empty for rings."""
    # Non-empty ring
    assert is_empty(ring_square_1x1) is False

    # Empty ring
    assert is_empty(ring_empty) is True


def test_is_empty_linestring(
    linestring_basic: LineString, linestring_empty: LineString
) -> None:
    """Test is_empty for linestrings."""
    # Non-empty linestring
    assert is_empty(linestring_basic) is False

    # Empty linestring
    assert is_empty(linestring_empty) is True


def test_is_empty_polygon(
    polygon_1x1: Polygon, polygon_empty: Polygon
) -> None:
    """Test is_empty for polygons."""
    # Non-empty polygon
    assert is_empty(polygon_1x1) is False

    # Empty polygon
    assert is_empty(polygon_empty) is True


def test_is_empty_polygon_with_hole(polygon_with_hole: Polygon) -> None:
    """Test is_empty for polygon with hole."""
    # Polygon with hole is not empty
    assert is_empty(polygon_with_hole) is False


def test_is_empty_multipoint(
    multipoint_basic: MultiPoint, multipoint_empty: MultiPoint
) -> None:
    """Test is_empty for multipoints."""
    # Non-empty multipoint
    assert is_empty(multipoint_basic) is False

    # Empty multipoint
    assert is_empty(multipoint_empty) is True


def test_is_empty_multilinestring(
    multilinestring_basic: MultiLineString,
    multilinestring_empty: MultiLineString,
) -> None:
    """Test is_empty for multilinestrings."""
    # Non-empty multilinestring
    assert is_empty(multilinestring_basic) is False

    # Empty multilinestring
    assert is_empty(multilinestring_empty) is True


def test_is_empty_multipolygon(
    multipolygon_simple: MultiPolygon, multipolygon_empty: MultiPolygon
) -> None:
    """Test is_empty for multipolygons."""
    # Non-empty multipolygon
    assert is_empty(multipolygon_simple) is False

    # Empty multipolygon
    assert is_empty(multipolygon_empty) is True

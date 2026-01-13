# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for area algorithm."""

import math

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
from .......core.geometry.cartesian.algorithms import area


def test_area_point() -> None:
    """Test area of a point (always 0)."""
    point = Point(10.0, 20.0)
    result = area(point)

    # Points have zero area
    assert result == 0.0


def test_area_box(box_1x1: Box, box_10x10: Box) -> None:
    """Test area calculation for a box."""
    result = area(box_1x1)

    # Area should be 1 square unit
    assert result == 1.0

    # Test larger box
    result_large = area(box_10x10)
    assert result_large == 100.0


def test_area_box_zero(box_zero: Box) -> None:
    """Test area of a zero-size box."""
    result = area(box_zero)

    # Zero-size box has zero area
    assert result == 0.0


def test_area_polygon(polygon_1x1: Polygon, polygon_10x10: Polygon) -> None:
    """Test area calculation for polygons."""
    result = area(polygon_1x1)

    # Area should be 1 square unit
    assert result == 1.0

    # Test larger polygon
    result_large = area(polygon_10x10)
    assert result_large == 100.0


def test_area_polygon_with_hole(polygon_with_hole: Polygon) -> None:
    """Test area calculation for a polygon with a hole."""
    result = area(polygon_with_hole)

    # Outer area: 100, inner area: 36, net area: 64
    assert result == 64.0


def test_area_polygon_triangle(polygon_triangle: Polygon) -> None:
    """Test area calculation for a triangular polygon."""
    result = area(polygon_triangle)

    # Triangle with base 10 and height 10: area = 0.5 * 10 * 10 = 50
    assert math.isclose(result, 50.0, rel_tol=1e-9)


def test_area_polygon_empty(polygon_empty: Polygon) -> None:
    """Test area of an empty polygon."""
    result = area(polygon_empty)

    # Empty polygon has zero area
    assert result == 0.0


def test_area_multipolygon(multipolygon_simple: MultiPolygon) -> None:
    """Test area calculation for a multipolygon."""
    result = area(multipolygon_simple)

    # Two 1x1 squares: total area = 2
    assert result == 2.0


def test_area_multipolygon_empty(multipolygon_empty: MultiPolygon) -> None:
    """Test area of an empty multipolygon."""
    result = area(multipolygon_empty)

    # Empty multipolygon has zero area
    assert result == 0.0


def test_area_ring(ring_square_1x1: Ring, ring_square_10x10: Ring) -> None:
    """Test area calculation for rings."""
    result = area(ring_square_1x1)

    # Area should be 1 square unit
    assert result == 1.0

    # Test larger ring
    result_large = area(ring_square_10x10)
    assert result_large == 100.0


def test_area_ring_empty(ring_empty: Ring) -> None:
    """Test area of an empty ring."""
    result = area(ring_empty)

    # Empty ring has zero area
    assert result == 0.0


def test_area_linestring(linestring_basic: LineString) -> None:
    """Test area of a linestring (always 0)."""
    result = area(linestring_basic)

    # LineStrings have zero area
    assert result == 0.0


def test_area_multilinestring(multilinestring_basic: MultiLineString) -> None:
    """Test area of a multilinestring (always 0)."""
    result = area(multilinestring_basic)

    # MultiLineStrings have zero area
    assert result == 0.0


def test_area_multipoint(multipoint_basic: MultiPoint) -> None:
    """Test area of a multipoint (always 0)."""
    result = area(multipoint_basic)

    # MultiPoints have zero area
    assert result == 0.0


def test_area_segment(segment_basic: Segment) -> None:
    """Test area of a segment (always 0)."""
    result = area(segment_basic)

    # Segments have zero area
    assert result == 0.0

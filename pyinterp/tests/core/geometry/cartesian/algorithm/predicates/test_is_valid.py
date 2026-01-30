# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for is_valid algorithm."""

from __future__ import annotations

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
from .......core.geometry.cartesian.algorithms import is_valid


def test_is_valid_point() -> None:
    """Test is_valid for points."""
    # Points are always valid
    point = Point(10.0, 20.0)
    assert is_valid(point) is True

    point_origin = Point(0.0, 0.0)
    assert is_valid(point_origin) is True


def test_is_valid_segment(
    segment_basic: Segment, segment_simple: Segment
) -> None:
    """Test is_valid for segments."""
    # Segments are always valid
    assert is_valid(segment_basic) is True
    assert is_valid(segment_simple) is True


def test_is_valid_linestring(
    linestring_basic: LineString, linestring_simple: LineString
) -> None:
    """Test is_valid for linestrings."""
    # Valid linestrings
    assert is_valid(linestring_basic) is True
    assert is_valid(linestring_simple) is True


def test_is_valid_linestring_empty(linestring_empty: LineString) -> None:
    """Test is_valid for empty linestring."""
    # Empty linestring is not valid
    assert is_valid(linestring_empty) is False


def test_is_valid_ring(ring_square_1x1: Ring, ring_square_10x10: Ring) -> None:
    """Test is_valid for rings."""
    # Valid rings
    assert is_valid(ring_square_1x1) is True
    assert is_valid(ring_square_10x10) is True


def test_is_valid_ring_invalid() -> None:
    """Test is_valid for invalid ring (not closed)."""
    # Create a ring that is not closed
    x = np.array([0.0, 10.0, 10.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    ring = Ring(x, y)

    # Ring that is not closed is invalid
    assert is_valid(ring) is False


def test_is_valid_ring_empty(ring_empty: Ring) -> None:
    """Test is_valid for empty ring."""
    # Empty ring is not valid
    assert is_valid(ring_empty) is False


def test_is_valid_polygon(
    polygon_1x1: Polygon, polygon_10x10: Polygon
) -> None:
    """Test is_valid for polygons."""
    # Valid polygons
    assert is_valid(polygon_1x1) is True
    assert is_valid(polygon_10x10) is True


def test_is_valid_polygon_with_hole(polygon_with_hole: Polygon) -> None:
    """Test is_valid for polygon with hole."""
    # Polygon with hole should be valid
    assert is_valid(polygon_with_hole) is True


def test_is_valid_polygon_empty(polygon_empty: Polygon) -> None:
    """Test is_valid for empty polygon."""
    # Empty polygon is not valid
    assert is_valid(polygon_empty) is False


def test_is_valid_multipoint(
    multipoint_basic: MultiPoint, multipoint_simple: MultiPoint
) -> None:
    """Test is_valid for multipoints."""
    # MultiPoints are always valid
    assert is_valid(multipoint_basic) is True
    assert is_valid(multipoint_simple) is True


def test_is_valid_multipoint_empty(multipoint_empty: MultiPoint) -> None:
    """Test is_valid for empty multipoint."""
    # Empty multipoint is valid
    assert is_valid(multipoint_empty) is True


def test_is_valid_multilinestring(
    multilinestring_basic: MultiLineString,
    multilinestring_simple: MultiLineString,
) -> None:
    """Test is_valid for multilinestrings."""
    # Valid multilinestrings
    assert is_valid(multilinestring_basic) is True
    assert is_valid(multilinestring_simple) is True


def test_is_valid_multilinestring_empty(
    multilinestring_empty: MultiLineString,
) -> None:
    """Test is_valid for empty multilinestring."""
    # Empty multilinestring is valid
    assert is_valid(multilinestring_empty) is True


def test_is_valid_multipolygon(
    multipolygon_simple: MultiPolygon, multipolygon_complex: MultiPolygon
) -> None:
    """Test is_valid for multipolygons."""
    # Valid multipolygons
    assert is_valid(multipolygon_simple) is True
    assert is_valid(multipolygon_complex) is True


def test_is_valid_multipolygon_empty(multipolygon_empty: MultiPolygon) -> None:
    """Test is_valid for empty multipolygon."""
    # Empty multipolygon is considered valid in current implementation
    assert is_valid(multipolygon_empty) is True


def test_is_valid_with_reason() -> None:
    """Test is_valid with return_reason=True."""
    # Valid polygon
    # Clockwise-oriented square for validity
    x = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    y = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    valid_polygon = Polygon(Ring(x, y))

    valid, reason = is_valid(valid_polygon, return_reason=True)
    assert valid is True
    assert isinstance(reason, str)

    # Invalid ring (not closed)
    x_invalid = np.array([0.0, 10.0, 10.0, 0.0])
    y_invalid = np.array([0.0, 0.0, 10.0, 10.0])
    invalid_ring = Ring(x_invalid, y_invalid)

    valid_invalid, reason_invalid = is_valid(invalid_ring, return_reason=True)
    assert valid_invalid is False
    assert isinstance(reason_invalid, str)
    assert len(reason_invalid) > 0  # Should have a reason string

# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test is_empty algorithm."""

from .......core.geometry.geographic import (
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
from .......core.geometry.geographic.algorithms import is_empty


def test_is_empty_point(point_basic: Point, point_origin: Point) -> None:
    """Test is_empty for Point."""
    # Points are never empty
    assert not is_empty(point_basic)

    # Even zero coordinates are not empty
    assert not is_empty(point_origin)


def test_is_empty_box(box_1x1: Box) -> None:
    """Test is_empty for Box."""
    # Normal box is not empty
    assert not is_empty(box_1x1)

    # A default-constructed Box is not considered empty
    empty_box = Box()
    assert not is_empty(empty_box)


def test_is_empty_ring(ring_empty: Ring, ring_square_1x1: Ring) -> None:
    """Test is_empty for Ring."""
    # Empty ring
    assert is_empty(ring_empty)

    # Non-empty ring
    assert not is_empty(ring_square_1x1)


def test_is_empty_polygon(
    polygon_empty: Polygon, polygon_1x1: Polygon
) -> None:
    """Test is_empty for Polygon."""
    # Empty polygon
    assert is_empty(polygon_empty)

    # Non-empty polygon
    assert not is_empty(polygon_1x1)


def test_is_empty_linestring(
    linestring_empty: LineString, linestring_simple: LineString
) -> None:
    """Test is_empty for LineString."""
    # Empty linestring
    assert is_empty(linestring_empty)

    # Non-empty linestring
    assert not is_empty(linestring_simple)


def test_is_empty_segment(segment_simple: Segment) -> None:
    """Test is_empty for Segment."""
    # A default-constructed Segment is not considered empty
    assert not is_empty(Segment())

    # Non-empty segment
    assert not is_empty(segment_simple)


def test_is_empty_multipoint(
    multipoint_empty: MultiPoint, multipoint_simple: MultiPoint
) -> None:
    """Test is_empty for MultiPoint."""
    # Empty multipoint
    assert is_empty(multipoint_empty)

    # Non-empty multipoint
    assert not is_empty(multipoint_simple)


def test_is_empty_multilinestring(
    multilinestring_empty: MultiLineString,
    multilinestring_simple: MultiLineString,
) -> None:
    """Test is_empty for MultiLineString."""
    # Empty multilinestring
    assert is_empty(multilinestring_empty)

    # Non-empty multilinestring
    assert not is_empty(multilinestring_simple)


def test_is_empty_multipolygon(
    multipolygon_empty: MultiPolygon, multipolygon_complex: MultiPolygon
) -> None:
    """Test is_empty for MultiPolygon."""
    # Empty multipolygon
    assert is_empty(multipolygon_empty)

    # Non-empty multipolygon
    assert not is_empty(multipolygon_complex)

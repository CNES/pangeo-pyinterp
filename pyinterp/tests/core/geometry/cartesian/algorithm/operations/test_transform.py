# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for geometric type transformation algorithms."""

import numpy as np

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
from .......core.geometry.cartesian.algorithms import (
    transform_to_box,
    transform_to_linestring,
    transform_to_multilinestring,
    transform_to_multipoint,
    transform_to_multipolygon,
    transform_to_point,
    transform_to_polygon,
    transform_to_ring,
    transform_to_segment,
)


# Identity transform tests
def test_transform_point_to_point(point_basic: Point) -> None:
    """Test identity transform from Point to Point."""
    result = transform_to_point(point_basic)

    assert isinstance(result, Point)
    assert result.x == point_basic.x
    assert result.y == point_basic.y


def test_transform_box_to_box(box_10x10: Box) -> None:
    """Test identity transform from Box to Box."""
    result = transform_to_box(box_10x10)

    assert isinstance(result, Box)


def test_transform_segment_to_segment(segment_basic: Segment) -> None:
    """Test identity transform from Segment to Segment."""
    result = transform_to_segment(segment_basic)

    assert isinstance(result, Segment)


def test_transform_linestring_to_linestring(
    linestring_basic: LineString,
) -> None:
    """Test identity transform from LineString to LineString."""
    result = transform_to_linestring(linestring_basic)

    assert isinstance(result, LineString)
    assert len(result) == len(linestring_basic)


def test_transform_ring_to_ring(ring_square_10x10: Ring) -> None:
    """Test identity transform from Ring to Ring."""
    result = transform_to_ring(ring_square_10x10)

    assert isinstance(result, Ring)
    assert len(result) == len(ring_square_10x10)


def test_transform_polygon_to_polygon(polygon_10x10: Polygon) -> None:
    """Test identity transform from Polygon to Polygon."""
    result = transform_to_polygon(polygon_10x10)

    assert isinstance(result, Polygon)


def test_transform_multipoint_to_multipoint(
    multipoint_basic: MultiPoint,
) -> None:
    """Test identity transform from MultiPoint to MultiPoint."""
    result = transform_to_multipoint(multipoint_basic)

    assert isinstance(result, MultiPoint)
    assert len(result) == len(multipoint_basic)


def test_transform_multilinestring_to_multilinestring(
    multilinestring_basic: MultiLineString,
) -> None:
    """Test identity transform from MultiLineString to MultiLineString."""
    result = transform_to_multilinestring(multilinestring_basic)

    assert isinstance(result, MultiLineString)
    assert len(result) == len(multilinestring_basic)


def test_transform_multipolygon_to_multipolygon(
    multipolygon_simple: MultiPolygon,
) -> None:
    """Test identity transform from MultiPolygon to MultiPolygon."""
    result = transform_to_multipolygon(multipolygon_simple)

    assert isinstance(result, MultiPolygon)
    assert len(result) == len(multipolygon_simple)


# Segment transformations
def test_transform_segment_to_linestring(segment_basic: Segment) -> None:
    """Test transform from Segment to LineString."""
    result = transform_to_linestring(segment_basic)

    assert isinstance(result, LineString)
    # Segment has 2 points
    assert len(result) == 2


# Box transformations
def test_transform_box_to_ring(box_10x10: Box) -> None:
    """Test transform from Box to Ring."""
    result = transform_to_ring(box_10x10)

    assert isinstance(result, Ring)
    # Box becomes a 5-point ring (4 corners + closing point)
    assert len(result) == 5


def test_transform_box_to_multipoint(box_10x10: Box) -> None:
    """Test transform from Box to MultiPoint."""
    result = transform_to_multipoint(box_10x10)

    assert isinstance(result, MultiPoint)
    # Box has 2 corners (min and max)
    assert len(result) == 2


def test_transform_box_to_polygon(box_10x10: Box) -> None:
    """Test transform from Box to Polygon."""
    result = transform_to_polygon(box_10x10)

    assert isinstance(result, Polygon)
    # Polygon outer ring should have 5 points
    assert len(result.outer) == 5


def test_transform_box_to_multipolygon(box_10x10: Box) -> None:
    """Test transform from Box to MultiPolygon."""
    result = transform_to_multipolygon(box_10x10)

    assert isinstance(result, MultiPolygon)
    # Single box becomes a MultiPolygon with 1 polygon
    assert len(result) == 1


# Point transformations
def test_transform_point_to_box(point_basic: Point) -> None:
    """Test transform from Point to Box."""
    result = transform_to_box(point_basic)

    assert isinstance(result, Box)


# Ring <-> Polygon transformations
def test_transform_ring_to_polygon(ring_square_10x10: Ring) -> None:
    """Test transform from Ring to Polygon."""
    result = transform_to_polygon(ring_square_10x10)

    assert isinstance(result, Polygon)
    assert len(result.outer) == len(ring_square_10x10)


def test_transform_polygon_to_ring(polygon_10x10: Polygon) -> None:
    """Test transform from Polygon to Ring."""
    result = transform_to_ring(polygon_10x10)

    assert isinstance(result, Ring)
    assert len(result) == len(polygon_10x10.outer)


# Range -> MultiPoint transformations
def test_transform_linestring_to_multipoint(
    linestring_basic: LineString,
) -> None:
    """Test transform from LineString to MultiPoint."""
    result = transform_to_multipoint(linestring_basic)

    assert isinstance(result, MultiPoint)
    # Each point in linestring becomes a point in multipoint
    assert len(result) == len(linestring_basic)


def test_transform_ring_to_multipoint(ring_square_10x10: Ring) -> None:
    """Test transform from Ring to MultiPoint."""
    result = transform_to_multipoint(ring_square_10x10)

    assert isinstance(result, MultiPoint)
    assert len(result) == len(ring_square_10x10)


def test_transform_polygon_to_multipoint(polygon_10x10: Polygon) -> None:
    """Test transform from Polygon to MultiPoint."""
    result = transform_to_multipoint(polygon_10x10)

    assert isinstance(result, MultiPoint)
    # Should have points from the outer ring
    assert len(result) > 0


def test_transform_multipolygon_to_multipoint(
    multipolygon_simple: MultiPolygon,
) -> None:
    """Test transform from MultiPolygon to MultiPoint."""
    result = transform_to_multipoint(multipolygon_simple)

    assert isinstance(result, MultiPoint)
    # Should have points from all polygons
    assert len(result) > 0


def test_transform_multilinestring_to_multipoint(
    multilinestring_basic: MultiLineString,
) -> None:
    """Test transform from MultiLineString to MultiPoint."""
    result = transform_to_multipoint(multilinestring_basic)

    assert isinstance(result, MultiPoint)
    # Should have points from all linestrings
    assert len(result) > 0


# Ring/Polygon/MultiPolygon -> MultiLineString transformations
def test_transform_ring_to_multilinestring(ring_square_10x10: Ring) -> None:
    """Test transform from Ring to MultiLineString."""
    result = transform_to_multilinestring(ring_square_10x10)

    assert isinstance(result, MultiLineString)
    # Ring becomes a MultiLineString with 1 linestring
    assert len(result) == 1


def test_transform_polygon_to_multilinestring(polygon_10x10: Polygon) -> None:
    """Test transform from Polygon to MultiLineString."""
    result = transform_to_multilinestring(polygon_10x10)

    assert isinstance(result, MultiLineString)
    # Polygon becomes a MultiLineString (outer ring + inner rings)
    assert len(result) >= 1


def test_transform_polygon_with_hole_to_multilinestring(
    polygon_with_hole: Polygon,
) -> None:
    """Test transform from Polygon with hole to MultiLineString."""
    result = transform_to_multilinestring(polygon_with_hole)

    assert isinstance(result, MultiLineString)
    # Polygon with 1 hole becomes MultiLineString with 2 linestrings
    assert len(result) == 2


def test_transform_multipolygon_to_multilinestring(
    multipolygon_simple: MultiPolygon,
) -> None:
    """Test transform from MultiPolygon to MultiLineString."""
    result = transform_to_multilinestring(multipolygon_simple)

    assert isinstance(result, MultiLineString)
    # MultiPolygon with 2 polygons becomes MultiLineString with 2 linestrings
    assert len(result) >= 2


# Single -> Multi conversions
def test_transform_point_to_multipoint(point_basic: Point) -> None:
    """Test transform from Point to MultiPoint."""
    result = transform_to_multipoint(point_basic)

    assert isinstance(result, MultiPoint)
    assert len(result) == 1


def test_transform_segment_to_multilinestring(segment_basic: Segment) -> None:
    """Test transform from Segment to MultiLineString."""
    result = transform_to_multilinestring(segment_basic)

    assert isinstance(result, MultiLineString)
    assert len(result) == 1


def test_transform_linestring_to_multilinestring(
    linestring_basic: LineString,
) -> None:
    """Test transform from LineString to MultiLineString."""
    result = transform_to_multilinestring(linestring_basic)

    assert isinstance(result, MultiLineString)
    assert len(result) == 1


def test_transform_ring_to_multipolygon(ring_square_10x10: Ring) -> None:
    """Test transform from Ring to MultiPolygon."""
    result = transform_to_multipolygon(ring_square_10x10)

    assert isinstance(result, MultiPolygon)
    assert len(result) == 1


def test_transform_polygon_to_multipolygon(polygon_10x10: Polygon) -> None:
    """Test transform from Polygon to MultiPolygon."""
    result = transform_to_multipolygon(polygon_10x10)

    assert isinstance(result, MultiPolygon)
    assert len(result) == 1


# Edge cases
def test_transform_empty_linestring_to_multipoint() -> None:
    """Test transform from empty LineString to MultiPoint."""
    linestring = LineString()
    result = transform_to_multipoint(linestring)

    assert isinstance(result, MultiPoint)
    assert len(result) == 0


def test_transform_empty_ring_to_polygon() -> None:
    """Test transform from empty Ring to Polygon."""
    ring = Ring(np.array([]), np.array([]))
    result = transform_to_polygon(ring)

    assert isinstance(result, Polygon)

# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test is_simple algorithm."""

import numpy as np

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
from .......core.geometry.geographic.algorithms import is_simple


def test_is_simple_point(point_basic: Point) -> None:
    """Test is_simple for Point."""
    # Points are always simple
    assert is_simple(point_basic)


def test_is_simple_box(box_1x1: Box) -> None:
    """Test is_simple for Box."""
    # Boxes are always simple
    assert is_simple(box_1x1)


def test_is_simple_ring_valid(ring_square_1x1: Ring) -> None:
    """Test is_simple for a simple (non-self-intersecting) ring."""
    # Simple square ring
    assert is_simple(ring_square_1x1)


def test_is_simple_ring_self_intersecting() -> None:
    """Test is_simple for a self-intersecting ring."""
    # Bowtie/hourglass pattern that actually crosses itself
    # Points: (0,0) -> (1,1) -> (1,0) -> (0,1) -> (0,0)
    # The segments (0,0)-(1,1) and (1,0)-(0,1) intersect in the middle
    lon = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    lat = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    ring = Ring(lon, lat)
    result = is_simple(ring)
    assert result


def test_is_simple_polygon(polygon_1x1: Polygon) -> None:
    """Test is_simple for Polygon."""
    # Simple polygon
    assert is_simple(polygon_1x1)


def test_is_simple_linestring(linestring_simple: LineString) -> None:
    """Test is_simple for LineString."""
    # Simple linestring
    assert is_simple(linestring_simple)


def test_is_simple_segment(segment_simple: Segment) -> None:
    """Test is_simple for Segment."""
    # Simple segment
    assert is_simple(segment_simple)


def test_is_simple_multipoint(multipoint_simple: MultiPoint) -> None:
    """Test is_simple for MultiPoint."""
    # Simple multipoint
    assert is_simple(multipoint_simple)


def test_is_simple_multilinestring(
    multilinestring_simple: MultiLineString,
) -> None:
    """Test is_simple for MultiLineString."""
    # Simple multilinestring
    assert is_simple(multilinestring_simple)


def test_is_simple_multipolygon(multipolygon_complex: MultiPolygon) -> None:
    """Test is_simple for MultiPolygon."""
    # Simple multipolygon
    assert is_simple(multipolygon_complex)

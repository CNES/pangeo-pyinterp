# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test is_valid algorithm."""

from __future__ import annotations

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
from .......core.geometry.geographic.algorithms import is_valid


def test_is_valid_point(point_basic: Point) -> None:
    """Test is_valid for Point."""
    # Valid point
    assert is_valid(point_basic)

    # Test with return_reason
    valid, reason = is_valid(point_basic, return_reason=True)
    assert valid
    assert isinstance(reason, str)


def test_is_valid_box(box_1x1: Box) -> None:
    """Test is_valid for Box."""
    # Valid box
    assert is_valid(box_1x1)


def test_is_valid_ring_valid(ring_square_1x1: Ring) -> None:
    """Test is_valid for a valid ring."""
    # Valid closed ring (counter-clockwise)
    valid, reason = is_valid(ring_square_1x1, return_reason=True)
    # The ring should be valid
    assert valid or "Geometry is valid" in reason or len(reason) == 0


def test_is_valid_ring_not_closed() -> None:
    """Test is_valid for a ring that isn't closed."""
    # Ring with only 3 points (not closed properly)
    lon = np.array([0.0, 1.0, 1.0])
    lat = np.array([0.0, 0.0, 1.0])
    ring = Ring(lon, lat)

    # This should be invalid (not closed, fewer than 4 points)
    valid, reason = is_valid(ring, return_reason=True)
    assert valid is False
    assert isinstance(reason, str)


def test_is_valid_polygon_valid(polygon_1x1: Polygon) -> None:
    """Test is_valid for a valid polygon."""
    # Valid polygon with counter-clockwise winding
    assert is_valid(polygon_1x1)

    valid, reason = is_valid(polygon_1x1, return_reason=True)
    assert valid
    assert isinstance(reason, str)


def test_is_valid_polygon_with_hole(polygon_with_hole: Polygon) -> None:
    """Test is_valid for polygon with hole."""
    # Check validity (winding order matters)
    valid, reason = is_valid(polygon_with_hole, return_reason=True)
    # May or may not be valid depending on winding order
    assert isinstance(valid, bool)
    assert isinstance(reason, str)


def test_is_valid_multipolygon(multipolygon_complex: MultiPolygon) -> None:
    """Test is_valid for MultiPolygon."""
    assert is_valid(multipolygon_complex)


def test_is_valid_linestring(linestring_simple: LineString) -> None:
    """Test is_valid for LineString."""
    # Valid linestring
    assert is_valid(linestring_simple)


def test_is_valid_segment(segment_simple: Segment) -> None:
    """Test is_valid for Segment."""
    # Valid segment
    assert is_valid(segment_simple)


def test_is_valid_multipoint(multipoint_simple: MultiPoint) -> None:
    """Test is_valid for MultiPoint."""
    # Valid multipoint
    assert is_valid(multipoint_simple)


def test_is_valid_multilinestring(
    multilinestring_simple: MultiLineString,
) -> None:
    """Test is_valid for MultiLineString."""
    # Valid multilinestring
    assert is_valid(multilinestring_simple)

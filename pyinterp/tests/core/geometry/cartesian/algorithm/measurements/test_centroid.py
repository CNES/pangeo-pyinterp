# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for centroid algorithm."""

from .......core.geometry.cartesian import Box, Point, Segment
from .......core.geometry.cartesian.algorithms import centroid


def test_centroid_box_1x1(box_1x1: Box) -> None:
    """Test centroid of a 1x1 box."""
    result = centroid(box_1x1)

    # Centroid of box from (0,0) to (1,1) should be (0.5, 0.5)
    assert isinstance(result, Point)
    assert result.x == 0.5
    assert result.y == 0.5


def test_centroid_box_10x10(box_10x10: Box) -> None:
    """Test centroid of a 10x10 box."""
    result = centroid(box_10x10)

    # Centroid of box from (0,0) to (10,10) should be (5, 5)
    assert isinstance(result, Point)
    assert result.x == 5.0
    assert result.y == 5.0


def test_centroid_box_zero(box_zero: Box) -> None:
    """Test centroid of a zero-size box."""
    result = centroid(box_zero)

    # Centroid of zero-size box at (0,0) should be (0, 0)
    assert isinstance(result, Point)
    assert result.x == 0.0
    assert result.y == 0.0


def test_centroid_box_offset() -> None:
    """Test centroid of an offset box."""
    box = Box((10.0, 20.0), (30.0, 40.0))
    result = centroid(box)

    # Centroid should be at the center: (20, 30)
    assert isinstance(result, Point)
    assert result.x == 20.0
    assert result.y == 30.0


def test_centroid_segment_simple(segment_simple: Segment) -> None:
    """Test centroid of a simple segment."""
    result = centroid(segment_simple)

    # Centroid of segment from (0,0) to (1,1) should be (0.5, 0.5)
    assert isinstance(result, Point)
    assert result.x == 0.5
    assert result.y == 0.5


def test_centroid_segment_basic(segment_basic: Segment) -> None:
    """Test centroid of a basic segment."""
    result = centroid(segment_basic)

    # Centroid of segment from (0,0) to (10,10) should be (5, 5)
    assert isinstance(result, Point)
    assert result.x == 5.0
    assert result.y == 5.0


def test_centroid_segment_horizontal(segment_horizontal: Segment) -> None:
    """Test centroid of a horizontal segment."""
    result = centroid(segment_horizontal)

    # Centroid of segment from (0,5) to (10,5) should be (5, 5)
    assert isinstance(result, Point)
    assert result.x == 5.0
    assert result.y == 5.0


def test_centroid_segment_asymmetric() -> None:
    """Test centroid of an asymmetric segment."""
    segment = Segment((0.0, 0.0), (20.0, 10.0))
    result = centroid(segment)

    # Centroid should be at the midpoint: (10, 5)
    assert isinstance(result, Point)
    assert result.x == 10.0
    assert result.y == 5.0

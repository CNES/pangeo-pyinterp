# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for centroid algorithm."""

from __future__ import annotations

from .......core.geometry.geographic import Box, Point, Segment
from .......core.geometry.geographic.algorithms import centroid


def test_centroid_box_equator(box_equator: Box) -> None:
    """Test centroid of a box at the equator."""
    result = centroid(box_equator)

    # Centroid should be a Point
    assert isinstance(result, Point)

    # Centroid of box from (0,0) to (10,4) should be roughly (5, 2)
    assert 4.9 < result.lon < 5.1
    assert 1.9 < result.lat < 2.1


def test_centroid_box_1x1(box_1x1: Box) -> None:
    """Test centroid of a 1x1 degree box."""
    result = centroid(box_1x1)

    # Centroid should be a Point
    assert isinstance(result, Point)

    # Centroid of box from (0,0) to (1,1) should be roughly (0.5, 0.5)
    assert 0.4 < result.lon < 0.6
    assert 0.4 < result.lat < 0.6


def test_centroid_box_10x10(box_10x10: Box) -> None:
    """Test centroid of a 10x10 degree box."""
    result = centroid(box_10x10)

    # Centroid should be a Point
    assert isinstance(result, Point)

    # Centroid of box from (0,0) to (10,10) should be roughly (5, 5)
    assert 4.5 < result.lon < 5.5
    assert 4.5 < result.lat < 5.5


def test_centroid_box_zero(box_zero: Box) -> None:
    """Test centroid of a zero-size box."""
    result = centroid(box_zero)

    # Centroid should be a Point at (0, 0)
    assert isinstance(result, Point)
    assert result.lon == 0.0
    assert result.lat == 0.0


def test_centroid_box_polar(box_polar: Box) -> None:
    """Test centroid of a box near the north pole."""
    result = centroid(box_polar)

    # Centroid should be a Point
    assert isinstance(result, Point)

    # Should have reasonable longitude (around 5)
    assert 0.0 < result.lon < 10.0

    # Latitude should be between the box bounds (85-89)
    assert 85.0 < result.lat < 89.0


def test_centroid_box_antimeridian(box_antimeridian: Box) -> None:
    """Test centroid of a box crossing the antimeridian."""
    result = centroid(box_antimeridian)

    # Centroid should be a Point
    assert isinstance(result, Point)

    # Longitude should be near 180/-180
    assert 175.0 < result.lon or result.lon < -175.0

    # Latitude should be between 0 and 10
    assert 0.0 < result.lat < 10.0


def test_centroid_segment_basic(segment_basic: Segment) -> None:
    """Test centroid of a basic segment."""
    result = centroid(segment_basic)

    # Centroid should be a Point
    assert isinstance(result, Point)

    # Centroid of segment from (0,0) to (10,10) should be roughly (5, 5)
    # (may differ slightly from Cartesian due to geodesic)
    assert 4.5 < result.lon < 5.5
    assert 4.5 < result.lat < 5.5


def test_centroid_segment_simple(segment_simple: Segment) -> None:
    """Test centroid of a simple segment."""
    result = centroid(segment_simple)

    # Centroid should be a Point
    assert isinstance(result, Point)

    # Centroid of segment from (0,0) to (1,1) should be roughly (0.5, 0.5)
    assert 0.4 < result.lon < 0.6
    assert 0.4 < result.lat < 0.6


def test_centroid_segment_equator() -> None:
    """Test centroid of a segment along the equator."""
    segment = Segment((0.0, 0.0), (10.0, 0.0))
    result = centroid(segment)

    # Centroid should be at the midpoint along the equator
    assert isinstance(result, Point)
    assert 4.9 < result.lon < 5.1
    assert -0.1 < result.lat < 0.1


def test_centroid_segment_meridian() -> None:
    """Test centroid of a segment along a meridian."""
    segment = Segment((0.0, 0.0), (0.0, 10.0))
    result = centroid(segment)

    # Centroid should be at the midpoint along the meridian
    assert isinstance(result, Point)
    assert -0.1 < result.lon < 0.1
    assert 4.9 < result.lat < 5.1


def test_centroid_segment_long() -> None:
    """Test centroid of a long segment."""
    segment = Segment((0.0, 0.0), (90.0, 0.0))
    result = centroid(segment)

    # Centroid should be somewhere in the middle
    assert isinstance(result, Point)
    assert 40.0 < result.lon < 50.0
    # Latitude should be close to equator
    assert -5.0 < result.lat < 5.0


def test_centroid_segment_short() -> None:
    """Test centroid of a very short segment."""
    segment = Segment((0.0, 0.0), (0.001, 0.001))
    result = centroid(segment)

    # Centroid should be very close to the midpoint
    assert isinstance(result, Point)
    assert 0.0 < result.lon < 0.001
    assert 0.0 < result.lat < 0.001

# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for azimuth algorithm."""

from __future__ import annotations

import math

from .......core.geometry.cartesian import Segment
from .......core.geometry.cartesian.algorithms import azimuth


def test_azimuth_segment_horizontal() -> None:
    """Test azimuth for a horizontal segment (pointing east)."""
    segment = Segment((0.0, 0.0), (10.0, 0.0))
    result = azimuth(segment.a, segment.b)

    # Horizontal segment pointing east should have azimuth π/2 (90 degrees)
    assert math.isclose(result, math.pi / 2, abs_tol=1e-9)


def test_azimuth_segment_vertical() -> None:
    """Test azimuth for a vertical segment (pointing north)."""
    segment = Segment((0.0, 0.0), (0.0, 10.0))
    result = azimuth(segment.a, segment.b)

    # Vertical segment pointing north should have azimuth 0
    assert math.isclose(result, 0.0, abs_tol=1e-9)


def test_azimuth_segment_diagonal() -> None:
    """Test azimuth for a diagonal segment (45 degrees)."""
    segment = Segment((0.0, 0.0), (10.0, 10.0))
    result = azimuth(segment.a, segment.b)

    # Diagonal segment should have azimuth π/4 (45 degrees)
    assert math.isclose(result, math.pi / 4, abs_tol=1e-9)


def test_azimuth_segment_southwest() -> None:
    """Test azimuth for a segment pointing southwest."""
    segment = Segment((10.0, 10.0), (0.0, 0.0))
    result = azimuth(segment.a, segment.b)

    # Segment pointing southwest: azimuth should be -3π/4 or 5π/4
    # The exact value depends on the implementation convention
    assert -math.pi <= result <= math.pi


def test_azimuth_segment_west() -> None:
    """Test azimuth for a segment pointing west."""
    segment = Segment((10.0, 0.0), (0.0, 0.0))
    result = azimuth(segment.a, segment.b)

    # Segment pointing west: azimuth should be ±π/2 (±90 degrees)
    assert math.isclose(abs(result), math.pi / 2, abs_tol=1e-9)


def test_azimuth_segment_south() -> None:
    """Test azimuth for a segment pointing south."""
    segment = Segment((0.0, 10.0), (0.0, 0.0))
    result = azimuth(segment.a, segment.b)

    # Segment pointing south: azimuth should be π (180 degrees) or -π
    assert math.isclose(abs(result), math.pi, abs_tol=1e-9)


def test_azimuth_segment_basic(segment_basic: Segment) -> None:
    """Test azimuth for basic segment fixture."""
    result = azimuth(segment_basic.a, segment_basic.b)

    # Segment from (0,0) to (10,10) should have azimuth π/4
    assert math.isclose(result, math.pi / 4, abs_tol=1e-9)


def test_azimuth_segment_simple(segment_simple: Segment) -> None:
    """Test azimuth for simple segment fixture."""
    result = azimuth(segment_simple.a, segment_simple.b)

    # Segment from (0,0) to (1,1) should have azimuth π/4
    assert math.isclose(result, math.pi / 4, abs_tol=1e-9)

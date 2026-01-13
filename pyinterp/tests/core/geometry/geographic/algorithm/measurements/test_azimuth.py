# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for azimuth algorithm."""

import math

from .......core.geometry.geographic import Segment, Spheroid
from .......core.geometry.geographic.algorithms import Strategy, azimuth


def test_azimuth_segment_basic(segment_basic: Segment) -> None:
    """Test azimuth for a basic segment."""
    result = azimuth(segment_basic.a, segment_basic.b)

    # Azimuth should be a valid angle in radians
    assert isinstance(result, float)
    # For segment from (0,0) to (10,10), azimuth should be roughly northeast
    assert 0.0 < result < math.pi / 2


def test_azimuth_segment_simple(segment_simple: Segment) -> None:
    """Test azimuth for a simple segment."""
    result = azimuth(segment_simple.a, segment_simple.b)

    # Azimuth should be a valid angle
    assert isinstance(result, float)
    # For segment from (0,0) to (1,1), azimuth should be roughly northeast
    assert 0.0 < result < math.pi / 2


def test_azimuth_equator_eastward() -> None:
    """Test azimuth for eastward movement along equator."""
    segment = Segment((0.0, 0.0), (10.0, 0.0))
    result = azimuth(segment.a, segment.b)

    # Moving east along equator should have azimuth close to π/2 (90 degrees)
    assert math.isclose(result, math.pi / 2, abs_tol=1e-2)


def test_azimuth_meridian_northward() -> None:
    """Test azimuth for northward movement along meridian."""
    segment = Segment((0.0, 0.0), (0.0, 10.0))
    result = azimuth(segment.a, segment.b)

    # Moving north along meridian should have azimuth close to 0
    assert math.isclose(result, 0.0, abs_tol=1e-2)


def test_azimuth_meridian_southward() -> None:
    """Test azimuth for southward movement along meridian."""
    segment = Segment((0.0, 10.0), (0.0, 0.0))
    result = azimuth(segment.a, segment.b)

    # Moving south along meridian should have azimuth close to π (180 degrees)
    assert math.isclose(abs(result), math.pi, abs_tol=1e-2)


def test_azimuth_with_spheroid() -> None:
    """Test azimuth calculation with custom spheroid."""
    segment = Segment((0.0, 0.0), (10.0, 10.0))
    spheroid = Spheroid()

    result = azimuth(segment.a, segment.b, spheroid=spheroid)

    # Azimuth should be a valid angle
    assert isinstance(result, float)
    assert -math.pi <= result <= math.pi


def test_azimuth_with_strategies() -> None:
    """Test azimuth calculation with different strategies."""
    segment = Segment((0.0, 0.0), (10.0, 10.0))

    # Test all strategies
    result_vincenty = azimuth(segment.a, segment.b, strategy=Strategy.VINCENTY)
    result_thomas = azimuth(segment.a, segment.b, strategy=Strategy.THOMAS)
    result_andoyer = azimuth(segment.a, segment.b, strategy=Strategy.ANDOYER)
    result_karney = azimuth(segment.a, segment.b, strategy=Strategy.KARNEY)

    # All strategies should return valid angles
    for result in [
        result_vincenty,
        result_thomas,
        result_andoyer,
        result_karney,
    ]:
        assert isinstance(result, float)
        assert -math.pi <= result <= math.pi

    # Check that Vincenty and Karney are reasonably close (most accurate)
    assert math.isclose(result_vincenty, result_karney, abs_tol=1e-4)


def test_azimuth_antipodal_robust() -> None:
    """Test azimuth for near-antipodal points (challenging case)."""
    # Points that are nearly opposite on the globe
    segment = Segment((0.0, 0.0), (180.0, 0.0))

    # Karney's method is most robust for antipodal points
    result = azimuth(segment.a, segment.b, strategy=Strategy.KARNEY)

    # Azimuth should be valid (east or west, ±π/2)
    assert isinstance(result, float)
    assert -math.pi <= result <= math.pi


def test_azimuth_cross_antimeridian() -> None:
    """Test azimuth for segment crossing the antimeridian."""
    # Segment crossing the international date line
    segment = Segment((170.0, 0.0), (-170.0, 0.0))

    result = azimuth(segment.a, segment.b)

    # Should return a valid azimuth
    assert isinstance(result, float)
    assert -math.pi <= result <= math.pi


def test_azimuth_polar_regions() -> None:
    """Test azimuth in polar regions."""
    # Segment near north pole
    segment = Segment((0.0, 85.0), (10.0, 85.0))

    result = azimuth(segment.a, segment.b)

    # Azimuth should be valid
    assert isinstance(result, float)
    assert -math.pi <= result <= math.pi

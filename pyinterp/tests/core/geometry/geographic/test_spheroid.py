# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test Spheroid class."""

from __future__ import annotations

import math

import pytest

from .....core.geometry.geographic import Spheroid


# WGS84 constants
WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_FLATTENING = 1.0 / 298.257223563


@pytest.fixture
def wgs84_spheroid() -> Spheroid:
    """Create a WGS84 spheroid for testing."""
    return Spheroid(WGS84_SEMI_MAJOR_AXIS, WGS84_FLATTENING)


@pytest.fixture
def wgs84_semi_minor_axis() -> float:
    """Calculate WGS84 semi-minor axis."""
    return WGS84_SEMI_MAJOR_AXIS * (1 - WGS84_FLATTENING)


class TestSpheroid:
    """Test Spheroid class."""

    def test_default_spheroid(self) -> None:
        """Test default Spheroid (WGS84)."""
        spheroid = Spheroid()

        # WGS84 parameters
        assert spheroid.semi_major_axis == pytest.approx(
            WGS84_SEMI_MAJOR_AXIS, rel=1e-9
        )
        assert spheroid.flattening == pytest.approx(WGS84_FLATTENING, rel=1e-9)

    def test_custom_spheroid(self, wgs84_spheroid: Spheroid) -> None:
        """Test custom Spheroid with specific parameters."""
        assert wgs84_spheroid.semi_major_axis == pytest.approx(
            WGS84_SEMI_MAJOR_AXIS, rel=1e-9
        )
        assert wgs84_spheroid.flattening == pytest.approx(
            WGS84_FLATTENING, rel=1e-9
        )

    def test_semi_major_axis_property(self) -> None:
        """Test semi_major_axis property."""
        semi_major_axis = 6378137.0
        spheroid = Spheroid(semi_major_axis, 0.1)

        assert spheroid.semi_major_axis == pytest.approx(
            semi_major_axis, rel=1e-9
        )

    def test_flattening_property(self) -> None:
        """Test flattening property."""
        flattening = 0.0033528106647
        spheroid = Spheroid(6378137.0, flattening)

        assert spheroid.flattening == pytest.approx(flattening, rel=1e-9)

    def test_semi_minor_axis(
        self, wgs84_spheroid: Spheroid, wgs84_semi_minor_axis: float
    ) -> None:
        """Test semi_minor_axis calculation."""
        # b = a(1 - f)
        assert wgs84_spheroid.semi_minor_axis() == pytest.approx(
            wgs84_semi_minor_axis, rel=1e-9
        )

    def test_first_eccentricity_squared(
        self, wgs84_spheroid: Spheroid
    ) -> None:
        """Test first eccentricity squared."""
        # e2 = 2f - f^2
        expected_e2 = 2 * WGS84_FLATTENING - WGS84_FLATTENING**2
        assert wgs84_spheroid.first_eccentricity_squared() == pytest.approx(
            expected_e2, rel=1e-9
        )

    def test_second_eccentricity_squared(
        self, wgs84_spheroid: Spheroid
    ) -> None:
        """Test second eccentricity squared."""
        # e'^2 = e2 / (1 - e2)
        e2 = 2 * WGS84_FLATTENING - WGS84_FLATTENING**2
        expected_e2_prime = e2 / (1 - e2)
        assert wgs84_spheroid.second_eccentricity_squared() == pytest.approx(
            expected_e2_prime, rel=1e-9
        )

    def test_linear_eccentricity(self, wgs84_spheroid: Spheroid) -> None:
        """Test linear eccentricity."""
        e2 = 2 * WGS84_FLATTENING - WGS84_FLATTENING**2
        expected_linear_eccentricity = WGS84_SEMI_MAJOR_AXIS * math.sqrt(e2)
        assert wgs84_spheroid.linear_eccentricity() == pytest.approx(
            expected_linear_eccentricity, rel=1e-9
        )

    def test_axis_ratio(self, wgs84_spheroid: Spheroid) -> None:
        """Test axis ratio (b/a)."""
        expected_ratio = 1 - WGS84_FLATTENING
        assert wgs84_spheroid.axis_ratio() == pytest.approx(
            expected_ratio, rel=1e-9
        )

    def test_equatorial_circumference(self, wgs84_spheroid: Spheroid) -> None:
        """Test equatorial circumference calculation."""
        # Circumference = 2 * pi * a
        expected_circumference = 2 * math.pi * WGS84_SEMI_MAJOR_AXIS
        assert wgs84_spheroid.equatorial_circumference() == pytest.approx(
            expected_circumference, rel=1e-9
        )

    def test_equatorial_circumference_using_semi_minor_axis(
        self, wgs84_spheroid: Spheroid
    ) -> None:
        """Test equatorial circumference using semi_minor_axis flag."""
        circ_semi_major = wgs84_spheroid.equatorial_circumference(
            semi_major_axis=True
        )
        circ_semi_minor = wgs84_spheroid.equatorial_circumference(
            semi_major_axis=False
        )

        assert circ_semi_major > circ_semi_minor

    def test_equatorial_radius_of_curvature(
        self, wgs84_spheroid: Spheroid, wgs84_semi_minor_axis: float
    ) -> None:
        """Test equatorial radius of curvature."""
        # Equatorial radius of curvature = b^2 / a
        expected_erc = (
            wgs84_semi_minor_axis * wgs84_semi_minor_axis
        ) / WGS84_SEMI_MAJOR_AXIS
        assert (
            wgs84_spheroid.equatorial_radius_of_curvature()
            == pytest.approx(expected_erc, rel=1e-9)
        )

    def test_polar_radius_of_curvature(
        self, wgs84_spheroid: Spheroid, wgs84_semi_minor_axis: float
    ) -> None:
        """Test polar radius of curvature."""
        # At pole: c = a^2 / b
        expected_polar_rc = (WGS84_SEMI_MAJOR_AXIS**2) / wgs84_semi_minor_axis
        assert wgs84_spheroid.polar_radius_of_curvature() == pytest.approx(
            expected_polar_rc, rel=1e-9
        )

    def test_geocentric_radius(
        self, wgs84_spheroid: Spheroid, wgs84_semi_minor_axis: float
    ) -> None:
        """Test geocentric radius at various latitudes."""
        # At equator (latitude = 0)
        radius_equator = wgs84_spheroid.geocentric_radius(0.0)
        assert radius_equator == pytest.approx(WGS84_SEMI_MAJOR_AXIS, rel=1e-9)

        # At pole (latitude = 90)
        radius_pole = wgs84_spheroid.geocentric_radius(90.0)
        assert radius_pole == pytest.approx(wgs84_semi_minor_axis, rel=1e-9)

        # At 45 degrees should be between equatorial and polar radius
        radius_45 = wgs84_spheroid.geocentric_radius(45.0)
        assert wgs84_semi_minor_axis < radius_45 < WGS84_SEMI_MAJOR_AXIS

    def test_authalic_radius(self, wgs84_spheroid: Spheroid) -> None:
        """Test authalic radius."""
        authalic = wgs84_spheroid.authalic_radius()
        assert authalic > 0

        # Should be close to semi_major_axis for small flattening
        assert authalic == pytest.approx(WGS84_SEMI_MAJOR_AXIS, rel=0.01)

    def test_mean_radius(
        self, wgs84_spheroid: Spheroid, wgs84_semi_minor_axis: float
    ) -> None:
        """Test mean radius."""
        mean = wgs84_spheroid.mean_radius()
        assert mean > 0

        # Should be between semi_minor and semi_major axis
        assert wgs84_semi_minor_axis < mean < WGS84_SEMI_MAJOR_AXIS

    def test_volumetric_radius(
        self, wgs84_spheroid: Spheroid, wgs84_semi_minor_axis: float
    ) -> None:
        """Test volumetric radius."""
        volumetric = wgs84_spheroid.volumetric_radius()
        assert volumetric > 0

        # Should be between semi_minor and semi_major axis
        assert wgs84_semi_minor_axis < volumetric < WGS84_SEMI_MAJOR_AXIS

    def test_equality(self, wgs84_spheroid: Spheroid) -> None:
        """Test equality operator."""
        spheroid_same = Spheroid(WGS84_SEMI_MAJOR_AXIS, WGS84_FLATTENING)
        spheroid_different = Spheroid(WGS84_SEMI_MAJOR_AXIS, 0.1)

        assert wgs84_spheroid == spheroid_same
        assert not (wgs84_spheroid == spheroid_different)

    def test_inequality(self, wgs84_spheroid: Spheroid) -> None:
        """Test inequality operator."""
        spheroid_same = Spheroid(WGS84_SEMI_MAJOR_AXIS, WGS84_FLATTENING)
        spheroid_different = Spheroid(WGS84_SEMI_MAJOR_AXIS, 0.1)

        assert not (wgs84_spheroid != spheroid_same)
        assert wgs84_spheroid != spheroid_different

    def test_wgs84_default(self, wgs84_spheroid: Spheroid) -> None:
        """Test that default spheroid is WGS84."""
        default = Spheroid()

        assert default == wgs84_spheroid

    def test_sphere_flatness_zero(self) -> None:
        """Test spheroid with zero flattening (perfect sphere)."""
        radius = 6371000.0
        sphere = Spheroid(radius, 0.0)

        assert sphere.semi_major_axis == pytest.approx(radius)
        assert sphere.semi_minor_axis() == pytest.approx(radius)
        assert sphere.axis_ratio() == pytest.approx(1.0)
        assert sphere.first_eccentricity_squared() == pytest.approx(0.0)

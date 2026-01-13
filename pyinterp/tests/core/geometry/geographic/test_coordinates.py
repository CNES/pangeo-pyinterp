# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test Coordinates class."""

from __future__ import annotations

import numpy as np
import pytest

from .....core.geometry.geographic import Coordinates, Spheroid


class TestCoordinates:
    """Test Coordinates class."""

    def test_default_coordinates(self) -> None:
        """Test default Coordinates (WGS84)."""
        coords = Coordinates()

        # Should have WGS84 spheroid
        assert coords.spheroid.semi_major_axis == pytest.approx(
            6378137.0, rel=1e-9
        )
        assert coords.spheroid.flattening == pytest.approx(
            1.0 / 298.257223563, rel=1e-9
        )

    def test_custom_spheroid_initialization(self) -> None:
        """Test Coordinates initialization with custom spheroid."""
        spheroid = Spheroid(6378137.0, 1.0 / 298.257223563)
        coords = Coordinates(spheroid)

        assert coords.spheroid == spheroid

    def test_spheroid_property(self) -> None:
        """Test spheroid property."""
        spheroid = Spheroid(6378137.0, 0.05)
        coords = Coordinates(spheroid)

        assert coords.spheroid == spheroid

    def test_ecef_to_lla_single_point_greenwich(self) -> None:
        """Test ECEF to LLA conversion at Greenwich meridian."""
        coords = Coordinates()

        # Point on Greenwich meridian at equator
        # ECEF coordinates for (0°, 0°, 0m altitude)
        x = np.array([6378137.0])
        y = np.array([0.0])
        z = np.array([0.0])

        lon, lat, alt = coords.ecef_to_lla(x, y, z)

        assert lon[0] == pytest.approx(0.0, abs=1e-6)
        assert lat[0] == pytest.approx(0.0, abs=1e-6)
        assert alt[0] == pytest.approx(0.0, abs=1e-3)

    def test_ecef_to_lla_single_point_north_pole(self) -> None:
        """Test ECEF to LLA conversion at North Pole."""
        coords = Coordinates()

        # Point at North Pole
        b = 6378137.0 * (1 - 1.0 / 298.257223563)
        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([b])

        lon, lat, alt = coords.ecef_to_lla(x, y, z)

        # Latitude should be 90 degrees
        assert lat[0] == pytest.approx(90.0, abs=1e-6)
        # Longitude is undefined at pole but should be finite
        assert np.isfinite(lon[0])
        # Altitude should be near zero
        assert alt[0] == pytest.approx(0.0, abs=1e-3)

    def test_lla_to_ecef_single_point_greenwich(self) -> None:
        """Test LLA to ECEF conversion at Greenwich meridian."""
        coords = Coordinates()

        # Point on Greenwich meridian at equator with zero altitude
        lon = np.array([0.0])
        lat = np.array([0.0])
        alt = np.array([0.0])

        x, y, z = coords.lla_to_ecef(lon, lat, alt)

        assert x[0] == pytest.approx(6378137.0, rel=1e-9)
        assert y[0] == pytest.approx(0.0, abs=1e-3)
        assert z[0] == pytest.approx(0.0, abs=1e-3)

    def test_lla_to_ecef_single_point_north_pole(self) -> None:
        """Test LLA to ECEF conversion at North Pole."""
        coords = Coordinates()

        # Point at North Pole with zero altitude
        lon = np.array([0.0])
        lat = np.array([90.0])
        alt = np.array([0.0])

        x, y, z = coords.lla_to_ecef(lon, lat, alt)

        assert x[0] == pytest.approx(0.0, abs=1e-3)
        assert y[0] == pytest.approx(0.0, abs=1e-3)
        b = 6378137.0 * (1 - 1.0 / 298.257223563)
        assert z[0] == pytest.approx(b, rel=1e-9)

    def test_ecef_lla_roundtrip_single(self) -> None:
        """Test roundtrip conversion ECEF -> LLA -> ECEF."""
        coords = Coordinates()

        # Original ECEF coordinates
        x_orig = np.array([6378137.0])
        y_orig = np.array([1000000.0])
        z_orig = np.array([2000000.0])

        # Convert to LLA
        lon, lat, alt = coords.ecef_to_lla(x_orig, y_orig, z_orig)

        # Convert back to ECEF
        x_back, y_back, z_back = coords.lla_to_ecef(lon, lat, alt)

        # Should match original values
        assert x_back[0] == pytest.approx(x_orig[0], rel=1e-9)
        assert y_back[0] == pytest.approx(y_orig[0], rel=1e-9)
        assert z_back[0] == pytest.approx(z_orig[0], rel=1e-9)

    def test_ecef_lla_roundtrip_multiple(self) -> None:
        """Test roundtrip conversion with multiple points."""
        coords = Coordinates()

        # Original ECEF coordinates
        x_orig = np.array([6378137.0, 6378137.0 * 0.8, 6378137.0 * 0.6])
        y_orig = np.array([0.0, 1000000.0, 2000000.0])
        z_orig = np.array([0.0, 2000000.0, 3000000.0])

        # Convert to LLA
        lon, lat, alt = coords.ecef_to_lla(x_orig, y_orig, z_orig)

        # Convert back to ECEF
        x_back, y_back, z_back = coords.lla_to_ecef(lon, lat, alt)

        # Should match original values
        np.testing.assert_allclose(x_back, x_orig, rtol=1e-9)
        np.testing.assert_allclose(y_back, y_orig, rtol=1e-9)
        np.testing.assert_allclose(z_back, z_orig, rtol=1e-9)

    def test_lla_ecef_roundtrip_single(self) -> None:
        """Test roundtrip conversion LLA -> ECEF -> LLA."""
        coords = Coordinates()

        # Original geographic coordinates
        lon_orig = np.array([45.0])
        lat_orig = np.array([30.0])
        alt_orig = np.array([1000.0])

        # Convert to ECEF
        x, y, z = coords.lla_to_ecef(lon_orig, lat_orig, alt_orig)

        # Convert back to LLA
        lon_back, lat_back, alt_back = coords.ecef_to_lla(x, y, z)

        # Should match original values
        assert lon_back[0] == pytest.approx(lon_orig[0], rel=1e-9)
        assert lat_back[0] == pytest.approx(lat_orig[0], rel=1e-9)
        assert alt_back[0] == pytest.approx(alt_orig[0], rel=1e-6)

    def test_lla_ecef_roundtrip_multiple(self) -> None:
        """Test roundtrip conversion LLA -> ECEF -> LLA with multiple points."""
        coords = Coordinates()

        # Original geographic coordinates
        lon_orig = np.array([0.0, 45.0, 90.0, 180.0, -45.0])
        lat_orig = np.array([0.0, 30.0, 45.0, -60.0, -30.0])
        alt_orig = np.array([0.0, 1000.0, 5000.0, -1000.0, 500.0])

        # Convert to ECEF
        x, y, z = coords.lla_to_ecef(lon_orig, lat_orig, alt_orig)

        # Convert back to LLA
        lon_back, lat_back, alt_back = coords.ecef_to_lla(x, y, z)

        # Should match original values
        np.testing.assert_allclose(lon_back, lon_orig, rtol=1e-9)
        np.testing.assert_allclose(lat_back, lat_orig, rtol=1e-9)
        np.testing.assert_allclose(alt_back, alt_orig, rtol=1e-6)

    def test_ecef_to_lla_with_altitude(self) -> None:
        """Test ECEF to LLA with non-zero altitude."""
        coords = Coordinates()

        # Point with altitude of 1000m
        altitude = 1000.0
        lon_expected = 45.0
        lat_expected = 30.0

        # First convert to ECEF
        lon = np.array([lon_expected])
        lat = np.array([lat_expected])
        alt = np.array([altitude])
        x, y, z = coords.lla_to_ecef(lon, lat, alt)

        # Then convert back
        lon_back, lat_back, alt_back = coords.ecef_to_lla(x, y, z)

        assert lon_back[0] == pytest.approx(lon_expected, rel=1e-9)
        assert lat_back[0] == pytest.approx(lat_expected, rel=1e-9)
        assert alt_back[0] == pytest.approx(altitude, rel=1e-9)

    def test_transform_same_system(self) -> None:
        """Test transform between same coordinate systems."""
        coords = Coordinates()

        lon = np.array([45.0])
        lat = np.array([30.0])
        alt = np.array([1000.0])

        # Transform to same system
        lon_t, lat_t, alt_t = coords.transform(coords, lon, lat, alt)

        # Should be identical
        np.testing.assert_allclose(lon_t, lon, rtol=1e-9)
        np.testing.assert_allclose(lat_t, lat, rtol=1e-9)
        np.testing.assert_allclose(alt_t, alt, rtol=1e-9)

    def test_transform_different_spheroid(self) -> None:
        """Test transform between different spheroid systems."""
        # WGS84
        spheroid_wgs84 = Spheroid(6378137.0, 1.0 / 298.257223563)
        coords_wgs84 = Coordinates(spheroid_wgs84)

        # Different spheroid
        spheroid_other = Spheroid(6378137.0, 0.05)
        coords_other = Coordinates(spheroid_other)

        lon = np.array([0.0])
        lat = np.array([0.0])
        alt = np.array([0.0])

        # Transform from WGS84 to other
        lon_t, lat_t, alt_t = coords_wgs84.transform(
            coords_other, lon, lat, alt
        )

        # At equator/Greenwich, should be minimal difference
        assert lon_t[0] == pytest.approx(lon[0], abs=1e-9)
        assert lat_t[0] == pytest.approx(lat[0], abs=1e-9)
        assert alt_t[0] == pytest.approx(alt[0], abs=1e-9)

    def test_transform_multiple_points(self) -> None:
        """Test transform with multiple points."""
        coords = Coordinates()

        lon = np.array([0.0, 45.0, 90.0])
        lat = np.array([0.0, 30.0, 45.0])
        alt = np.array([0.0, 1000.0, 5000.0])

        # Transform to same system
        lon_t, lat_t, alt_t = coords.transform(coords, lon, lat, alt)

        # Should be identical
        np.testing.assert_allclose(lon_t, lon, rtol=1e-9)
        np.testing.assert_allclose(lat_t, lat, rtol=1e-9)
        np.testing.assert_allclose(alt_t, alt, rtol=1e-9)

    def test_ecef_to_lla_with_num_threads(self) -> None:
        """Test ECEF to LLA conversion with multithreading."""
        coords = Coordinates()

        # Multiple points
        x = np.array([6378137.0, 6378137.0 * 0.8, 6378137.0 * 0.6])
        y = np.array([0.0, 1000000.0, 2000000.0])
        z = np.array([0.0, 2000000.0, 3000000.0])

        # Single thread
        lon_1, lat_1, alt_1 = coords.ecef_to_lla(x, y, z, num_threads=1)

        # Multiple threads
        lon_n, lat_n, alt_n = coords.ecef_to_lla(x, y, z, num_threads=4)

        # Results should be identical
        np.testing.assert_allclose(lon_1, lon_n, rtol=1e-9)
        np.testing.assert_allclose(lat_1, lat_n, rtol=1e-9)
        np.testing.assert_allclose(alt_1, alt_n, rtol=1e-9)

    def test_lla_to_ecef_with_num_threads(self) -> None:
        """Test LLA to ECEF conversion with multithreading."""
        coords = Coordinates()

        # Multiple points
        lon = np.array([0.0, 45.0, 90.0])
        lat = np.array([0.0, 30.0, 45.0])
        alt = np.array([0.0, 1000.0, 5000.0])

        # Single thread
        x_1, y_1, z_1 = coords.lla_to_ecef(lon, lat, alt, num_threads=1)

        # Multiple threads
        x_n, y_n, z_n = coords.lla_to_ecef(lon, lat, alt, num_threads=4)

        # Results should be identical
        np.testing.assert_allclose(x_1, x_n, rtol=1e-9)
        np.testing.assert_allclose(y_1, y_n, rtol=1e-9)
        np.testing.assert_allclose(z_1, z_n, rtol=1e-9)

    def test_transform_with_num_threads(self) -> None:
        """Test transform with multithreading."""
        coords = Coordinates()

        # Multiple points
        lon = np.array([0.0, 45.0, 90.0])
        lat = np.array([0.0, 30.0, 45.0])
        alt = np.array([0.0, 1000.0, 5000.0])

        # Single thread
        lon_1, lat_1, alt_1 = coords.transform(
            coords, lon, lat, alt, num_threads=1
        )

        # Multiple threads
        lon_n, lat_n, alt_n = coords.transform(
            coords, lon, lat, alt, num_threads=4
        )

        # Results should be identical
        np.testing.assert_allclose(lon_1, lon_n, rtol=1e-9)
        np.testing.assert_allclose(lat_1, lat_n, rtol=1e-9)
        np.testing.assert_allclose(alt_1, alt_n, rtol=1e-9)

    def test_array_shapes(self) -> None:
        """Test that input arrays must have matching shapes."""
        coords = Coordinates()

        x = np.array([6378137.0, 6378137.0])
        y = np.array([0.0, 1000000.0])
        z = np.array([0.0])  # Wrong shape

        with pytest.raises(
            ValueError, match="x, z could not be broadcast together"
        ):
            coords.ecef_to_lla(x, y, z)

    def test_empty_arrays(self) -> None:
        """Test with empty arrays."""
        coords = Coordinates()

        x = np.array([])
        y = np.array([])
        z = np.array([])

        lon, lat, alt = coords.ecef_to_lla(x, y, z)

        assert len(lon) == 0
        assert len(lat) == 0
        assert len(alt) == 0

    def test_large_array(self) -> None:
        """Test with large array of points."""
        coords = Coordinates()

        n = 10000
        rng = np.random.default_rng(42)
        lon_orig = rng.uniform(-180, 180, n)
        lat_orig = rng.uniform(-90, 90, n)
        alt_orig = rng.uniform(-10000, 50000, n)

        # Convert to ECEF and back
        x, y, z = coords.lla_to_ecef(lon_orig, lat_orig, alt_orig)
        lon_back, lat_back, alt_back = coords.ecef_to_lla(x, y, z)

        # Should match
        np.testing.assert_allclose(lon_back, lon_orig, rtol=1e-9)
        np.testing.assert_allclose(lat_back, lat_orig, rtol=1e-9)
        np.testing.assert_allclose(alt_back, alt_orig, rtol=1e-6)

    def test_negative_altitude(self) -> None:
        """Test with negative altitudes (below surface)."""
        coords = Coordinates()

        lon = np.array([0.0])
        lat = np.array([45.0])
        alt = np.array([-1000.0])  # 1000m below surface

        x, y, z = coords.lla_to_ecef(lon, lat, alt)
        lon_back, lat_back, alt_back = coords.ecef_to_lla(x, y, z)

        assert lon_back[0] == pytest.approx(lon[0], rel=1e-9)
        assert lat_back[0] == pytest.approx(lat[0], rel=1e-9)
        assert alt_back[0] == pytest.approx(alt[0], rel=1e-6)

    def test_extreme_latitudes(self) -> None:
        """Test with extreme latitude values."""
        coords = Coordinates()

        lon = np.array([0.0, 0.0, 0.0])
        lat = np.array([-90.0, 0.0, 90.0])
        alt = np.array([0.0, 0.0, 0.0])

        x, y, z = coords.lla_to_ecef(lon, lat, alt)
        lon_back, lat_back, alt_back = coords.ecef_to_lla(x, y, z)

        np.testing.assert_allclose(lon_back, lon, rtol=1e-9)
        np.testing.assert_allclose(lat_back, lat, rtol=1e-9)
        np.testing.assert_allclose(alt_back, alt, atol=1e-9)

    def test_antimeridian_crossing(self) -> None:
        """Test coordinates near the antimeridian."""
        coords = Coordinates()

        lon = np.array([179.9, -179.9])
        lat = np.array([0.0, 0.0])
        alt = np.array([0.0, 0.0])

        x, y, z = coords.lla_to_ecef(lon, lat, alt)
        lon_back, lat_back, alt_back = coords.ecef_to_lla(x, y, z)

        # Both should convert properly
        np.testing.assert_allclose(lon_back, lon, rtol=1e-9)
        np.testing.assert_allclose(lat_back, lat, rtol=1e-9)
        np.testing.assert_allclose(alt_back, alt, rtol=1e-9)

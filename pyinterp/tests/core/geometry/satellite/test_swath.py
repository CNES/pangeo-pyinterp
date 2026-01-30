# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test satellite swath calculation functions."""

from __future__ import annotations

import numpy as np
import pytest

from .....core.geometry import satellite
from .....core.geometry.geographic import Coordinates, Spheroid


class TestCalculateSwath:
    """Tests for calculate_swath function."""

    @staticmethod
    def create_equatorial_track(
        num_points: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a simple equatorial nadir track."""
        lon_nadir = np.linspace(-5.0, 5.0, num_points)
        lat_nadir = np.zeros(num_points)
        return lon_nadir, lat_nadir

    @staticmethod
    def create_polar_track(
        num_points: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a polar (north-south) nadir track."""
        lon_nadir = np.zeros(num_points)
        lat_nadir = np.linspace(-45.0, 45.0, num_points)
        return lon_nadir, lat_nadir

    @staticmethod
    def create_swot_track(
        num_points: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a realistic SWOT orbit segment."""
        lon_nadir = np.linspace(10.0, 15.0, num_points)
        lat_nadir = np.linspace(-30.0, 30.0, num_points)
        return lon_nadir, lat_nadir

    def test_swath_basic_dimensions(self) -> None:
        """Test that swath calculation returns correct dimensions."""
        lon_nadir, lat_nadir = self.create_equatorial_track(10)

        # SWOT parameters
        delta_ac = 2000.0  # 2 km across-track resolution (meters)
        half_gap = 2000.0  # 2 km gap (meters)
        half_swath = 35  # 35 pixels per side

        lon_swath, lat_swath = satellite.calculate_swath(
            lon_nadir, lat_nadir, delta_ac, half_gap, half_swath
        )

        # Check dimensions
        assert lon_swath.shape == (10, 70), "Longitude swath shape incorrect"
        assert lat_swath.shape == (10, 70), "Latitude swath shape incorrect"

        # Check all values are finite
        assert np.all(np.isfinite(lon_swath)), "Non-finite longitude values"
        assert np.all(np.isfinite(lat_swath)), "Non-finite latitude values"

    def test_swath_equatorial_symmetry(self) -> None:
        """Test that equatorial swath is symmetric left/right."""
        lon_nadir, lat_nadir = self.create_equatorial_track(10)

        delta_ac = 2000.0
        half_gap = 2000.0
        half_swath = 35

        lon_swath, lat_swath = satellite.calculate_swath(
            lon_nadir, lat_nadir, delta_ac, half_gap, half_swath
        )

        # For equatorial track, left and right sides should be symmetric in
        # latitude
        for i in range(lon_swath.shape[0]):
            for j in range(half_swath):
                left_idx = j
                right_idx = 2 * half_swath - 1 - j

                left_delta = abs(lat_swath[i, left_idx] - lat_nadir[i])
                right_delta = abs(lat_swath[i, right_idx] - lat_nadir[i])

                # Allow small tolerance for numerical differences
                assert abs(left_delta - right_delta) < 0.01, (
                    f"Asymmetry at row {i}, pixel {j}"
                )

    def test_swath_bounds(self) -> None:
        """Test that swath coordinates are within valid geographic bounds."""
        lon_nadir, lat_nadir = self.create_swot_track(20)

        delta_ac = 2000.0
        half_gap = 2000.0
        half_swath = 35

        lon_swath, lat_swath = satellite.calculate_swath(
            lon_nadir, lat_nadir, delta_ac, half_gap, half_swath
        )

        # Longitude should be in [-180, 180]
        assert np.all(lon_swath >= -180.0), "Longitude below -180"
        assert np.all(lon_swath <= 180.0), "Longitude above 180"

        # Latitude should be in [-90, 90]
        assert np.all(lat_swath >= -90.0), "Latitude below -90"
        assert np.all(lat_swath <= 90.0), "Latitude above 90"

    def test_swath_edge_distances(self) -> None:
        """Test that swath edges are at expected distances from nadir."""
        lon_nadir, lat_nadir = self.create_equatorial_track(10)

        # SWOT parameters
        delta_ac = 2000.0  # meters
        half_gap = 2000.0  # meters
        half_swath = 35  # pixels

        lon_swath, lat_swath = satellite.calculate_swath(
            lon_nadir, lat_nadir, delta_ac, half_gap, half_swath
        )

        # Expected edge distance: (half_swath - 1) * delta_ac + half_gap
        # = 34 * 2000 + 2000 = 70,000 meters = 70 km
        expected_edge_dist = (half_swath - 1) * delta_ac + half_gap

        # Check a middle point (avoid boundaries)
        mid_idx = 5
        coords = Coordinates()

        # Convert nadir to ECEF
        x_nadir, y_nadir, z_nadir = coords.lla_to_ecef(
            np.array([lon_nadir[mid_idx]]),
            np.array([lat_nadir[mid_idx]]),
            np.array([0.0]),
        )

        # Check left edge
        x_left, y_left, z_left = coords.lla_to_ecef(
            np.array([lon_swath[mid_idx, 0]]),
            np.array([lat_swath[mid_idx, 0]]),
            np.array([0.0]),
        )
        dist_left = np.sqrt(
            (x_left[0] - x_nadir[0]) ** 2
            + (y_left[0] - y_nadir[0]) ** 2
            + (z_left[0] - z_nadir[0]) ** 2
        )

        # Allow 5% tolerance
        assert (
            abs(dist_left - expected_edge_dist) < expected_edge_dist * 0.05
        ), (
            f"Left edge distance {dist_left} "
            f"not near expected {expected_edge_dist}"
        )

        # Check right edge
        x_right, y_right, z_right = coords.lla_to_ecef(
            np.array([lon_swath[mid_idx, -1]]),
            np.array([lat_swath[mid_idx, -1]]),
            np.array([0.0]),
        )
        dist_right = np.sqrt(
            (x_right[0] - x_nadir[0]) ** 2
            + (y_right[0] - y_nadir[0]) ** 2
            + (z_right[0] - z_nadir[0]) ** 2
        )

        assert (
            abs(dist_right - expected_edge_dist) < expected_edge_dist * 0.05
        ), (
            f"Right edge distance {dist_right} "
            f"not near expected {expected_edge_dist}"
        )

        # Left and right should be symmetric
        assert abs(dist_left - dist_right) < expected_edge_dist * 0.01, (
            "Left and right edges not symmetric"
        )

    def test_swath_with_custom_spheroid(self) -> None:
        """Test swath calculation with custom spheroid."""
        lon_nadir, lat_nadir = self.create_equatorial_track(5)

        # Use a sphere instead of WGS84
        sphere = Spheroid(6371000.0, 0.0)

        lon_swath, lat_swath = satellite.calculate_swath(
            lon_nadir, lat_nadir, 2000.0, 2000.0, 10, sphere
        )

        assert lon_swath.shape == (5, 20)
        assert lat_swath.shape == (5, 20)
        assert np.all(np.isfinite(lon_swath))
        assert np.all(np.isfinite(lat_swath))

    def test_swath_minimum_points(self) -> None:
        """Test swath calculation with minimum number of points."""
        # Minimum is 3 points for satellite_direction calculation
        lon_nadir = np.array([0.0, 1.0, 2.0])
        lat_nadir = np.array([0.0, 0.0, 0.0])

        lon_swath, lat_swath = satellite.calculate_swath(
            lon_nadir, lat_nadir, 2000.0, 2000.0, 5
        )

        assert lon_swath.shape == (3, 10)
        assert lat_swath.shape == (3, 10)
        assert np.all(np.isfinite(lon_swath))
        assert np.all(np.isfinite(lat_swath))

    def test_swath_mismatched_sizes(self) -> None:
        """Test that mismatched input sizes raise an error."""
        lon_nadir = np.array([0.0, 1.0, 2.0])
        lat_nadir = np.array([0.0, 1.0])  # Wrong size

        with pytest.raises(ValueError, match="broadcast"):
            satellite.calculate_swath(lon_nadir, lat_nadir, 2000.0, 2000.0, 5)

    def test_swath_realistic_swot_parameters(self) -> None:
        """Test with actual SWOT mission parameters."""
        # SWOT parameters
        across_track_resolution = 2.0  # km
        half_swath_km = 70.0  # km
        half_gap_km = 2.0  # km

        # Compute number of pixels
        half_swath_pixels = (
            int((half_swath_km - half_gap_km) / across_track_resolution) + 1
        )

        # Convert to meters
        delta_ac = across_track_resolution * 1000.0
        half_gap = half_gap_km * 1000.0

        assert half_swath_pixels == 35, "SWOT parameter calculation incorrect"

        lon_nadir, lat_nadir = self.create_swot_track(20)

        lon_swath, lat_swath = satellite.calculate_swath(
            lon_nadir, lat_nadir, delta_ac, half_gap, half_swath_pixels
        )

        # Should produce 70 pixels per nadir point (35 left + 35 right)
        assert lon_swath.shape == (20, 70)
        assert lat_swath.shape == (20, 70)

    def test_swath_polar_track(self) -> None:
        """Test swath calculation for polar (north-south) track."""
        lon_nadir, lat_nadir = self.create_polar_track(10)

        lon_swath, _ = satellite.calculate_swath(
            lon_nadir, lat_nadir, 2000.0, 2000.0, 10
        )

        # For polar track along prime meridian, longitude should spread
        # symmetrically around 0
        for i in range(1, lon_swath.shape[0] - 1):
            lon_mean = (lon_swath[i, :].min() + lon_swath[i, :].max()) / 2.0
            # Should be centered near prime meridian (within 5 degrees)
            assert abs(lon_mean) < 5.0, (
                f"Longitude not centered at row {i}: {lon_mean}"
            )

    def test_swath_pixel_spacing(self) -> None:
        """Test that pixel spacing matches across-track resolution."""
        lon_nadir, lat_nadir = self.create_equatorial_track(10)

        delta_ac = 2000.0  # 2 km spacing
        half_gap = 2000.0
        half_swath = 10

        lon_swath, lat_swath = satellite.calculate_swath(
            lon_nadir, lat_nadir, delta_ac, half_gap, half_swath
        )

        # Check pixel 0 (leftmost) should be at distance ~(9*2000 + 2000) = 20km
        # Check pixel 1 should be at distance ~(8*2000 + 2000) = 18km
        # Difference should be approximately 2km = 2000m

        coords = Coordinates()
        mid_idx = 5

        x_nadir, y_nadir, z_nadir = coords.lla_to_ecef(
            np.array([lon_nadir[mid_idx]]),
            np.array([lat_nadir[mid_idx]]),
            np.array([0.0]),
        )

        # Get distances for two adjacent pixels on the left side
        x_0, y_0, z_0 = coords.lla_to_ecef(
            np.array([lon_swath[mid_idx, 0]]),
            np.array([lat_swath[mid_idx, 0]]),
            np.array([0.0]),
        )
        x_1, y_1, z_1 = coords.lla_to_ecef(
            np.array([lon_swath[mid_idx, 1]]),
            np.array([lat_swath[mid_idx, 1]]),
            np.array([0.0]),
        )

        dist_0 = np.sqrt(
            (x_0[0] - x_nadir[0]) ** 2
            + (y_0[0] - y_nadir[0]) ** 2
            + (z_0[0] - z_nadir[0]) ** 2
        )
        dist_1 = np.sqrt(
            (x_1[0] - x_nadir[0]) ** 2
            + (y_1[0] - y_nadir[0]) ** 2
            + (z_1[0] - z_nadir[0]) ** 2
        )

        pixel_spacing = abs(dist_0 - dist_1)

        # Should be approximately delta_ac (allow 10% tolerance due to
        # curvature)
        assert abs(pixel_spacing - delta_ac) < delta_ac * 0.1, (
            f"Pixel spacing {pixel_spacing} not near {delta_ac}"
        )

    def test_swath_different_resolutions(self) -> None:
        """Test swath with different across-track resolutions."""
        lon_nadir, lat_nadir = self.create_equatorial_track(5)

        resolutions = [1000.0, 2000.0, 5000.0]  # 1km, 2km, 5km

        for delta_ac in resolutions:
            lon_swath, lat_swath = satellite.calculate_swath(
                lon_nadir, lat_nadir, delta_ac, 2000.0, 10
            )

            assert lon_swath.shape == (5, 20)
            assert lat_swath.shape == (5, 20)
            assert np.all(np.isfinite(lon_swath))
            assert np.all(np.isfinite(lat_swath))

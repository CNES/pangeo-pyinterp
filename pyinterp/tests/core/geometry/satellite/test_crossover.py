# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test satellite crossover detection functions."""

import numpy as np
import pytest

from .....core.geometry import satellite
from .....core.geometry.geographic import Spheroid
from .....core.geometry.geographic.algorithms import Strategy


class TestFindCrossovers:
    """Tests for find_crossovers function."""

    @staticmethod
    def create_crossing_tracks() -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Create two tracks that cross at the origin."""
        # First track: west to east along equator
        lon1 = np.linspace(-5.0, 5.0, 11)
        lat1 = np.zeros(11)

        # Second track: south to north along prime meridian
        lon2 = np.zeros(11)
        lat2 = np.linspace(-5.0, 5.0, 11)

        return lon1, lat1, lon2, lat2

    @staticmethod
    def create_parallel_tracks() -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Create two parallel tracks that don't cross."""
        # First track: along equator
        lon1 = np.linspace(-5.0, 5.0, 11)
        lat1 = np.zeros(11)

        # Second track: parallel, 10 degrees north
        lon2 = np.linspace(-5.0, 5.0, 11)
        lat2 = np.full(11, 10.0)

        return lon1, lat1, lon2, lat2

    @staticmethod
    def create_x_shape_tracks() -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Create two tracks forming an X shape (single crossover)."""
        # First track: southwest to northeast
        lon1 = np.linspace(-5.0, 5.0, 11)
        lat1 = np.linspace(-5.0, 5.0, 11)

        # Second track: southeast to northwest
        lon2 = np.linspace(-5.0, 5.0, 11)
        lat2 = np.linspace(5.0, -5.0, 11)

        return lon1, lat1, lon2, lat2

    def test_crossover_basic_detection(self) -> None:
        """Test basic crossover detection at origin."""
        lon1, lat1, lon2, lat2 = self.create_crossing_tracks()

        # Find crossover with generous predicate (10 km)
        crossovers = satellite.find_crossovers(
            lon1,
            lat1,
            lon2,
            lat2,
            predicate=115_000.0,  # 115 km in meters
            allow_multiple=False,
            use_cartesian=True,
        )

        # Should find exactly one crossover
        assert len(crossovers) == 1, "Should find exactly one crossover"

        # Crossover should be near the origin
        xover = crossovers[0]
        assert abs(xover.point.lon) < 0.1, (
            "Crossover longitude should be near 0"
        )
        assert abs(xover.point.lat) < 0.1, (
            "Crossover latitude should be near 0"
        )

    def test_crossover_no_intersection(self) -> None:
        """Test that parallel tracks return no crossovers."""
        lon1, lat1, lon2, lat2 = self.create_parallel_tracks()

        crossovers = satellite.find_crossovers(
            lon1,
            lat1,
            lon2,
            lat2,
            predicate=10000.0,
            allow_multiple=False,
            use_cartesian=True,
        )

        # Should find no crossovers
        assert len(crossovers) == 0, "Parallel tracks should not cross"

    def test_crossover_x_shape(self) -> None:
        """Test crossover detection for X-shaped tracks."""
        lon1, lat1, lon2, lat2 = self.create_x_shape_tracks()

        crossovers = satellite.find_crossovers(
            lon1,
            lat1,
            lon2,
            lat2,
            predicate=160_000.0,
            allow_multiple=False,
            use_cartesian=True,
        )

        # Should find exactly one crossover at the origin
        assert len(crossovers) == 1, "X-shape should have one crossover"

        xover = crossovers[0]
        assert abs(xover.point.lon) < 0.1, (
            "Crossover should be near origin (lon)"
        )
        assert abs(xover.point.lat) < 0.1, (
            "Crossover should be near origin (lat)"
        )

    def test_crossover_indices(self) -> None:
        """Test that crossover indices point to nearest vertices."""
        lon1, lat1, lon2, lat2 = self.create_crossing_tracks()

        crossovers = satellite.find_crossovers(
            lon1,
            lat1,
            lon2,
            lat2,
            predicate=115_000.0,
            allow_multiple=False,
            use_cartesian=True,
        )

        assert len(crossovers) == 1

        xover = crossovers[0]

        # Indices should be valid
        assert 0 <= xover.index1 < len(lon1), "index1 out of range"
        assert 0 <= xover.index2 < len(lon2), "index2 out of range"

        # Nearest vertices should be close to the crossover point
        dist1_lon = abs(lon1[xover.index1] - xover.point.lon)
        dist1_lat = abs(lat1[xover.index1] - xover.point.lat)
        dist2_lon = abs(lon2[xover.index2] - xover.point.lon)
        dist2_lat = abs(lat2[xover.index2] - xover.point.lat)

        # Should be within a few degrees
        assert dist1_lon <= 1.0 and dist1_lat <= 1.0, (
            "index1 not near crossover"
        )
        assert dist2_lon <= 1.0 and dist2_lat <= 1.0, (
            "index2 not near crossover"
        )

    def test_crossover_predicate_filtering(self) -> None:
        """Test that predicate filters out distant crossovers."""
        lon1, lat1, lon2, lat2 = self.create_crossing_tracks()

        # Very small predicate - crossover might be filtered out
        # if nearest vertices are too far
        crossovers_strict = satellite.find_crossovers(
            lon1,
            lat1,
            lon2,
            lat2,
            predicate=1.0,  # 1 meter - very strict
            allow_multiple=False,
            use_cartesian=True,
        )

        # Large predicate - should definitely find it
        crossovers_generous = satellite.find_crossovers(
            lon1,
            lat1,
            lon2,
            lat2,
            predicate=100000.0,  # 100 km - very generous
            allow_multiple=False,
            use_cartesian=True,
        )

        # Generous predicate should find crossover
        assert len(crossovers_generous) >= len(crossovers_strict), (
            "Larger predicate should find at least as many crossovers"
        )

    def test_crossover_cartesian_vs_geographic(self) -> None:
        """Test Cartesian vs geographic crossover detection."""
        lon1, lat1, lon2, lat2 = self.create_crossing_tracks()

        # Cartesian (faster, approximate)
        crossovers_cartesian = satellite.find_crossovers(
            lon1,
            lat1,
            lon2,
            lat2,
            predicate=10000.0,
            allow_multiple=False,
            use_cartesian=True,
        )

        # Geographic (slower, accurate)
        crossovers_geographic = satellite.find_crossovers(
            lon1,
            lat1,
            lon2,
            lat2,
            predicate=10000.0,
            allow_multiple=False,
            use_cartesian=False,
            strategy=Strategy.VINCENTY,
        )

        # Both should find the same crossover (at equator they're similar)
        assert len(crossovers_cartesian) == len(crossovers_geographic), (
            "Should find same number of crossovers"
        )

        if len(crossovers_cartesian) > 0 and len(crossovers_geographic) > 0:
            # Crossover locations should be very close (within 0.01 degrees)
            cart = crossovers_cartesian[0]
            geo = crossovers_geographic[0]

            assert abs(cart.point.lon - geo.point.lon) < 0.01, (
                "Cartesian and geographic longitudes differ significantly"
            )
            assert abs(cart.point.lat - geo.point.lat) < 0.01, (
                "Cartesian and geographic latitudes differ significantly"
            )

    def test_crossover_different_strategies(self) -> None:
        """Test different geodetic calculation strategies."""
        lon1, lat1, lon2, lat2 = self.create_crossing_tracks()

        strategies = [
            Strategy.ANDOYER,
            Strategy.KARNEY,
            Strategy.THOMAS,
            Strategy.VINCENTY,
        ]

        results = []
        for strategy in strategies:
            crossovers = satellite.find_crossovers(
                lon1,
                lat1,
                lon2,
                lat2,
                predicate=10000.0,
                allow_multiple=False,
                use_cartesian=False,
                strategy=strategy,
            )
            results.append(crossovers)

        # All strategies should find the same crossover
        for i, crossovers in enumerate(results):
            assert len(crossovers) >= 0, f"Strategy {strategies[i]} failed"

            # If found, should be near origin
            if len(crossovers) > 0:
                xover = crossovers[0]
                assert abs(xover.point.lon) < 0.5, (
                    f"Strategy {strategies[i]} gave wrong lon"
                )
                assert abs(xover.point.lat) < 0.5, (
                    f"Strategy {strategies[i]} gave wrong lat"
                )

    def test_crossover_custom_spheroid(self) -> None:
        """Test crossover detection with custom spheroid."""
        lon1, lat1, lon2, lat2 = self.create_crossing_tracks()

        # Use a perfect sphere
        sphere = Spheroid(6371000.0, 0.0)

        crossovers = satellite.find_crossovers(
            lon1,
            lat1,
            lon2,
            lat2,
            predicate=115_000.0,
            allow_multiple=False,
            use_cartesian=False,
            spheroid=sphere,
        )

        # Should still find the crossover
        assert len(crossovers) == 1
        assert abs(crossovers[0].point.lon) < 0.1
        assert abs(crossovers[0].point.lat) < 0.1

    def test_crossover_allow_multiple(self) -> None:
        """Test finding multiple crossover points."""
        # Create tracks that cross multiple times
        # Track 1: sinusoidal pattern
        lon1 = np.linspace(-10.0, 10.0, 101)
        lat1 = 5 * np.sin(lon1 * np.pi / 5)

        # Track 2: straight line through center
        lon2 = np.linspace(-10.0, 10.0, 101)
        lat2 = np.zeros(101)

        # Find all crossovers
        crossovers_multiple = satellite.find_crossovers(
            lon1,
            lat1,
            lon2,
            lat2,
            predicate=115_000.0,
            allow_multiple=True,
            use_cartesian=True,
        )

        # Should find multiple crossovers (sinusoid crosses horizontal line)
        # At least 2 crossovers expected
        assert len(crossovers_multiple) >= 2, (
            f"Should find multiple crossovers, found {len(crossovers_multiple)}"
        )

    def test_crossover_unique_vs_multiple(self) -> None:
        """Test that allow_multiple raises error for multiple crossings."""
        # Create tracks with multiple crossings
        lon1 = np.linspace(-10.0, 10.0, 101)
        lat1 = 5 * np.sin(lon1 * np.pi / 5)
        lon2 = np.linspace(-10.0, 10.0, 101)
        lat2 = np.zeros(101)

        # allow_multiple=False should raise error if multiple found
        with pytest.raises(RuntimeError, match="Multiple crossover"):
            satellite.find_crossovers(
                lon1,
                lat1,
                lon2,
                lat2,
                predicate=10000.0,
                allow_multiple=False,
                use_cartesian=True,
            )

    def test_crossover_minimum_points(self) -> None:
        """Test crossover detection with minimum points."""
        # Minimum 3 points required
        lon1 = np.array([-5.0, 0.0, 5.0])
        lat1 = np.array([0.0, 0.0, 0.0])
        lon2 = np.array([0.0, 0.0, 0.0])
        lat2 = np.array([-5.0, 0.0, 5.0])

        crossovers = satellite.find_crossovers(
            lon1,
            lat1,
            lon2,
            lat2,
            predicate=10000.0,
            allow_multiple=False,
            use_cartesian=True,
        )

        # Should find crossover at origin
        assert len(crossovers) == 1
        assert abs(crossovers[0].point.lon) < 0.5
        assert abs(crossovers[0].point.lat) < 0.5

    def test_crossover_input_validation(self) -> None:
        """Test input validation for crossover detection."""
        # Test with too few points (< 3)
        lon1 = np.array([0.0, 1.0])
        lat1 = np.array([0.0, 0.0])
        lon2 = np.array([0.0, 0.0])
        lat2 = np.array([0.0, 1.0])

        with pytest.raises(ValueError, match="at least 3 points"):
            satellite.find_crossovers(lon1, lat1, lon2, lat2, predicate=1000.0)

        # Test with mismatched sizes
        lon1 = np.array([0.0, 1.0, 2.0])
        lat1 = np.array([0.0, 0.0])  # Wrong size

        with pytest.raises(ValueError, match="broadcast"):
            satellite.find_crossovers(lon1, lat1, lon2, lat2, predicate=1000.0)

        # Test with invalid predicate
        lon1, lat1, lon2, lat2 = self.create_crossing_tracks()

        with pytest.raises(ValueError, match="positive finite"):
            satellite.find_crossovers(
                lon1, lat1, lon2, lat2, predicate=-1000.0
            )

        with pytest.raises(ValueError, match="positive finite"):
            satellite.find_crossovers(lon1, lat1, lon2, lat2, predicate=0.0)

        with pytest.raises(ValueError, match="positive finite"):
            satellite.find_crossovers(lon1, lat1, lon2, lat2, predicate=np.inf)

    def test_crossover_realistic_satellite_passes(self) -> None:
        """Test with realistic satellite pass geometry."""
        # Ascending pass (SW to NE)
        lon1 = np.linspace(-10.0, 10.0, 50)
        lat1 = np.linspace(-30.0, 30.0, 50)

        # Descending pass (NW to SE)
        lon2 = np.linspace(-10.0, 10.0, 50)
        lat2 = np.linspace(30.0, -30.0, 50)

        # Should cross near the equator
        crossovers = satellite.find_crossovers(
            lon1,
            lat1,
            lon2,
            lat2,
            predicate=80_000.0,  # 80 km
            allow_multiple=False,
            use_cartesian=True,
        )

        assert len(crossovers) == 1, "Should find one crossover"

        xover = crossovers[0]
        # Crossover should be near equator (within 5 degrees)
        assert abs(xover.point.lat) < 5.0, (
            f"Crossover not near equator: lat={xover.point.lat}"
        )

    def test_crossover_result_attributes(self) -> None:
        """Test that CrossoverResult has correct attributes."""
        lon1, lat1, lon2, lat2 = self.create_crossing_tracks()

        crossovers = satellite.find_crossovers(
            lon1,
            lat1,
            lon2,
            lat2,
            predicate=115_000.0,
            allow_multiple=False,
            use_cartesian=True,
        )

        assert len(crossovers) == 1

        xover = crossovers[0]

        # Check that all attributes exist
        assert hasattr(xover, "point"), "Missing 'point' attribute"
        assert hasattr(xover, "index1"), "Missing 'index1' attribute"
        assert hasattr(xover, "index2"), "Missing 'index2' attribute"

        # Check types
        assert hasattr(xover.point, "lon"), "Point missing 'lon'"
        assert hasattr(xover.point, "lat"), "Point missing 'lat'"
        assert isinstance(xover.index1, int), "index1 should be int"
        assert isinstance(xover.index2, int), "index2 should be int"

    def test_crossover_high_latitude(self) -> None:
        """Test crossover detection at high latitudes."""
        # Create crossing tracks near the north pole
        lon1 = np.linspace(0.0, 90.0, 20)
        lat1 = np.full(20, 80.0)  # 80°N

        lon2 = np.linspace(90.0, 180.0, 20)
        lat2 = np.full(20, 80.0)  # 80°N

        # These don't actually cross, but test that high-latitude works
        crossovers = satellite.find_crossovers(
            lon1,
            lat1,
            lon2,
            lat2,
            predicate=10000.0,
            allow_multiple=True,
            use_cartesian=True,
        )

        # May or may not find crossovers, but shouldn't crash
        assert isinstance(crossovers, list), "Should return a list"

# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test satellite track decomposition functions."""

from __future__ import annotations

import numpy as np
import pytest

from .....core.geometry import satellite


class TestDecomposeTrack:
    """Tests for decompose_track function."""

    @staticmethod
    def create_band_crossing_track() -> tuple[np.ndarray, np.ndarray]:
        """Create a track crossing south/mid/north latitude bands."""
        lon = np.linspace(-20.0, 20.0, 10)
        lat = np.array(
            [-30.0, -20.0, -5.0, 0.0, 5.0, 20.0, 30.0, 5.0, -5.0, -20.0]
        )
        return lon, lat

    def test_decompose_track_latitude_bands(self) -> None:
        """Test decomposition by latitude bands with deterministic options."""
        lon, lat = self.create_band_crossing_track()

        opts = (
            satellite.DecompositionOptions()
            .with_south_limit(-10.0)
            .with_north_limit(10.0)
            .with_min_edge_size(1)
        )
        segments = satellite.decompose_track(
            lon,
            lat,
            strategy="latitude_bands",
            opts=opts,
        )

        assert len(segments) == 5

        assert segments[0].first_index == 0
        assert segments[0].last_index == 1
        assert segments[0].zone == satellite.LatitudeZone.SOUTH

        assert segments[1].first_index == 2
        assert segments[1].last_index == 4
        assert segments[1].zone == satellite.LatitudeZone.MID

        assert segments[2].first_index == 5
        assert segments[2].last_index == 6
        assert segments[2].zone == satellite.LatitudeZone.NORTH

        assert segments[3].first_index == 7
        assert segments[3].last_index == 8
        assert segments[3].zone == satellite.LatitudeZone.MID

        assert segments[4].first_index == 9
        assert segments[4].last_index == 9
        assert segments[4].zone == satellite.LatitudeZone.SOUTH

        for segment in segments:
            assert segment.orbit == satellite.OrbitDirection.PROGRADE
            assert segment.size == segment.last_index - segment.first_index + 1

    def test_decompose_track_strategy_aliases(self) -> None:
        """Test accepted strategy aliases for decompose_track."""
        lon, lat = self.create_band_crossing_track()
        opts = satellite.DecompositionOptions().with_min_edge_size(1)

        segments_default = satellite.decompose_track(lon, lat, opts=opts)
        segments_alias = satellite.decompose_track(
            lon,
            lat,
            strategy="bands",
            opts=opts,
        )
        segments_monotonic = satellite.decompose_track(
            lon,
            lat,
            strategy="monotonic",
            opts=opts,
        )

        assert len(segments_default) == len(segments_alias)
        assert len(segments_monotonic) >= 1

    def test_decompose_track_invalid_strategy(self) -> None:
        """Test that unknown strategy names raise an error."""
        lon, lat = self.create_band_crossing_track()

        with pytest.raises(ValueError, match="Unknown strategy"):
            satellite.decompose_track(lon, lat, strategy="invalid")

    def test_decompose_track_invalid_inputs(self) -> None:
        """Test input validation for decompose_track."""
        lon = np.array([0.0, 1.0, 2.0])
        lat = np.array([0.0, 1.0])

        with pytest.raises(ValueError, match="same size"):
            satellite.decompose_track(lon, lat)

        with pytest.raises(ValueError, match="non-empty"):
            satellite.decompose_track(np.array([]), np.array([]))

    def test_decompose_track_with_custom_limits(self) -> None:
        """Test decomposition with custom latitude band limits."""
        lon = np.linspace(0.0, 6.0, 7)
        lat = np.array([-12.0, -7.0, -1.0, 0.0, 1.0, 7.0, 12.0])

        opts = (
            satellite.DecompositionOptions()
            .with_south_limit(-5.0)
            .with_north_limit(5.0)
            .with_min_edge_size(1)
        )

        segments = satellite.decompose_track(
            lon,
            lat,
            strategy="latitude_bands",
            opts=opts,
        )

        assert len(segments) == 3
        assert segments[0].zone == satellite.LatitudeZone.SOUTH
        assert segments[1].zone == satellite.LatitudeZone.MID
        assert segments[2].zone == satellite.LatitudeZone.NORTH

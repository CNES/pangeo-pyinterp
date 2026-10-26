# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Satellite track processing: crossovers, decomposition and swaths."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyinterp.core.geometry import geographic, satellite

from ._runner import Case, Context


if TYPE_CHECKING:
    from collections.abc import Callable

#: Inclination of the synthetic orbit, in degrees.
INCLINATION = 77.6


def _ground_track(
    phase: np.ndarray, lon0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Ground track of a circular orbit over a rotating Earth.

    ``phase`` is the argument of latitude, in radians. The longitudes are
    wrapped to [-180, 180[, as in real products.
    """
    inclination = np.radians(INCLINATION)
    lat = np.degrees(np.arcsin(np.sin(inclination) * np.sin(phase)))
    lon = np.degrees(
        np.unwrap(
            np.arctan2(np.cos(inclination) * np.sin(phase), np.cos(phase))
        )
    )
    # Earth rotation under the orbit: about 25 degrees per revolution.
    lon = lon - 25.0 * phase / (2 * np.pi) + lon0
    return (lon + 180) % 360 - 180, lat


def _revolutions(n: int, count: float) -> tuple[np.ndarray, np.ndarray]:
    """``count`` revolutions sampled with ``n`` points."""
    return _ground_track(np.linspace(0, 2 * np.pi * count, n))


def _crossovers(
    n: int, *, use_cartesian: bool
) -> Callable[[], Callable[[], object]]:
    """Time the search of the crossover of an ascending/descending pair."""

    def setup() -> Callable[[], object]:
        # Both passes span about [-100, 100] degrees of longitude, without
        # crossing the antimeridian.
        lon1, lat1 = _ground_track(np.linspace(-np.pi / 2, np.pi / 2, n))
        lon2, lat2 = _ground_track(
            np.linspace(np.pi / 2, 3 * np.pi / 2, n), lon0=-180.0
        )
        return lambda: satellite.find_crossovers(
            lon1,
            lat1,
            lon2,
            lat2,
            predicate=50_000.0,
            use_cartesian=use_cartesian,
        )

    return setup


def _decompose(n: int, strategy: str) -> Callable[[], Callable[[], object]]:
    def setup() -> Callable[[], object]:
        lon, lat = _revolutions(n, 14.0)
        return lambda: satellite.decompose_track(lon, lat, strategy)

    return setup


def _swath(n: int) -> Callable[[], Callable[[], object]]:
    """Time the computation of two 60 km half swaths along a track."""

    def setup() -> Callable[[], object]:
        lon, lat = _revolutions(n, 0.5)
        spheroid = geographic.Spheroid()
        return lambda: satellite.calculate_swath(
            lon, lat, 2_000.0, 10_000.0, 30, spheroid
        )

    return setup


def cases(ctx: Context) -> list[Case]:
    """Build the cases of the family."""
    # About one point per second over a day: 14 revolutions.
    n_track = ctx.size(86_400, 8_640)
    n_pass = ctx.size(20_000, 2_000)
    n_swath = ctx.size(50_000, 5_000)
    return [
        Case(
            "find_crossovers/cartesian",
            _crossovers(n_pass, use_cartesian=True),
            n_pass,
        ),
        Case(
            "find_crossovers/geodetic",
            _crossovers(n_pass, use_cartesian=False),
            n_pass,
        ),
        *(
            Case(
                f"decompose_track/{strategy}",
                _decompose(n_track, strategy),
                n_track,
            )
            for strategy in ("latitude_bands", "monotonic")
        ),
        Case("calculate_swath", _swath(n_swath), n_swath * 60),
    ]

# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""The point-wise predicates reject points with the envelope of the container.

The vectorized functions must give the results of the scalar predicates,
which test every point against the container directly, on the polygons where
an envelope is the most likely to be wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from .......core.geometry.geographic import (
    Box,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    Ring,
)
from .......core.geometry.geographic.algorithms import (
    covered_by,
    for_each_point_covered_by,
    for_each_point_within,
    within,
)


def _ring(lon: np.ndarray, lat: np.ndarray) -> Ring:
    """Build a closed ring from its vertices."""
    return Ring(np.append(lon, lon[0]), np.append(lat, lat[0]))


def _circle(
    lon0: float, lat0: float, radius: float, *, clockwise: bool = True
) -> Ring:
    """Build a ring approximating a circle; holes are counter-clockwise."""
    theta = np.linspace(2 * np.pi, 0, 72, endpoint=False)
    if not clockwise:
        theta = theta[::-1]
    return _ring(lon0 + radius * np.cos(theta), lat0 + radius * np.sin(theta))


def _containers() -> dict[str, object]:
    """Build containers with features an envelope may get wrong."""
    # Long edges along parallels: the geodesics bulge towards the pole,
    # beyond the latitudes of the vertices.
    wide = _ring(
        np.array([-60.0, -60.0, 60.0, 60.0]),
        np.array([40.0, 60.0, 60.0, 40.0]),
    )
    # Crossing the antimeridian
    antimeridian = _ring(
        np.array([170.0, 170.0, -170.0, -170.0]),
        np.array([-10.0, 10.0, 10.0, -10.0]),
    )
    # Around the north pole: the boundary never reaches the pole.
    lon = np.linspace(180, -180, 36, endpoint=False)
    polar = _ring(lon, np.full_like(lon, 75.0))
    with_hole = Polygon(
        _circle(0, 0, 30), [_circle(0, 0, 10, clockwise=False)]
    )
    return {
        "wide": Polygon(wide),
        "antimeridian": Polygon(antimeridian),
        "polar": Polygon(polar),
        "ring": _circle(20, -20, 15),
        "with_hole": with_hole,
        "multipolygon": MultiPolygon(
            [Polygon(_circle(-100, 0, 10)), Polygon(antimeridian)]
        ),
        "box": Box((-30.0, -30.0), (30.0, 30.0)),
    }


@pytest.mark.parametrize("name", list(_containers()))
@pytest.mark.parametrize(
    ("vectorized", "scalar"),
    [(for_each_point_within, within), (for_each_point_covered_by, covered_by)],
)
def test_matches_scalar_predicate(
    name: str, vectorized: object, scalar: object
) -> None:
    """Vectorized and scalar predicates agree everywhere on the sphere."""
    container = _containers()[name]
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, 4000)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 4000)))
    # Points near the poles and on both sides of the antimeridian
    lon = np.concatenate([lon, rng.uniform(-180, 180, 200), [179.9, -179.9]])
    lat = np.concatenate([lat, rng.uniform(85, 90, 200), [0.0, 0.0]])

    result = vectorized(MultiPoint(lon, lat), container)  # type: ignore[operator]
    expected = np.array(
        [
            scalar(Point(x, y), container)  # type: ignore[operator]
            for x, y in zip(lon, lat, strict=True)
        ]
    )
    np.testing.assert_array_equal(result, expected)
    # The cases must not be trivially empty
    assert expected.any()

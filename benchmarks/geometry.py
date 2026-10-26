# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Geographic and Cartesian geometry algorithms.

Only the algorithms whose cost grows with the input are timed: the
vectorized point-wise predicates and distances, and the algorithms walking
large line strings or polygons. Predicates on two small geometries measure
the binding overhead rather than the algorithm, and are left out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyinterp.core.geometry import cartesian, geographic
from pyinterp.core.geometry.geographic import algorithms

from ._runner import Case, Context


if TYPE_CHECKING:
    from collections.abc import Callable

    from pyinterp.type_hints import NDArray1DFloat64


def _circle(
    n: int, radius: float, lon0: float = 0.0
) -> tuple[NDArray1DFloat64, NDArray1DFloat64]:
    """Vertices of a polygon approximating a circle, with a jagged edge."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rho = radius * (1 + 0.05 * np.sin(17 * theta))
    lon = lon0 + rho * np.cos(theta)
    lat = rho * np.sin(theta)
    # Clockwise and closed, as expected by the polygon model.
    return (
        np.append(lon[::-1], lon[-1]).astype(np.float64),
        np.append(lat[::-1], lat[-1]).astype(np.float64),
    )


def _track(n: int) -> tuple[NDArray1DFloat64, NDArray1DFloat64]:
    """Vertices of a long sinuous line, like a satellite ground track."""
    lon = np.linspace(-170, 170, n)
    return lon, 60 * np.sin(np.radians(lon) * 3)


def _points(
    rng: np.random.Generator, n: int
) -> tuple[NDArray1DFloat64, NDArray1DFloat64]:
    """Random points around the polygons built from :func:`_circle`."""
    return rng.uniform(-30, 30, n), rng.uniform(-30, 30, n)


def _geographic_polygon(
    n: int, radius: float, lon0: float = 0.0
) -> geographic.Polygon:
    return geographic.Polygon(geographic.Ring(*_circle(n, radius, lon0)))


def _cartesian_polygon(
    n: int, radius: float, lon0: float = 0.0
) -> cartesian.Polygon:
    return cartesian.Polygon(cartesian.Ring(*_circle(n, radius, lon0)))


def _geographic_point_cases(ctx: Context) -> list[Case]:
    """Vectorized predicates and distances, for every distance strategy."""
    n_points = ctx.size(200_000, 20_000)
    n_vertices = ctx.size(100_000, 10_000)
    # The point-wise predicates and distances test every point against every
    # edge of the polygon, on the ellipsoid: far fewer points are needed.
    n_within = ctx.size(20_000, 2_000)
    n_distance = ctx.size(5_000, 500)
    result: list[Case] = []

    def point_wise(
        func: Callable[..., object], n: int, vertices: int, **kwargs: object
    ) -> Callable[[], Callable[[], object]]:
        def setup() -> Callable[[], object]:
            points = geographic.MultiPoint(*_points(ctx.rng(), n))
            polygon = _geographic_polygon(vertices, 20)
            return lambda: func(points, polygon, **kwargs)

        return setup

    result += [
        Case(
            "geographic/for_each_point_within",
            point_wise(algorithms.for_each_point_within, n_within, 500),
            n_within,
        ),
        Case(
            "geographic/for_each_point_covered_by",
            point_wise(algorithms.for_each_point_covered_by, n_within, 500),
            n_within,
        ),
    ]
    for strategy in algorithms.Strategy:
        label = strategy.name.lower()
        # Karney's series are about 40 times more expensive than the others.
        cost = 40 if strategy == algorithms.Strategy.KARNEY else 1
        n_strategy = n_distance // cost
        n_pairs = n_points // cost

        def pairwise(
            strategy: algorithms.Strategy = strategy,
            n_pairs: int = n_pairs,
        ) -> Callable[[], object]:
            rng = ctx.rng()
            first = geographic.MultiPoint(*_points(rng, n_pairs))
            second = geographic.MultiPoint(*_points(rng, n_pairs))
            return lambda: algorithms.for_each_point_pairwise_distance(
                first, second, strategy=strategy
            )

        def along_track(
            strategy: algorithms.Strategy = strategy,
        ) -> Callable[[], object]:
            line = geographic.LineString(*_track(n_vertices))
            return lambda: algorithms.curvilinear_distance(
                line, strategy=strategy
            )

        result += [
            Case(
                f"geographic/for_each_point_distance/{label}",
                point_wise(
                    algorithms.for_each_point_distance,
                    n_strategy,
                    200,
                    strategy=strategy,
                ),
                n_strategy,
            ),
            Case(
                f"geographic/for_each_point_pairwise_distance/{label}",
                pairwise,
                n_pairs,
            ),
            Case(
                f"geographic/curvilinear_distance/{label}",
                along_track,
                n_vertices,
            ),
        ]
    return result


def _geographic_shape_cases(ctx: Context) -> list[Case]:
    """Algorithms walking large line strings and polygons."""
    n_vertices = ctx.size(100_000, 10_000)
    n_polygon = ctx.size(2_000, 500)
    n_simplify = ctx.size(20_000, 2_000)

    def on_track(
        func: Callable[..., object], *args: object, n: int = n_vertices
    ) -> Callable[[], Callable[[], object]]:
        def setup() -> Callable[[], object]:
            line = geographic.LineString(*_track(n))
            return lambda: func(line, *args)

        return setup

    def on_polygon(
        func: Callable[..., object], *args: object
    ) -> Callable[[], Callable[[], object]]:
        def setup() -> Callable[[], object]:
            polygon = _geographic_polygon(n_vertices, 20)
            return lambda: func(polygon, *args)

        return setup

    def overlay(
        func: Callable[..., object],
    ) -> Callable[[], Callable[[], object]]:
        def setup() -> Callable[[], object]:
            first = _geographic_polygon(n_polygon * 5, 20)
            second = _geographic_polygon(n_polygon * 5, 20, lon0=15)
            return lambda: func(first, second)

        return setup

    return [
        Case("geographic/length", on_track(algorithms.length), n_vertices),
        Case(
            "geographic/simplify",
            on_track(algorithms.simplify, 1000.0, n=n_simplify),
            n_simplify,
        ),
        Case(
            "geographic/densify",
            on_track(algorithms.densify, 50_000.0),
            n_vertices,
        ),
        Case("geographic/is_valid", on_track(algorithms.is_valid), n_vertices),
        Case("geographic/area", on_polygon(algorithms.area), n_vertices),
        Case(
            "geographic/perimeter",
            on_polygon(algorithms.perimeter),
            n_vertices,
        ),
        Case(
            "geographic/convex_hull",
            on_polygon(algorithms.convex_hull),
            n_vertices,
        ),
        Case(
            "geographic/intersection",
            overlay(algorithms.intersection),
            n_polygon * 10,
        ),
        Case("geographic/union", overlay(algorithms.union), n_polygon * 10),
        Case(
            "geographic/difference",
            overlay(algorithms.difference),
            n_polygon * 10,
        ),
    ]


def _coordinates_cases(ctx: Context) -> list[Case]:
    n = ctx.size(2_000_000, 200_000)

    def transform(method: str) -> Callable[[], Callable[[], object]]:
        def setup() -> Callable[[], object]:
            rng = ctx.rng()
            wgs84 = geographic.Coordinates()
            lon = rng.uniform(-180, 180, n)
            lat = rng.uniform(-90, 90, n)
            alt = rng.uniform(0, 10_000, n)
            if method == "lla_to_ecef":
                return lambda: wgs84.lla_to_ecef(lon, lat, alt, ctx.threads)
            if method == "ecef_to_lla":
                xyz = wgs84.lla_to_ecef(lon, lat, alt)
                return lambda: wgs84.ecef_to_lla(*xyz, ctx.threads)
            # GRS80 differs slightly from WGS84: a datum shift.
            grs80 = geographic.Coordinates(
                geographic.Spheroid(6378137.0, 1 / 298.257222101)
            )
            return lambda: wgs84.transform(grs80, lon, lat, alt, ctx.threads)

        return setup

    return [
        Case(f"coordinates/{method}", transform(method), n)
        for method in ("lla_to_ecef", "ecef_to_lla", "transform")
    ]


def _cartesian_cases(ctx: Context) -> list[Case]:
    algo = cartesian.algorithms
    n_points = ctx.size(200_000, 20_000)
    n_vertices = ctx.size(100_000, 10_000)
    n_polygon = ctx.size(2_000, 500)

    def point_wise(
        func: Callable[..., object],
    ) -> Callable[[], Callable[[], object]]:
        def setup() -> Callable[[], object]:
            points = cartesian.MultiPoint(*_points(ctx.rng(), n_points))
            polygon = _cartesian_polygon(500, 20)
            return lambda: func(points, polygon)

        return setup

    def overlay(
        func: Callable[..., object],
    ) -> Callable[[], Callable[[], object]]:
        def setup() -> Callable[[], object]:
            first = _cartesian_polygon(n_polygon * 5, 20)
            second = _cartesian_polygon(n_polygon * 5, 20, lon0=15)
            return lambda: func(first, second)

        return setup

    def buffer() -> Callable[[], object]:
        line = cartesian.LineString(*_track(n_vertices // 10))
        strategies = (
            algo.DistanceSymmetric(0.5),
            algo.JoinRound(16),
            algo.EndRound(16),
            algo.PointCircle(16),
        )
        return lambda: algo.buffer(line, *strategies)

    def simplify() -> Callable[[], object]:
        line = cartesian.LineString(*_track(n_vertices))
        return lambda: algo.simplify(line, 0.01)

    def convex_hull() -> Callable[[], object]:
        points = cartesian.MultiPoint(*_points(ctx.rng(), n_points))
        return lambda: algo.convex_hull(points)

    return [
        Case(
            "cartesian/for_each_point_within",
            point_wise(algo.for_each_point_within),
            n_points,
        ),
        Case(
            "cartesian/for_each_point_distance",
            point_wise(algo.for_each_point_distance),
            n_points,
        ),
        Case("cartesian/convex_hull", convex_hull, n_points),
        Case("cartesian/simplify", simplify, n_vertices),
        Case("cartesian/buffer", buffer, n_vertices // 10),
        Case(
            "cartesian/intersection",
            overlay(algo.intersection),
            n_polygon * 10,
        ),
        Case("cartesian/union", overlay(algo.union), n_polygon * 10),
    ]


def cases(ctx: Context) -> list[Case]:
    """Build the cases of the family."""
    return [
        *_coordinates_cases(ctx),
        *_geographic_point_cases(ctx),
        *_geographic_shape_cases(ctx),
        *_cartesian_cases(ctx),
    ]

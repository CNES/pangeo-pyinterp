# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Unstructured interpolation through R*-trees.

Covers the construction of the trees (bulk loading and insertion), the k-NN
query, and every estimator built on top of it: inverse distance weighting,
radial basis functions, window functions, kriging and the 4D optimal
interpolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyinterp import core
from pyinterp.core.config import rtree as config
from pyinterp.core.geometry import geographic

from ._runner import Case, Context


if TYPE_CHECKING:
    from collections.abc import Callable

#: Neighbors used by the estimators.
K = 16


def _scatter(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Random observations of a smooth field on the sphere."""
    coordinates = np.column_stack(
        (rng.uniform(-180, 180, n), rng.uniform(-80, 80, n))
    )
    lon, lat = np.radians(coordinates.T)
    return coordinates, np.sin(2 * lon) * np.cos(lat)


def _tree3d(
    rng: np.random.Generator,
    n: int,
    dtype: str = "float64",
    distance: core.RTree3DDistance = "euclidean",
) -> tuple[core.RTree3DFloat32 | core.RTree3DFloat64, np.ndarray]:
    """Bulk-loaded tree and the coordinates of its observations.

    The observations are longitudes and latitudes: the tree converts them to
    ECEF with the WGS84 spheroid.
    """
    coordinates, values = _scatter(rng, n)
    tree = core.RTree3D(geographic.Spheroid(), dtype=dtype, distance=distance)
    tree.packing(coordinates, values)
    return tree, coordinates


def _build(
    ctx: Context, method: str, n: int
) -> Callable[[], Callable[[], object]]:
    """Time the construction of the tree."""

    def setup() -> Callable[[], object]:
        coordinates, values = _scatter(ctx.rng(), n)
        tree = core.RTree3D(geographic.Spheroid())
        return lambda: getattr(tree, method)(coordinates, values)

    return setup


def _estimator(
    ctx: Context,
    method: str,
    settings: object,
    n_obs: int,
    n_query: int,
    *,
    dtype: str = "float64",
    distance: core.RTree3DDistance = "euclidean",
) -> Callable[[], Callable[[], object]]:
    """Time an estimator of the 3D tree."""

    def setup() -> Callable[[], object]:
        rng = ctx.rng()
        tree, _ = _tree3d(rng, n_obs, dtype, distance)
        query, _ = _scatter(rng, n_query)
        return lambda: getattr(tree, method)(query, settings)

    return setup


def _tree4d(
    ctx: Context, method: str, n_obs: int, n_query: int
) -> Callable[[], Callable[[], object]]:
    """Time the 4D tree used by the optimal interpolation."""

    def setup() -> Callable[[], object]:
        rng = ctx.rng()

        def sample(n: int) -> np.ndarray:
            # Cartesian (x, y, z, t): a 1000 km square observed for 30 days.
            return rng.uniform([0, 0, 0, 0], [1000, 1000, 0, 30], (n, 4))

        obs = sample(n_obs)
        values = rng.standard_normal(n_obs)
        sigma2 = rng.uniform(0.01, 0.02, n_obs)
        tree = core.RTree4D()
        if method == "packing":
            return lambda: tree.packing(obs, values, sigma2)
        tree.packing(obs, values, sigma2)
        query = sample(n_query)
        if method == "query":
            knn = config.Query().with_k(K).with_num_threads(ctx.threads)
            return lambda: tree.query(query, knn)
        lx, ly, lz, lt = (
            np.full(n_query, value) for value in (50.0, 50.0, 1.0, 5.0)
        )
        sigma = np.ones(n_query)
        oi = (
            config.OptimalInterpolation()
            .with_k(K)
            .with_num_threads(ctx.threads)
        )
        return lambda: tree.optimal_interpolation(
            query, lx, ly, lz, lt, sigma, oi
        )

    return setup


def _geographic(
    ctx: Context, method: str, n_obs: int, n_query: int
) -> Callable[[], Callable[[], object]]:
    """Time the geographic tree."""

    def setup() -> Callable[[], object]:
        rng = ctx.rng()
        coordinates, values = _scatter(rng, n_obs)
        tree = geographic.RTree()
        if method == "packing":
            return lambda: tree.packing(coordinates, values)
        tree.packing(coordinates, values)
        query, _ = _scatter(rng, n_query)
        if method == "query":
            knn = config.Query().with_k(K).with_num_threads(ctx.threads)
            return lambda: tree.query(query, knn)
        idw = (
            config.InverseDistanceWeighting()
            .with_k(K)
            .with_num_threads(ctx.threads)
        )
        return lambda: tree.inverse_distance_weighting(query, idw)

    return setup


def cases(ctx: Context) -> list[Case]:
    """Build the cases of the family."""
    n_obs = ctx.size(1_000_000, 100_000)
    n_query = ctx.size(200_000, 20_000)
    # Kriging and RBF solve a k x k system per point.
    n_solve = ctx.size(50_000, 5_000)
    threads = ctx.threads

    estimators: list[tuple[str, str, object, int]] = [
        (
            "query",
            "query",
            config.Query().with_k(K).with_num_threads(threads),
            n_query,
        ),
        (
            "idw",
            "inverse_distance_weighting",
            config.InverseDistanceWeighting()
            .with_k(K)
            .with_num_threads(threads),
            n_query,
        ),
        *(
            (
                f"rbf/{kernel.name.lower()}",
                "radial_basis_function",
                config.RadialBasisFunction()
                .with_k(K)
                .with_rbf(kernel)
                .with_num_threads(threads),
                n_solve,
            )
            for kernel in config.RBFKernel
        ),
        *(
            (
                f"window/{kernel.name.lower()}",
                "window_function",
                config.InterpolationWindow()
                .with_k(K)
                .with_wf(kernel)
                .with_num_threads(threads),
                n_query,
            )
            for kernel in config.WindowKernel
        ),
        *(
            (
                f"kriging/{model.name.lower()}",
                "kriging",
                config.Kriging()
                .with_k(K)
                .with_covariance_model(model)
                .with_num_threads(threads),
                n_solve,
            )
            for model in config.CovarianceFunction
        ),
    ]

    result = [
        Case("rtree3d/packing", _build(ctx, "packing", n_obs), n_obs),
        Case(
            "rtree3d/insert",
            _build(ctx, "insert", n_obs // 10),
            n_obs // 10,
            fresh=True,
        ),
        *(
            Case(
                f"rtree3d/{name}",
                _estimator(ctx, method, settings, n_obs, n),
                n,
            )
            for name, method, settings, n in estimators
        ),
        Case(
            "rtree3d/idw/float32",
            _estimator(
                ctx,
                "inverse_distance_weighting",
                estimators[1][2],
                n_obs,
                n_query,
                dtype="float32",
            ),
            n_query,
        ),
        *(
            Case(
                f"rtree3d/{name}/geodesic",
                _estimator(
                    ctx, method, settings, n_obs, n, distance="geodesic"
                ),
                n,
            )
            for name, method, settings, n in estimators[:2]
        ),
        Case("rtree4d/packing", _tree4d(ctx, "packing", n_obs, 0), n_obs),
        Case("rtree4d/query", _tree4d(ctx, "query", n_obs, n_query), n_query),
        Case(
            "rtree4d/optimal_interpolation",
            _tree4d(ctx, "optimal_interpolation", n_obs, n_solve),
            n_solve,
        ),
    ]
    # The geographic tree computes geodesic distances during the search.
    n_geodesic = n_query // 10
    result += [
        Case(
            "geographic/packing",
            _geographic(ctx, "packing", n_obs, 0),
            n_obs,
        ),
        *(
            Case(
                f"geographic/{method}",
                _geographic(ctx, method, n_obs, n_geodesic),
                n_geodesic,
            )
            for method in ("query", "idw")
        ),
    ]
    return result

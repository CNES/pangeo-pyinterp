# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Binning, histograms and streaming statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyinterp import core
from pyinterp.core.geometry import geographic

from ._runner import Case, Context


if TYPE_CHECKING:
    from collections.abc import Callable


def _samples(
    ctx: Context, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random longitudes, latitudes and values."""
    rng = ctx.rng()
    return (
        rng.uniform(-180, 180, n),
        rng.uniform(-90, 90, n),
        rng.standard_normal(n),
    )


def _axes() -> tuple[core.Axis, core.Axis]:
    """One degree global grid."""
    return (
        core.Axis(np.arange(-180.0, 180.0, 1.0), period=360.0),
        core.Axis(np.arange(-90.0, 90.5, 1.0)),
    )


def _binning2d(
    ctx: Context, n: int, *, simple: bool, geodetic: bool
) -> Callable[[], Callable[[], object]]:
    """Time the accumulation of samples in a 2D binning."""

    def setup() -> Callable[[], object]:
        x, y, z = _samples(ctx, n)
        spheroid = geographic.Spheroid() if geodetic else None
        binning = core.Binning2D(*_axes(), spheroid)
        return lambda: binning.push(x, y, z, simple)

    return setup


def _binning1d(ctx: Context, n: int) -> Callable[[], Callable[[], object]]:
    """Time the accumulation of samples in a 1D binning."""

    def setup() -> Callable[[], object]:
        x, _, z = _samples(ctx, n)
        binning = core.Binning1D(_axes()[0])
        return lambda: binning.push(x, z)

    return setup


def _histogram2d(
    ctx: Context, n: int, method: str
) -> Callable[[], Callable[[], object]]:
    """Time the 2D histogram: filling it, or computing a quantile."""

    def setup() -> Callable[[], object]:
        x, y, z = _samples(ctx, n)
        histogram = core.Histogram2D(*_axes())
        if method == "push":
            return lambda: histogram.push(x, y, z)
        histogram.push(x, y, z)
        return lambda: histogram.quantile(0.5)

    return setup


def _reduction(
    ctx: Context,
    factory: Callable[..., object],
    shape: tuple[int, ...],
    *,
    axis: list[int] | None = None,
    weighted: bool = False,
) -> Callable[[], Callable[[], object]]:
    """Time the construction of a statistics holder from an array."""

    def setup() -> Callable[[], object]:
        rng = ctx.rng()
        values = rng.standard_normal(shape)
        weights = rng.uniform(0.5, 1.5, shape) if weighted else None
        return lambda: factory(values, weights, axis)

    return setup


def _tdigest_quantile(
    ctx: Context, shape: tuple[int, ...]
) -> Callable[[], Callable[[], object]]:
    """Time the quantile evaluation of a t-digest reduced along an axis."""

    def setup() -> Callable[[], object]:
        digest = core.TDigest(ctx.rng().standard_normal(shape), axis=[0])
        quantiles = np.linspace(0.01, 0.99, 99)
        return lambda: digest.quantile(quantiles)

    return setup


def cases(ctx: Context) -> list[Case]:
    """Build the cases of the family."""
    n = ctx.size(2_000_000, 200_000)
    # The geodetic binning computes the area of every cell it touches.
    n_geodetic = n // 10
    # The array reduced along its first axis: a time series per grid cell.
    shape = ctx.size((100, 360, 180), (20, 90, 45))
    # A t-digest over the whole array merges far more centroids than one
    # per cell, at a much lower throughput.
    shape_digest = ctx.size((20, 90, 45), (5, 45, 45))
    result = [
        Case("binning1d/push", _binning1d(ctx, n), n, fresh=True),
        Case(
            "binning2d/push/simple",
            _binning2d(ctx, n, simple=True, geodetic=False),
            n,
            fresh=True,
        ),
        Case(
            "binning2d/push/linear",
            _binning2d(ctx, n, simple=False, geodetic=False),
            n,
            fresh=True,
        ),
        Case(
            "binning2d/push/linear/geodetic",
            _binning2d(ctx, n_geodetic, simple=False, geodetic=True),
            n_geodetic,
            fresh=True,
        ),
        Case(
            "histogram2d/push",
            _histogram2d(ctx, n, "push"),
            n,
            fresh=True,
        ),
        Case("histogram2d/quantile", _histogram2d(ctx, n, "quantile"), n),
    ]
    for name, factory, whole in (
        ("descriptive_statistics", core.DescriptiveStatistics, shape),
        ("tdigest", core.TDigest, shape_digest),
    ):
        result += [
            Case(
                f"{name}/all",
                _reduction(ctx, factory, whole),
                int(np.prod(whole)),
            ),
            Case(
                f"{name}/all/weighted",
                _reduction(ctx, factory, whole, weighted=True),
                int(np.prod(whole)),
            ),
            Case(
                f"{name}/axis0",
                _reduction(ctx, factory, shape, axis=[0]),
                int(np.prod(shape)),
            ),
        ]
    result.append(
        Case(
            "tdigest/axis0/quantile",
            _tdigest_quantile(ctx, shape),
            int(np.prod(shape)),
        )
    )
    return result

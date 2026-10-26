# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Interpolation on regular grids, from one to four dimensions.

Every method of the geometric (bilinear, nearest, IDW on the cell corners)
and windowed (splines fitted on a moving window) families is timed on a
longitude/latitude grid whose first axis is periodic, the common case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pyinterp import core
from pyinterp.core.config import geometric, windowed

from ._runner import Case, Context


if TYPE_CHECKING:
    from collections.abc import Callable

GEOMETRIC = ("bilinear", "nearest", "idw")

#: Methods of the windowed univariate interpolator.
WINDOWED_1D = (
    "linear",
    "polynomial",
    "c_spline",
    "c_spline_not_a_knot",
    "c_spline_periodic",
    "akima",
    "akima_periodic",
    "steffen",
)

#: The multivariate interpolators add the tensor-product methods.
WINDOWED_ND = ("bilinear", "bicubic", *WINDOWED_1D)


def _axes(*shape: int) -> list[core.Axis]:
    """Longitude, latitude, then generic axes of the given lengths."""
    lon = core.Axis(np.linspace(0, 360, shape[0], endpoint=False), period=360)
    lat = core.Axis(np.linspace(-90, 90, shape[1]))
    others = [core.Axis(np.linspace(0, 100, n)) for n in shape[2:]]
    return [lon, lat, *others]


def _field(rng: np.random.Generator, *shape: int) -> np.ndarray:
    """Smooth field plus noise, so that no method sees a trivial input."""
    lon = np.radians(np.linspace(0, 360, shape[0], endpoint=False))
    lat = np.radians(np.linspace(-90, 90, shape[1]))
    base = np.sin(2 * lon)[:, None] * np.cos(lat)[None, :]
    base = base.reshape(base.shape + (1,) * (len(shape) - 2))
    return base + 0.01 * rng.standard_normal(shape)


def _coordinates(
    rng: np.random.Generator, n: int, ndim: int
) -> list[np.ndarray]:
    """Random points covering the whole grid."""
    return [
        rng.uniform(0, 360, n),
        rng.uniform(-90, 90, n),
        *(rng.uniform(0, 100, n) for _ in range(ndim - 2)),
    ]


def _univariate(
    ctx: Context, method: str, n: int, derivative: bool
) -> Callable[[], Callable[[], object]]:
    def setup() -> Callable[[], object]:
        rng = ctx.rng()
        x = np.linspace(0, 360, 36_000, endpoint=False)
        grid = core.Grid(
            core.Axis(x, period=360),
            np.sin(np.radians(x)) + 0.01 * rng.standard_normal(x.size),
        )
        xq = rng.uniform(0, 360, n)
        config = getattr(windowed.Univariate, method)().with_num_threads(
            ctx.threads
        )
        func = core.univariate_derivative if derivative else core.univariate
        return lambda: func(grid, xq, config)

    return setup


def _grid(axes: list[core.Axis], array: np.ndarray) -> core.GridHolder[Any]:
    """Grid of two to four dimensions.

    The overloads of :func:`core.Grid` need a fixed number of axes.
    """
    if len(axes) == 2:  # noqa: PLR2004
        return core.Grid(axes[0], axes[1], array)
    if len(axes) == 3:  # noqa: PLR2004
        return core.Grid(axes[0], axes[1], axes[2], array)
    return core.Grid(axes[0], axes[1], axes[2], axes[3], array)


#: Interpolation function and configuration namespace per dimension.
_ND: dict[int, tuple[Callable[..., object], str]] = {
    2: (core.bivariate, "Bivariate"),
    3: (core.trivariate, "Trivariate"),
    4: (core.quadrivariate, "Quadrivariate"),
}


def _multivariate(
    ctx: Context,
    family: str,
    method: str,
    shape: tuple[int, ...],
    n: int,
    *,
    dtype: type[np.floating[Any]] = np.float64,
) -> Callable[[], Callable[[], object]]:
    def setup() -> Callable[[], object]:
        rng = ctx.rng()
        grid = _grid(_axes(*shape), _field(rng, *shape).astype(dtype))
        points = _coordinates(rng, n, len(shape))
        func, name = _ND[len(shape)]
        module = geometric if family == "geometric" else windowed
        config = getattr(getattr(module, name), method)().with_num_threads(
            ctx.threads
        )
        return lambda: func(grid, *points, config=config)

    return setup


def _temporal(
    ctx: Context, method: str, shape: tuple[int, int, int], n: int
) -> Callable[[], Callable[[], object]]:
    """Trivariate interpolation with a datetime64 third axis."""

    def setup() -> Callable[[], object]:
        rng = ctx.rng()
        lon, lat = _axes(*shape[:2])
        start = np.datetime64("2020-01-01T00:00:00", "us")
        time = core.TemporalAxis(
            start + np.arange(shape[2]) * np.timedelta64(1, "h")
        )
        grid = core.Grid(lon, lat, time, _field(rng, *shape))
        x, y = _coordinates(rng, n, 2)
        z = start + rng.integers(0, (shape[2] - 1) * 3600, n).astype(
            "timedelta64[s]"
        )
        config = getattr(geometric.Trivariate, method)().with_num_threads(
            ctx.threads
        )
        return lambda: core.trivariate(grid, x, y, z, config=config)

    return setup


def cases(ctx: Context) -> list[Case]:
    """Build the cases of the family."""
    # The windowed methods fit a spline per point: fewer points keep each
    # case under a second.
    n_geo = ctx.size(1_000_000, 100_000)
    n_win = ctx.size(200_000, 20_000)
    shapes = {
        2: ctx.size((720, 360), (180, 90)),
        3: ctx.size((360, 180, 50), (90, 45, 10)),
        4: ctx.size((180, 90, 20, 10), (45, 23, 5, 5)),
    }
    result: list[Case] = []
    for method in WINDOWED_1D:
        result += [
            Case(
                f"univariate/{method}",
                _univariate(ctx, method, n_win, derivative=False),
                n_win,
            ),
            Case(
                f"univariate_derivative/{method}",
                _univariate(ctx, method, n_win, derivative=True),
                n_win,
            ),
        ]
    for ndim, (_, name) in _ND.items():
        prefix = name.lower()
        shape = shapes[ndim]
        result += [
            Case(
                f"{prefix}/geometric/{method}",
                _multivariate(ctx, "geometric", method, shape, n_geo),
                n_geo,
            )
            for method in GEOMETRIC
        ]
        result += [
            Case(
                f"{prefix}/windowed/{method}",
                _multivariate(ctx, "windowed", method, shape, n_win),
                n_win,
            )
            for method in WINDOWED_ND
        ]
    # Single precision grids take a different template instantiation.
    result += [
        Case(
            "bivariate/geometric/bilinear/float32",
            _multivariate(
                ctx,
                "geometric",
                "bilinear",
                shapes[2],
                n_geo,
                dtype=np.float32,
            ),
            n_geo,
        ),
        Case(
            "bivariate/windowed/bicubic/float32",
            _multivariate(
                ctx, "windowed", "bicubic", shapes[2], n_win, dtype=np.float32
            ),
            n_win,
        ),
    ]
    result += [
        Case(
            f"trivariate/temporal/{method}",
            _temporal(ctx, method, (shapes[3][0], shapes[3][1], 240), n_geo),
            n_geo,
        )
        for method in GEOMETRIC
    ]
    return result

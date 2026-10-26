# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Gap filling of the undefined values of a grid.

The iterative solvers run until a fixed tolerance is reached, so a change
that improves their convergence rate shows up as a speed-up as well. The
grid mixes scattered missing cells with a large hole, which is what makes
the solvers iterate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyinterp import core
from pyinterp.core.config import fill as config

from ._runner import Case, Context


if TYPE_CHECKING:
    from collections.abc import Callable

#: Fraction of cells removed at random.
MISSING = 0.1


def _grid(ctx: Context, shape: tuple[int, int]) -> np.ndarray:
    """Smooth field with scattered gaps and a rectangular hole."""
    rng = ctx.rng()
    nx, ny = shape
    lon = np.radians(np.linspace(0, 360, nx, endpoint=False))
    lat = np.radians(np.linspace(-80, 80, ny))
    grid = np.sin(2 * lon)[:, None] * np.cos(lat)[None, :]
    grid[rng.random(shape) < MISSING] = np.nan
    grid[nx // 3 : nx // 3 + nx // 10, ny // 3 : ny // 3 + ny // 10] = np.nan
    return grid


def _solver(
    ctx: Context, name: str, settings: object, shape: tuple[int, int]
) -> Callable[[], Callable[[], object]]:
    """Time a solver; they all fill the grid in place."""

    def setup() -> Callable[[], object]:
        grid = _grid(ctx, shape)
        func = getattr(core.fill, name)
        return lambda: func(grid, settings)

    return setup


def _fill_value(
    ctx: Context, name: str, shape: tuple[int, int]
) -> Callable[[], Callable[[], object]]:
    """Time the linear fillers, which also work in place."""

    def setup() -> Callable[[], object]:
        grid = _grid(ctx, shape)
        if name == "vector":
            grid = grid.ravel()
        func = getattr(core.fill, name)
        return lambda: func(grid, np.nan)

    return setup


def cases(ctx: Context) -> list[Case]:
    """Build the cases of the family."""
    # The iteration count of the relaxation solvers grows with the grid
    # size: beyond this one, a single solve takes tens of seconds.
    shape = ctx.size((720, 360), (360, 180))
    # LOESS fits a weighted regression per cell: a smaller grid keeps it in
    # the same time range as the other solvers.
    loess_shape = ctx.size((240, 120), (90, 45))
    threads = ctx.threads
    cells = shape[0] * shape[1]
    solvers = {
        "gauss_seidel": config.GaussSeidel()
        .with_epsilon(1e-4)
        .with_max_iterations(2000)
        .with_is_periodic(True)
        .with_num_threads(threads),
        "multigrid": config.Multigrid()
        .with_epsilon(1e-4)
        .with_max_iterations(200)
        .with_is_periodic(True)
        .with_num_threads(threads),
        "fft_inpaint": config.FFTInpaint()
        .with_epsilon(1e-4)
        .with_max_iterations(200)
        .with_is_periodic(True)
        .with_num_threads(threads),
    }
    result = [
        Case(name, _solver(ctx, name, settings, shape), cells, fresh=True)
        for name, settings in solvers.items()
    ]
    for value_type in config.LoessValueType:
        settings = (
            config.Loess()
            .with_nx(3)
            .with_ny(3)
            .with_value_type(value_type)
            .with_is_periodic(True)
            .with_num_threads(threads)
        )
        result.append(
            Case(
                f"loess/{value_type.name.lower()}",
                _solver(ctx, "loess", settings, loess_shape),
                loess_shape[0] * loess_shape[1],
            )
        )
    result += [
        Case(name, _fill_value(ctx, name, shape), cells, fresh=True)
        for name in ("matrix", "vector")
    ]
    return result

# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Index lookups on axes, the first step of every grid interpolation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyinterp import core

from ._runner import Case, Context


if TYPE_CHECKING:
    from collections.abc import Callable

#: Five years of hourly dates.
_HOURS = 24 * 365 * 5


def _axis(kind: str, rng: np.random.Generator) -> core.Axis:
    """Build an axis of the kind given, each having its own code path."""
    if kind == "regular":
        return core.Axis(np.arange(-90.0, 90.0, 0.05))
    if kind == "periodic":
        return core.Axis(np.arange(0.0, 360.0, 0.1), period=360.0)
    steps = np.cumsum(rng.uniform(0.5, 1.5, 3600))
    return core.Axis(steps * (180.0 / steps[-1]) - 90.0)


def _lookup(
    ctx: Context, kind: str, method: str, n: int
) -> Callable[[], Callable[[], object]]:
    """Time ``method`` on an axis of the kind given."""

    def setup() -> Callable[[], object]:
        rng = ctx.rng()
        axis = _axis(kind, rng)
        x = rng.uniform(axis.min_value(), axis.max_value(), n)
        if method == "find_index":
            return lambda: axis.find_index(x, bounded=True)
        return lambda: axis.find_indexes(x)

    return setup


def _temporal_lookup(
    ctx: Context, method: str, n: int
) -> Callable[[], Callable[[], object]]:
    """Time ``method`` on an hourly temporal axis."""

    def setup() -> Callable[[], object]:
        rng = ctx.rng()
        start = np.datetime64("2020-01-01", "ns")
        axis = core.TemporalAxis(
            start + np.arange(_HOURS) * np.timedelta64(1, "h")
        )
        offsets = rng.integers(0, _HOURS * 3600, n)
        dates = start + offsets.astype("timedelta64[s]")
        return lambda: getattr(axis, method)(dates)

    return setup


def cases(ctx: Context) -> list[Case]:
    """Build the cases of the family."""
    n = ctx.size(2_000_000, 200_000)
    return [
        *(
            Case(f"{method}/{kind}", _lookup(ctx, kind, method, n), n)
            for kind in ("regular", "periodic", "irregular")
            for method in ("find_index", "find_indexes")
        ),
        *(
            Case(f"{method}/temporal", _temporal_lookup(ctx, method, n), n)
            for method in ("find_index", "find_indexes")
        ),
    ]

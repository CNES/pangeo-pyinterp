# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Calendar decomposition of dates, and operations on period lists."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyinterp.core import dateutils, period

from ._runner import Case, Context


if TYPE_CHECKING:
    from collections.abc import Callable

#: Twenty years, in seconds.
_SPAN = 20 * 365 * 86_400


def _dates(ctx: Context, n: int) -> np.ndarray:
    """Random dates with a microsecond resolution."""
    offsets = ctx.rng().integers(0, _SPAN * 1_000_000, n)
    return np.datetime64("2000-01-01", "us") + offsets.astype(
        "timedelta64[us]"
    )


def _calendar(
    ctx: Context, name: str, n: int
) -> Callable[[], Callable[[], object]]:
    def setup() -> Callable[[], object]:
        dates = _dates(ctx, n)
        func = getattr(dateutils, name)
        return lambda: func(dates)

    return setup


def _periods(ctx: Context, n: int) -> period.PeriodList:
    """Sorted, disjoint periods lasting between one minute and one day."""
    rng = ctx.rng()
    start = np.datetime64("2000-01-01", "s")
    gaps = rng.integers(60, 86_400, n).astype("timedelta64[s]")
    durations = rng.integers(60, 86_400, n).astype("timedelta64[s]")
    begins = start + np.cumsum(gaps + durations) - durations
    return period.PeriodList(
        [
            period.Period(begin, begin + duration)
            for begin, duration in zip(begins, durations, strict=True)
        ]
    )


def _period_list(
    ctx: Context, method: str, n_periods: int, n_dates: int
) -> Callable[[], Callable[[], object]]:
    def setup() -> Callable[[], object]:
        periods = _periods(ctx, n_periods)
        if method == "join_adjacent_periods":
            epsilon = np.timedelta64(3600, "s")
            return lambda: periods.join_adjacent_periods(epsilon)
        if method == "sort":
            # Reverse the list so that sorting has work to do.
            periods = period.PeriodList(list(periods)[::-1])
            return periods.sort
        first, last = periods[0].begin, periods[len(periods) - 1].last
        span = (last - first).astype("int64")
        dates = first + np.sort(ctx.rng().integers(0, span, n_dates)).astype(
            "timedelta64[s]"
        )
        return lambda: getattr(periods, method)(dates)

    return setup


def cases(ctx: Context) -> list[Case]:
    """Build the cases of the family."""
    n = ctx.size(2_000_000, 200_000)
    n_periods = ctx.size(100_000, 10_000)
    result = [
        Case(name, _calendar(ctx, name, n), n)
        for name in (
            "date",
            "time",
            "isocalendar",
            "weekday",
            "timedelta_since_january",
        )
    ]
    result += [
        Case(
            f"period_list/{method}",
            _period_list(ctx, method, n_periods, n),
            n,
        )
        for method in ("belong_to_a_period", "cross_a_period")
    ]
    result += [
        Case(
            f"period_list/{method}",
            _period_list(ctx, method, n_periods, 0),
            n_periods,
            fresh=method == "sort",
        )
        for method in ("join_adjacent_periods", "sort")
    ]
    return result

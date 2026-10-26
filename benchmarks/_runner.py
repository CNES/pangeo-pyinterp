# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Timing harness shared by the benchmark families.

A family module exposes ``cases(ctx: Context) -> list[Case]``. Each case
builds its inputs in ``setup`` (never timed) and returns the zero-argument
callable that is timed. Repetitions continue until both ``min_repeat`` calls
and ``min_time`` seconds have been accumulated, so fast and slow operations
get comparable statistical weight.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import platform
import statistics
import sys
import time
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Iterable


@dataclasses.dataclass(frozen=True)
class Case:
    """A single timed operation."""

    #: Name, unique within its family.
    name: str
    #: Builds the inputs and returns the callable to time.
    setup: Callable[[], Callable[[], object]]
    #: Work items processed per call (points, cells...), for the throughput.
    items: int
    #: Rebuild the inputs before every repetition, for operations that
    #: modify their inputs in place (gap filling, R-tree insertion...).
    fresh: bool = False


@dataclasses.dataclass(frozen=True)
class Context:
    """Run-wide parameters handed to the family modules."""

    #: Use the reduced problem sizes.
    quick: bool
    #: Value passed to every ``num_threads`` option (0 = all cores).
    threads: int
    #: Seed of the random generators, so that runs are comparable.
    seed: int = 42

    def size[T](self, full: T, quick: T) -> T:
        """Select the problem size matching the run mode."""
        return quick if self.quick else full

    def rng(self) -> np.random.Generator:
        """Return a freshly seeded random generator."""
        return np.random.default_rng(self.seed)


@dataclasses.dataclass
class Result:
    """Timings collected for one case."""

    family: str
    name: str
    items: int
    times: list[float]

    @property
    def key(self) -> str:
        """Identifier used to match results across runs."""
        return f"{self.family}/{self.name}"

    @property
    def best(self) -> float:
        """Fastest repetition, the least noisy estimator."""
        return min(self.times)

    @property
    def median(self) -> float:
        """Median repetition."""
        return statistics.median(self.times)

    @property
    def spread(self) -> float:
        """Relative standard deviation of the repetitions."""
        if len(self.times) < 2:  # noqa: PLR2004
            return 0.0
        return statistics.stdev(self.times) / statistics.mean(self.times)

    @property
    def throughput(self) -> float:
        """Items processed per second, from the best repetition."""
        return self.items / self.best if self.best > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result."""
        return {
            "items": self.items,
            "repeat": len(self.times),
            "best_s": self.best,
            "median_s": self.median,
            "spread": self.spread,
            "times_s": self.times,
        }


@dataclasses.dataclass(frozen=True)
class Timer:
    """Repetition policy."""

    min_repeat: int = 5
    max_repeat: int = 1000
    min_time: float = 0.5

    def run(self, family: str, case: Case) -> Result:
        """Time one case."""
        func = case.setup()
        func()  # Warm-up: page faults, lazy initialization, caches.
        times: list[float] = []
        # Time spent on the case, rebuilding the inputs included: a fast
        # operation whose inputs are slow to build would otherwise repeat up
        # to max_repeat times, spending nearly all the time in setup.
        total = 0.0
        while len(times) < self.max_repeat and (
            len(times) < self.min_repeat or total < self.min_time
        ):
            if case.fresh:
                start = time.perf_counter()
                func = case.setup()
                total += time.perf_counter() - start
            # The collector would otherwise fire at random inside the timed
            # region when an operation allocates many Python objects.
            gc.disable()
            try:
                start = time.perf_counter()
                func()
                elapsed = time.perf_counter() - start
            finally:
                gc.enable()
            times.append(elapsed)
            total += elapsed
        return Result(family, case.name, case.items, times)


def _format_time(seconds: float) -> str:
    """Format a duration with a suitable unit."""
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if seconds >= scale:
            return f"{seconds / scale:7.2f} {unit}"
    return f"{seconds / 1e-9:7.2f} ns"


def _format_rate(rate: float) -> str:
    """Format a throughput with a suitable multiplier."""
    for suffix, scale in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if rate >= scale:
            return f"{rate / scale:7.2f} {suffix}/s"
    return f"{rate:7.2f}  /s"


def print_header(family: str) -> None:
    """Print the table header of a family."""
    print(f"\n== {family} ".ljust(106, "="))
    print(
        f"{'case':<52} {'best':>10} {'median':>10} {'spread':>7} "
        f"{'throughput':>12} {'vs base':>9}"
    )


def print_result(result: Result, baseline: dict[str, Any] | None) -> None:
    """Print one row, with the speed-up over the baseline if known."""
    ratio = ""
    if baseline is not None and result.key in baseline:
        ratio = f"{baseline[result.key]['best_s'] / result.best:8.2f}x"
    print(
        f"{result.name:<52} {_format_time(result.best):>10} "
        f"{_format_time(result.median):>10} {result.spread:6.1%} "
        f"{_format_rate(result.throughput):>12} {ratio:>9}",
        flush=True,
    )


def metadata(ctx: Context, timer: Timer) -> dict[str, Any]:
    """Describe the environment, to judge whether two runs are comparable."""
    import pyinterp  # noqa: PLC0415

    return {
        "pyinterp": getattr(pyinterp, "__version__", "unknown"),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "quick": ctx.quick,
        "threads": ctx.threads,
        "seed": ctx.seed,
        "min_repeat": timer.min_repeat,
        "min_time": timer.min_time,
    }


def load_baseline(path: pathlib.Path) -> dict[str, Any]:
    """Load the results of a previous run."""
    with path.open() as stream:
        return json.load(stream)["results"]


def save(
    path: pathlib.Path, meta: dict[str, Any], results: Iterable[Result]
) -> None:
    """Write the results so that a later run can compare against them."""
    payload = {
        "metadata": meta,
        "results": {r.key: r.to_dict() for r in results},
    }
    with path.open("w") as stream:
        json.dump(payload, stream, indent=2)

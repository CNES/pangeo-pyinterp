# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Run the pyinterp benchmarks, grouped by algorithm family.

Examples:
    Run everything with the reduced sizes::

        python -m benchmarks --quick

    Record a baseline, change the C++ code, rebuild, then compare::

        python -m benchmarks grid -o before.json
        python -m benchmarks grid -c before.json

    Time only the windowed bivariate cases::

        python -m benchmarks grid -k bivariate/windowed

"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import sys

from . import _runner


#: Family name -> module implementing it, in execution order.
FAMILIES = {
    "axis": "Axis and temporal axis index lookups",
    "grid": "Interpolation on regular grids (1D to 4D)",
    "rtree": "Unstructured interpolation through R*-trees",
    "fill": "Gap filling of undefined grid values",
    "statistics": "Binning, histograms and streaming statistics",
    "geometry": "Geographic and Cartesian geometry algorithms",
    "geohash": "GeoHash encoding, decoding and indexing",
    "satellite": "Satellite track processing",
    "temporal": "Calendar decomposition and period lists",
}


def usage() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="families:\n"
        + "\n".join(f"  {k:<12} {v}" for k, v in FAMILIES.items()),
    )
    parser.add_argument(
        "families",
        nargs="*",
        metavar="family",
        help="families to run (default: all)",
    )
    parser.add_argument(
        "-k",
        "--filter",
        action="append",
        default=[],
        help="only run cases whose name contains this text (repeatable)",
    )
    parser.add_argument(
        "-q", "--quick", action="store_true", help="use reduced sizes"
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help="num_threads given to the operations, 0 = all cores "
        "(default: 1, to measure the kernels without scheduling noise)",
    )
    parser.add_argument(
        "-n",
        "--repeat",
        type=int,
        default=5,
        help="minimum number of timed repetitions (default: 5)",
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=0.5,
        help="minimum accumulated time per case, in seconds (default: 0.5)",
    )
    parser.add_argument(
        "-o", "--output", type=pathlib.Path, help="write the results as JSON"
    )
    parser.add_argument(
        "-c",
        "--compare",
        type=pathlib.Path,
        help="JSON file of a previous run; prints the speed-up over it",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="list the cases without running them",
    )
    args = parser.parse_args()
    unknown = set(args.families) - set(FAMILIES)
    if unknown:
        parser.error(f"unknown family: {', '.join(sorted(unknown))}")
    return args


def main() -> int:
    """Run the selected benchmarks."""
    args = usage()
    ctx = _runner.Context(quick=args.quick, threads=args.threads)
    timer = _runner.Timer(min_repeat=args.repeat, min_time=args.min_time)
    baseline = _runner.load_baseline(args.compare) if args.compare else None

    results: list[_runner.Result] = []
    for family in args.families or FAMILIES:
        module = importlib.import_module(f".{family}", __package__)
        cases = [
            case
            for case in module.cases(ctx)
            if all(item in case.name for item in args.filter)
        ]
        if not cases:
            continue
        if args.list:
            for case in cases:
                print(f"{family}/{case.name}")
            continue
        _runner.print_header(family)
        for case in cases:
            result = timer.run(family, case)
            _runner.print_result(result, baseline)
            results.append(result)

    if args.output and results:
        _runner.save(args.output, _runner.metadata(ctx, timer), results)
        print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

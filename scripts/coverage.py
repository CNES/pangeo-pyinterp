#!/usr/bin/env python3
# Copyright (c) 2025 CNES.
#
# This software is distributed by the CNES under a proprietary license.
# It is not public and cannot be redistributed or used without permission.
"""Summarizes the code coverage."""

import argparse


def usage() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="Summarizes the code coverage",
    )
    parser.add_argument(
        "tracefile",
        help="LCOV tracefile",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the main logic."""
    args = usage()

    # Sum the number of lines hit (LH) and found (LF) of each source file.
    # These records do not depend on the LCOV version, unlike the HTML report.
    samples = 0
    total = 0

    with open(args.tracefile) as stream:
        for line in stream:
            if line.startswith("LH:"):
                samples += int(line[3:])
            elif line.startswith("LF:"):
                total += int(line[3:])

    if total == 0:
        raise SystemExit(f"no line coverage data found in {args.tracefile}")

    print("-" * 80)
    print("{:^80}".format("Code Coverage Report"))
    print("-" * 80)
    print(f"TOTAL {round((samples / total) * 100):>73d}%")
    print("-" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2025 CNES.
#
# This software is distributed by the CNES under a proprietary license.
# It is not public and cannot be redistributed or used without permission.
"""Summarizes the code coverage."""

import argparse
import re


def usage() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="Summarizes the code coverage",
    )
    parser.add_argument(
        "index",
        help="LCOV index page",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the main logic."""
    args = usage()

    # First match is the line coverage, second match is the function coverage.
    # We only want the line coverage.
    line_coverage = True

    pattern = re.compile(r"(\d+)\s*\/\s*(\d+)").search
    samples: float = 0
    total: float = 0

    for line in open(args.index):
        match = pattern(line)
        if match is not None:
            if line_coverage:
                samples += float(match.group(1))
                total += float(match.group(2))
            line_coverage = not line_coverage

    print("-" * 80)
    print("{:^80}".format("Code Coverage Report"))
    print("-" * 80)
    print(f"TOTAL {round((samples / total) * 100):>73d}%")
    print("-" * 80)


if __name__ == "__main__":
    main()

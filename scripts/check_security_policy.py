#!/usr/bin/env python3
# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Check that SECURITY.md supports the year of the version being released.

The versions follow the ``YEAR.MONTH.PATCH`` scheme and the security policy
supports the releases of the current year, listed as ``YEAR.x`` in the table
of SECURITY.md. The table has to be updated by hand with the first release of
each year: the release workflows run this script so that a release cannot be
published while the policy still lists the previous year.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


#: Matches the year that starts a version or a release tag, e.g. ``2026`` in
#: ``2026.4.0rc1``.
_VERSION_YEAR = re.compile(r"^v?(?P<year>\d{4})\.")

#: Matches a row of the table of supported versions, e.g.
#: ``| 2026.x  | :white_check_mark: |``.
_SUPPORTED_ROW = re.compile(
    r"^\|\s*(?P<year>\d{4})\.x\s*\|\s*:white_check_mark:\s*\|", re.MULTILINE
)

#: Path of the security policy, relative to the root of the repository.
SECURITY_POLICY = pathlib.Path(__file__).parent.parent / "SECURITY.md"


def supported_years(policy: str) -> set[int]:
    """Return the years whose releases the security policy supports.

    Args:
        policy: Content of SECURITY.md.

    Returns:
        The years marked as supported in the table of supported versions.

    """
    return {int(m.group("year")) for m in _SUPPORTED_ROW.finditer(policy)}


def main() -> int:
    """Run the check and return the exit status of the script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version", help="version or release tag, e.g. 2026.4.0rc1"
    )
    parser.add_argument(
        "--policy",
        type=pathlib.Path,
        default=SECURITY_POLICY,
        help="path of the security policy (default: %(default)s)",
    )
    args = parser.parse_args()

    match = _VERSION_YEAR.match(args.version)
    if match is None:
        print(
            f"error: {args.version!r} does not start with a year.",
            file=sys.stderr,
        )
        return 2
    year = int(match.group("year"))

    years = supported_years(args.policy.read_text(encoding="utf-8"))
    if year not in years:
        listed = ", ".join(f"{item}.x" for item in sorted(years)) or "none"
        print(
            f"error: {args.policy.name} does not list {year}.x as a "
            f"supported version (supported: {listed}). Update its table of "
            "supported versions before releasing.",
            file=sys.stderr,
        )
        return 1

    print(f"{args.policy.name} supports {year}.x.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Update the copyright notice in the files."""

import datetime
import os
import re

#: Copyright notice
COPYRIGHT = """{c} Copyright (c) {year} CNES.
{c}
{c} All rights reserved. Use of this source code is governed by a
{c} BSD-style license that can be found in the LICENSE file."""

#: Root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def update_copyright(
    comment_delimiter: str, year: int, ext: tuple[str, ...], dir: str
) -> None:
    """Update the copyright notice in the files.

    Args:
        comment_delimiter: Comment delimiter
        year: Year
        ext: File extensions
        dir: Directory

    """
    copyright = re.escape(COPYRIGHT.format(c=comment_delimiter, year=year))
    copyright = copyright.replace(str(year), r"\d{4}")

    pattern = re.compile(f"({copyright})").search

    for root, _, files in os.walk(dir):
        for file in files:
            if file in ["version.hpp"]:
                continue
            file_ext = os.path.splitext(file)[1]
            if file_ext in ext:
                path = os.path.join(root, file)
                with open(path, encoding="utf-8") as stream:
                    content = stream.read()
                match = pattern(content)
                pos = int(content.startswith("#!"))
                if not match:
                    lines = content.split("\n")
                    lines.insert(
                        pos,
                        COPYRIGHT.format(
                            c=comment_delimiter,
                            year=year,
                        ),
                    )
                    content = "\n".join(lines)
                else:
                    content = content.replace(
                        match.group(1),
                        COPYRIGHT.format(
                            c=comment_delimiter,
                            year=year,
                        ),
                    )
                with open(path, "w", encoding="utf-8") as stream:
                    stream.write(content)


def main() -> None:
    """Execute the main script logic."""
    year = datetime.date.today().year
    for comment_delimiter, ext, dir in [
        ("//", (".hpp", ".cpp"), "cxx"),
        ("#", (".txt",), "cxx"),
        ("#", (".py",), "pyinterp"),
        ("#", (".py", ".sh"), "scripts"),
    ]:
        update_copyright(
            comment_delimiter,
            year,
            ext,
            os.path.join(ROOT_DIR, dir),
        )


if __name__ == "__main__":
    main()

# Copyright (c) 2025 CNES.
#
# This software is distributed by the CNES under a proprietary license.
# It is not public and cannot be redistributed or used without permission.
"""Simple script to print the path of temporary build directory."""

import pathlib
import sys
import sysconfig

# Check Python requirement
MAJOR = sys.version_info[0]
MINOR = sys.version_info[1]

# Working directory
WORKING_DIRECTORY = pathlib.Path(__file__).parent.parent.absolute()


def build_dirname() -> pathlib.Path:
    """Return the build directory name."""
    path = pathlib.Path(
        WORKING_DIRECTORY,
        "build",
        f"lib.{sysconfig.get_platform()}-{MAJOR}.{MINOR}",
    )
    if path.exists():
        return path
    return pathlib.Path(
        WORKING_DIRECTORY,
        "build",
        f"temp.{sysconfig.get_platform()}-{sys.implementation.cache_tag}",
    )


def main() -> None:
    """Print the path of the compile_commands.json file."""
    print(str(build_dirname().resolve()))


if __name__ == "__main__":
    main()

"""
Generates stubs for the core modules.
=====================================
"""
from typing import List
import pathlib
import sys
import os
import re
import mypy.stubgen
import yapf

PATTERN = re.compile(r"(numpy\.\w+\d+)(\[\w+,\w+\])").search
GRID = re.compile(r"class (.*Grid\dD\w+):").search


def fix_core(src: pathlib.Path):
    grids = []
    core = src / "pyinterp" / "core" / "__init__.pyi"
    with core.open("r") as stream:
        lines = stream.readlines()

    lines[0] = lines[0].rstrip() + ", overload\n"
    lines.pop(1)
    lines.pop(1)
    for item in reversed([
            "from . import dateutils\n", "from . import geodetic\n",
            "from . import geohash\n", "from . import fill\n", "\n"
    ]):
        lines.insert(2, item)

    for item in lines:
        m = GRID(item)
        if m:
            grids.append(m.group(1))

    with core.open("w") as stream:
        stream.writelines(lines)

    return grids


def fix_core_geodetic(src: pathlib.Path):
    core = src / "pyinterp" / "core" / "geodetic.pyi"
    with core.open("r") as stream:
        lines = stream.readlines()

    for ix, item in enumerate(
        ["", "import numpy\n"
         "from .. import geodetic\n", "", "", ""]):
        lines[ix] = item if ix else lines[ix].rstrip() + ", overload\n"

    for ix, item in enumerate(lines):
        m = PATTERN(item)
        while m:
            item = item.replace(m.group(1) + m.group(2), m.group(1))
            m = PATTERN(item)
        lines[ix] = item

    with core.open("w") as stream:
        stream.writelines(lines)


def fix_core_fill(src: pathlib.Path, grids: List[str]):
    core = src / "pyinterp" / "core" / "fill.pyi"
    with core.open("r") as stream:
        lines = stream.readlines()

    for ix, item in enumerate([
            "", "import numpy\n"
            f"from . import ({','.join(grids)},)\n", "", "", "", ""
    ]):
        lines[ix] = item if ix else lines[ix].rstrip() + ", overload\n"

    for ix, item in enumerate(lines):
        item = item.replace("pyinterp.core.", "")
        item = item.replace(",flags.writeable", "")
        m = PATTERN(item)
        while m:
            item = item.replace(m.group(1) + m.group(2), m.group(1))
            m = PATTERN(item)
        lines[ix] = item

    with core.open("w") as stream:
        stream.writelines(lines)


def fix_core_geohash(src: pathlib.Path):
    for stub in ["int64.pyi", "__init__.pyi"]:
        core = src / "pyinterp" / "core" / "geohash" / stub
        with core.open("r") as stream:
            lines = stream.readlines()

        for ix, item in enumerate(
            ["", "import numpy\n"
             "from .. import geodetic\n", "", "", ""]):
            lines[ix] = item if ix else lines[ix].rstrip() + ", overload\n"

        for ix, item in enumerate(lines):
            item = item.replace("pyinterp.core.geodetic", "geodetic")
            m = PATTERN(item)
            while m:
                item = item.replace(m.group(1) + m.group(2), m.group(1))
                m = PATTERN(item)
            lines[ix] = item

        with core.open("w") as stream:
            stream.writelines(lines)


def fix_core_storage(src: pathlib.Path):
    core = src / "pyinterp" / "core" / "storage" / "__init__.pyi"
    with core.open("r") as stream:
        lines = stream.readlines()

    lines.insert(0, "from . import unqlite\n")

    with core.open("w") as stream:
        stream.writelines(lines)


def main():
    modules = [
        # "pyinterp.core.dateutils",
        "pyinterp.core.fill",
        "pyinterp.core.geodetic",
        "pyinterp.core.geohash.int64",
        "pyinterp.core.geohash",
        "pyinterp.core.storage.unqlite",
        "pyinterp.core.storage",
        "pyinterp.core",
    ]
    out = pathlib.Path(__file__).parent / "src"
    options = mypy.stubgen.Options(pyversion=(sys.version_info[0],
                                              sys.version_info[1]),
                                   no_import=False,
                                   doc_dir='',
                                   search_path=[''],
                                   interpreter=sys.executable,
                                   parse_only=False,
                                   ignore_errors=False,
                                   include_private=False,
                                   output_dir=str(out),
                                   modules=modules,
                                   packages=[],
                                   files=[],
                                   verbose=False,
                                   quiet=True,
                                   export_less=False)
    mypy.stubgen.generate_stubs(options)
    grids = fix_core(out)
    fix_core_fill(
        out,
        [grid for grid in grids if re.compile(r'Grid[2-3]D').search(grid)])
    fix_core_geodetic(out)
    fix_core_geohash(out)
    fix_core_storage(out)

    stubs = []
    for root, dirs, files in os.walk(str(out)):
        for name in files:
            if name.endswith(".pyi"):
                stubs.append(str(pathlib.Path(root) / name))
    yapf.main([sys.argv[0], "-i"] + stubs)


if __name__ == '__main__':
    main()

from typing import Optional
import pathlib
import pytest

ROOT = pathlib.Path(__file__).parent.joinpath("dataset").resolve()


def grid2d_path():
    """Return path to the Grid 2D"""
    return ROOT.joinpath("mss.nc")


def aoml_path():
    """Return path to the AOML dataset"""
    return ROOT.joinpath("aoml_v2019.nc")


def positions_path():
    """Return path to the ARGO positions"""
    return ROOT.joinpath("positions.csv")


def grid3d_path():
    """Return path to the Grid 3D"""
    return ROOT.joinpath("tcw.nc")


def grid4d_path():
    """Return path to the Grid 4D"""
    return ROOT.joinpath("pres_temp_4D.nc")


def run(pattern: Optional[str] = None) -> None:
    """Run tests.

    Args:
        pattern (str, optional): A regex pattern to match against test names.
    """
    args = ["-x", str(pathlib.Path(__file__).parent.resolve())]
    if pattern is not None:
        args += ["-k", pattern]
    pytest.main(args)

from typing import Optional
import pathlib

import numpy
import pytest

ROOT = pathlib.Path(__file__).parent.joinpath("dataset").resolve()


def make_or_compare_reference(filename: str, values: numpy.ndarray,
                              dump: bool) -> None:
    """Make a reference file or compare against it.

    Args:
        filename (str): The filename to create or compare against.
        values (np.ndarray): The values to compare against.
        dump (bool): Whether to dump the values to the file.
    """
    path = ROOT.joinpath(filename)
    if dump:
        numpy.save(path, values)
        return
    if path.exists():
        reference = numpy.load(path)
        assert numpy.allclose(reference, values, equal_nan=True)


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


def geohash_bbox_path():
    """Return path to the GeoHash bounding box"""
    return ROOT.joinpath("geohash_bbox.json")


def geohash_neighbors_path():
    """Return path to the GeoHash neighbors"""
    return ROOT.joinpath("geohash_neighbors.json")


def geohash_path():
    """Return path to the GeoHash dataset"""
    return ROOT.joinpath("geohash.json")


def polygon_path():
    """Return path to the polygon dataset"""
    return ROOT.joinpath("polygon.json")


def run(pattern: Optional[str] = None) -> None:
    """Run tests.

    Args:
        pattern (str, optional): A regex pattern to match against test names.
    """
    args = ["-x", str(pathlib.Path(__file__).parent.resolve())]
    if pattern is not None:
        args += ["-k", pattern]
    pytest.main(args)

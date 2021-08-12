import pathlib

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
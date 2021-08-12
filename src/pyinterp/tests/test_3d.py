# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import datetime
import collections
import os
import pickle
import pytest
import numpy as np
import xarray as xr
import pyinterp.backends.xarray
import pyinterp
from . import grid3d_path


def test_3d():
    grid = pyinterp.backends.xarray.Grid3D(xr.load_dataset(grid3d_path()).tcw,
                                           increasing_axes=True)

    assert isinstance(grid, pyinterp.backends.xarray.Grid3D)
    assert isinstance(grid, pyinterp.Grid3D)
    other = pickle.loads(pickle.dumps(grid))
    assert isinstance(other, pyinterp.backends.xarray.Grid3D)
    assert isinstance(grid, pyinterp.Grid3D)

    assert isinstance(grid.x, pyinterp.Axis)
    assert isinstance(grid.y, pyinterp.Axis)
    assert isinstance(grid.z, pyinterp.TemporalAxis)
    assert isinstance(grid.array, np.ndarray)

    lon = np.arange(-180, 180, 1) + 1 / 3.0
    lat = np.arange(-90, 90, 1) + 1 / 3.0
    time = np.array([datetime.datetime(2002, 7, 2, 15, 0)], dtype="datetime64")
    x, y, t = np.meshgrid(lon, lat, time, indexing="ij")

    z = grid.trivariate(
        collections.OrderedDict(longitude=x.flatten(),
                                latitude=y.flatten(),
                                time=t.flatten()))
    assert isinstance(z, np.ndarray)

    z = grid.bicubic(
        collections.OrderedDict(longitude=x.flatten()[1:2],
                                latitude=y.flatten()[1:2],
                                time=t.flatten()[1:2]))
    assert isinstance(z, np.ndarray)

    with pytest.raises(ValueError):
        time = np.array([datetime.datetime(2012, 7, 2, 15, 0)],
                        dtype="datetime64")
        x, y, t = np.meshgrid(lon, lat, time, indexing="ij")
        grid.trivariate(collections.OrderedDict(longitude=x.flatten(),
                                                latitude=y.flatten(),
                                                time=t.flatten()),
                        bounds_error=True)

    array = xr.load_dataset(grid3d_path()).tcw
    grid = pyinterp.backends.xarray.Grid3D(array, increasing_axes=True)
    x, y, t = np.meshgrid(lon, lat, time, indexing="ij")
    z = grid.trivariate(
        collections.OrderedDict(longitude=x.flatten(),
                                latitude=y.flatten(),
                                time=t.flatten()))
    assert isinstance(z, np.ndarray)

    grid = pyinterp.backends.xarray.RegularGridInterpolator(
        xr.load_dataset(grid3d_path()).tcw, increasing_axes=True)
    assert grid.ndim == 3
    assert isinstance(grid.grid, pyinterp.backends.xarray.Grid3D)
    x, y, t = np.meshgrid(lon, lat, time, indexing="ij")
    z = grid(
        dict(longitude=x.flatten(), latitude=y.flatten(), time=t.flatten()))
    assert isinstance(z, np.ndarray)

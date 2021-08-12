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
from . import grid4d_path


def test_4d():
    grid = pyinterp.backends.xarray.Grid4D(xr.load_dataset(
        grid4d_path()).pressure,
                                           increasing_axes=True)

    assert isinstance(grid, pyinterp.backends.xarray.Grid4D)
    assert isinstance(grid, pyinterp.Grid4D)
    other = pickle.loads(pickle.dumps(grid))
    assert isinstance(other, pyinterp.backends.xarray.Grid4D)
    assert isinstance(grid, pyinterp.Grid4D)

    assert isinstance(grid.x, pyinterp.Axis)
    assert isinstance(grid.y, pyinterp.Axis)
    assert isinstance(grid.z, pyinterp.TemporalAxis)
    assert isinstance(grid.u, pyinterp.Axis)
    assert isinstance(grid.array, np.ndarray)

    lon = np.arange(-125, -70, 0.25)
    lat = np.arange(-25, 50, 0.25)
    level = 0.5
    time = np.datetime64('2000-01-01T12:00')
    x, y, z, t = np.meshgrid(lon, lat, level, time, indexing="ij")

    pressure = grid.quadrivariate(
        collections.OrderedDict(longitude=x.flatten(),
                                latitude=y.flatten(),
                                level=z.flatten(),
                                time=t.flatten()))
    assert isinstance(pressure, np.ndarray)

    pressure = grid.bicubic(
        collections.OrderedDict(longitude=x.flatten(),
                                latitude=y.flatten(),
                                level=z.flatten(),
                                time=t.flatten()))
    assert isinstance(pressure, np.ndarray)

    with pytest.raises(ValueError):
        time = 5
        x, y, t = np.meshgrid(lon, lat, level, time, indexing="ij")
        pressure = grid.quadrivariate(collections.OrderedDict(
            longitude=x.flatten(),
            latitude=y.flatten(),
            level=z.flatten(),
            time=t.flatten()),
                                      bounds_error=True)

    with pytest.raises(ValueError):
        pressure = grid.bicubic(collections.OrderedDict(longitude=x.flatten(),
                                                        latitude=y.flatten(),
                                                        level=z.flatten(),
                                                        time=t.flatten()),
                                bounds_error=True)

    grid = pyinterp.backends.xarray.RegularGridInterpolator(
        xr.load_dataset(grid4d_path()).pressure, increasing_axes=True)
    assert grid.ndim, 4
    assert isinstance(grid.grid, pyinterp.backends.xarray.Grid4D)


def test_4d_degraded():
    grid = pyinterp.backends.xarray.Grid4D(xr.load_dataset(
        grid4d_path()).pressure,
                                           increasing_axes=True)
    zero = np.array([0])
    with pytest.raises(ValueError):
        pyinterp.bicubic(grid, zero, zero, zero)

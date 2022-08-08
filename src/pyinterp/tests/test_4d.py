# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import collections
import pickle

import numpy as np
import pytest
import xarray as xr

import pyinterp

from . import load_grid4d, make_or_compare_reference
from .. import Axis, Grid4D, TemporalAxis, bicubic
from ..backends import xarray as xr_backend


def test_4d(pytestconfig):
    grid = xr_backend.Grid4D(load_grid4d().pressure, increasing_axes=True)

    assert isinstance(grid, xr_backend.Grid4D)
    assert isinstance(grid, Grid4D)
    other = pickle.loads(pickle.dumps(grid))
    assert isinstance(other, xr_backend.Grid4D)
    assert isinstance(grid, Grid4D)

    assert isinstance(grid.x, Axis)
    assert isinstance(grid.y, Axis)
    assert isinstance(grid.z, TemporalAxis)
    assert isinstance(grid.u, Axis)
    assert isinstance(grid.array, np.ndarray)

    lon = np.arange(-125, -70, 0.25)
    lat = np.arange(-25, 50, 0.25)
    level = 0.5
    time = np.datetime64('2000-01-01T12:00')
    x, y, z, t = np.meshgrid(lon, lat, level, time, indexing='ij')

    pressure = grid.quadrivariate(
        collections.OrderedDict(longitude=x.ravel(),
                                latitude=y.ravel(),
                                level=z.ravel(),
                                time=t.ravel()))
    assert isinstance(pressure, np.ndarray)
    make_or_compare_reference('4d_pressure_bilinear.npy', pressure,
                              pytestconfig.getoption('dump'))

    pressure = grid.bicubic(
        collections.OrderedDict(longitude=x.ravel(),
                                latitude=y.ravel(),
                                level=z.ravel(),
                                time=t.ravel()))
    assert isinstance(pressure, np.ndarray)
    make_or_compare_reference('4d_pressure_bicubic.npy', pressure,
                              pytestconfig.getoption('dump'))

    with pytest.raises(ValueError):
        time = 5
        x, y, t = np.meshgrid(lon, lat, level, time, indexing='ij')
        pressure = grid.quadrivariate(collections.OrderedDict(
            longitude=x.ravel(),
            latitude=y.ravel(),
            level=z.ravel(),
            time=t.ravel()),
                                      bounds_error=True)

    with pytest.raises(ValueError):
        pressure = grid.bicubic(collections.OrderedDict(longitude=x.ravel(),
                                                        latitude=y.ravel(),
                                                        level=z.ravel(),
                                                        time=t.ravel()),
                                bounds_error=True)

    grid = xr_backend.RegularGridInterpolator(load_grid4d().pressure,
                                              increasing_axes=True)
    assert grid.ndim, 4
    assert isinstance(grid.grid, xr_backend.Grid4D)


def test_4d_swap_dim():
    ds = load_grid4d()
    ds = ds.transpose('level', 'latitude', 'longitude', 'time')
    grid = xr_backend.Grid4D(ds.pressure, increasing_axes=True)
    assert isinstance(grid.z, pyinterp.TemporalAxis)
    assert grid.array.shape == (12, 6, 2, 2)

    ds = ds.assign_coords(level=ds.level.values.astype('datetime64[s]'))
    with pytest.raises(ValueError):
        grid = xr_backend.Grid4D(ds.pressure, increasing_axes=True)


def test_4d_degraded():
    grid = xr_backend.Grid4D(load_grid4d().pressure, increasing_axes=True)
    zero = np.array([0])
    with pytest.raises(ValueError):
        bicubic(grid, zero, zero, zero)

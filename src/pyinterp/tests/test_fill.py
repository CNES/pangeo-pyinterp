# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import numpy as np
import pytest

from pyinterp.core import TemporalAxis
from pyinterp.grid import Grid4D

from . import load_grid2d, load_grid4d
from .. import Axis, Grid2D, Grid3D, fill


def load_data(cube=False):
    ds = load_grid2d()  # type: ignore
    x_axis = Axis(ds['lon'].values[::5], is_circle=True)
    y_axis = Axis(ds['lat'].values[::5])
    mss = ds['mss'].values[::5, ::5].T
    if cube:
        z_axis = Axis(np.arange(2))
        mss = np.stack([mss.data] * len(z_axis)).transpose(1, 2, 0)
        return Grid3D(x_axis, y_axis, z_axis, mss)
    return Grid2D(x_axis, y_axis, mss)


def test_loess():
    grid = load_data()
    filled0 = fill.loess(grid, num_threads=0)
    filled1 = fill.loess(grid, num_threads=1)
    data = np.copy(grid.array)
    data[np.isnan(data)] = 0
    filled0[np.isnan(filled0)] = 0
    filled1[np.isnan(filled1)] = 0
    assert (filled0 - filled1).mean() == 0
    assert np.ma.fix_invalid(grid.array - filled1).mean() == 0
    assert (data - filled1).mean() != 0

    with pytest.raises(ValueError):
        fill.loess(grid, value_type='x')


def test_gauss_seidel():
    grid = load_data()
    _, filled0 = fill.gauss_seidel(grid, num_threads=0)
    _, filled1 = fill.gauss_seidel(grid, num_threads=1)
    _, filled2 = fill.gauss_seidel(grid, first_guess='zero', num_threads=0)
    data = np.copy(grid.array)
    data[np.isnan(data)] = 0
    filled0[np.isnan(filled0)] = 0
    filled1[np.isnan(filled1)] = 0
    filled2[np.isnan(filled2)] = 0
    assert (filled0 - filled1).mean() == 0
    assert np.ma.fix_invalid(grid.array - filled1).mean() == 0
    assert (data - filled1).mean() != 0
    assert (filled2 - filled1).mean() != 0

    with pytest.raises(ValueError):
        fill.gauss_seidel(grid, '_')

    x_axis = Axis(np.linspace(-180, 180, 10), is_circle=True)
    y_axis = Axis(np.linspace(-90, 90, 10), is_circle=False)
    data = np.random.rand(len(x_axis), len(y_axis))
    grid = Grid2D(x_axis, y_axis, data)
    _, filled0 = fill.gauss_seidel(grid, num_threads=0)
    assert isinstance(filled0, np.ndarray)


def test_loess_3d():
    grid = load_data(True)
    mask = np.isnan(grid.array)
    filled0 = fill.loess(grid, num_threads=0)
    filled0[mask] = np.nan
    assert np.nanmean(filled0 - grid.array) == 0

    with pytest.raises(ValueError):
        fill.loess(grid, num_threads=0, nx=0, ny=1)

    with pytest.raises(ValueError):
        fill.loess(grid, num_threads=0, nx=1, ny=0)


def test_gauss_seidel_3d():
    grid = load_data(True)
    _, filled0 = fill.gauss_seidel(grid, num_threads=0)
    assert (filled0[:, :, 0] - filled0[:, :, 1]).mean() == 0


def test_loess_4d():
    variable = load_grid4d().pressure
    variable = variable.transpose('longitude', 'latitude', 'time', 'level')
    x_axis = Axis(variable['longitude'].values, is_circle=True)
    y_axis = Axis(variable['latitude'].values)
    z_axis = TemporalAxis(variable['time'].values)
    u_axis = Axis(variable['level'].values)

    grid = Grid4D(x_axis, y_axis, z_axis, u_axis, variable.values)
    filled = fill.loess(grid, num_threads=0)
    assert np.nanmean(filled - grid.array) == 0

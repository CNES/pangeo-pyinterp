# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for the `pyinterp.fill` module."""
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyinterp.core import TemporalAxis
from pyinterp.grid import Grid4D

from . import load_grid2d, load_grid4d
from .. import Axis, Grid2D, Grid3D, fill

if TYPE_CHECKING:
    from ..typing import NDArrayDateTime


def load_data(cube: bool = False) -> Grid3D | Grid2D:
    """Load test data."""
    ds = load_grid2d()
    x_axis = Axis(ds['lon'].values[::5], is_circle=True)
    y_axis = Axis(ds['lat'].values[::5])
    mss = ds['mss'].values[::5, ::5].T
    if cube:
        z_axis = Axis(np.arange(2, dtype=np.float64))
        mss = np.stack([mss.data] * len(z_axis)).transpose(1, 2, 0)
        return Grid3D(x_axis, y_axis, z_axis, mss)
    return Grid2D(x_axis, y_axis, mss)


def test_loess() -> None:
    """Test the fill.loess function with 2D grids."""
    grid = load_data()
    assert isinstance(grid, Grid2D)
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


def test_gauss_seidel() -> None:
    """Test the fill.gauss_seidel function with 2D grids."""
    grid = load_data()
    assert isinstance(grid, Grid2D)
    _, _, filled0 = fill.gauss_seidel(grid.array,
                                      is_circle=grid.x.is_circle,
                                      num_threads=0)
    _, _, filled1 = fill.gauss_seidel(grid.array,
                                      is_circle=grid.x.is_circle,
                                      num_threads=1)
    _, _, filled2 = fill.gauss_seidel(grid.array,
                                      is_circle=grid.x.is_circle,
                                      first_guess='zero',
                                      num_threads=0)
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
        fill.gauss_seidel(grid.array,
                          is_circle=grid.x.is_circle,
                          first_guess='_')

    x_axis = Axis(np.linspace(-180, 180, 10), is_circle=True)
    y_axis = Axis(np.linspace(-90, 90, 10), is_circle=False)
    rng = np.random.default_rng(42)
    data = rng.random((len(x_axis), len(y_axis)))
    grid = Grid2D(x_axis, y_axis, data)
    _, _, filled0 = fill.gauss_seidel(grid.array,
                                      is_circle=grid.x.is_circle,
                                      num_threads=0)
    assert isinstance(filled0, np.ndarray)


def test_loess_3d() -> None:
    """Test the fill.loess function with 3D grids."""
    grid = load_data(True)
    assert isinstance(grid, Grid3D)
    mask = np.isnan(grid.array)
    filled0 = fill.loess(grid, num_threads=0)
    filled0[mask] = np.nan
    assert np.nanmean(filled0 - grid.array) == 0

    with pytest.raises(ValueError):
        fill.loess(grid, num_threads=0, nx=0, ny=1)

    with pytest.raises(ValueError):
        fill.loess(grid, num_threads=0, nx=1, ny=0)


def test_gauss_seidel_dynamic_strides() -> None:
    """Test that gauss_seidel accepts dynamic strides."""
    grid = load_data(True)
    assert isinstance(grid, Grid3D)

    # But should work on individual slices
    _, _, filled_slice = fill.gauss_seidel(grid.array[:, :, 0],
                                           is_circle=grid.x.is_circle,
                                           num_threads=0)
    assert filled_slice.shape == grid.array[:, :, 0].shape


def test_loess_4d() -> None:
    """Test the fill.loess function with 4D grids."""
    variable = load_grid4d().pressure
    variable = variable.transpose('longitude', 'latitude', 'time', 'level')
    x_axis = Axis(variable['longitude'].values, is_circle=True)
    y_axis = Axis(variable['latitude'].values)
    z_axis = TemporalAxis(variable['time'].values)
    u_axis = Axis(variable['level'].values)

    grid = Grid4D(x_axis, y_axis, z_axis, u_axis, variable.values)
    filled = fill.loess(grid, num_threads=0)
    assert np.nanmean(filled - grid.array) == 0


def test_matrix() -> None:
    """Test the fill.matrix function."""
    x = np.ndarray([10, 4])
    x[:, 0] = np.array([0, 1, 2, 3, 4, 5, 6, np.nan, 8, 9], dtype=np.float64)
    x[:, 1] = np.array([0, np.nan, 2, 3, 4, 5, 6, np.nan, 8, 9],
                       dtype=np.float64)
    x[:, 2] = np.full(10, np.nan, dtype=np.float64)
    x[:, 3] = np.array([0, np.nan, 2, 3, 4, 5, 6, np.nan, 8, 9],
                       dtype=np.float64)
    y = np.copy(x)

    fill.matrix(x)
    fill.matrix(y)

    assert np.all(x[:, 0] == np.arange(10))
    assert np.all(x[:, 1] == np.arange(10))
    assert np.all(x[:, 2] == np.arange(10))
    assert np.all(x[:, 3] == np.arange(10))
    assert np.all(y[:, 0] == np.arange(10))
    assert np.all(y[:, 1] == np.arange(10))
    assert np.all(y[:, 2] == np.arange(10))
    assert np.all(y[:, 3] == np.arange(10))


def test_vector() -> None:
    """Test the fill.vector function with datetime64 arrays."""
    x: NDArrayDateTime = np.arange(
        np.datetime64('2000-01-01'),
        np.datetime64('2000-01-10'),
        np.timedelta64(1, 'h'),
    )
    rng = np.random.default_rng()
    random_indices = rng.choice(len(x), 48, replace=False)
    xp = np.copy(x)
    xp[random_indices] = np.datetime64('NaT')
    xp[:2] = np.datetime64('NaT')
    xp[-2:] = np.datetime64('NaT')
    yp = fill.vector(xp, fill_value=np.datetime64('NaT'))
    assert np.all(yp == x)

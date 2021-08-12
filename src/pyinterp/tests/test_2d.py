# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import collections
import os
import pickle
import pytest
import numpy as np
import xarray as xr
import pyinterp.backends.xarray
import pyinterp
from . import grid2d_path


def test_axis_identifier():
    ident = pyinterp.backends.xarray.AxisIdentifier(xr.DataArray())
    assert ident.longitude() is None
    assert ident.latitude() is None


def test_dims_from_data_array():
    array = xr.DataArray()
    with pytest.raises(ValueError):
        pyinterp.backends.xarray._dims_from_data_array(array, True, 1)
    array = xr.DataArray(data=np.zeros((2, 2), dtype="float64"))
    with pytest.raises(ValueError):
        pyinterp.backends.xarray._dims_from_data_array(array, True, 2)
    array = xr.DataArray(data=np.zeros((2, 2), dtype="float64"),
                         coords=[("lon", np.arange(2),
                                  dict(units="degrees_east")),
                                 ("lat", np.arange(2))],
                         dims=['lon', 'lat'])
    with pytest.raises(ValueError):
        pyinterp.backends.xarray._dims_from_data_array(array, True, 2)


def test_biavariate():
    grid = pyinterp.backends.xarray.Grid2D(xr.load_dataset(grid2d_path()).mss)

    assert isinstance(grid, pyinterp.backends.xarray.Grid2D)
    assert isinstance(grid, pyinterp.Grid2D)
    other = pickle.loads(pickle.dumps(grid))
    assert isinstance(other, pyinterp.backends.xarray.Grid2D)
    assert isinstance(grid, pyinterp.Grid2D)

    assert isinstance(grid.x, pyinterp.Axis)
    assert isinstance(grid.y, pyinterp.Axis)
    assert isinstance(grid.array, np.ndarray)

    lon = np.arange(-180, 180, 1) + 1 / 3.0
    lat = np.arange(-90, 90, 1) + 1 / 3.0
    x, y = np.meshgrid(lon, lat, indexing="ij")

    z = grid.bivariate(
        collections.OrderedDict(lon=x.flatten(), lat=y.flatten()))
    assert isinstance(z, np.ndarray)

    z = grid.bivariate(collections.OrderedDict(lon=x.flatten(),
                                               lat=y.flatten()),
                       interpolator="nearest")
    assert isinstance(z, np.ndarray)

    z = grid.bivariate(collections.OrderedDict(lon=x.flatten(),
                                               lat=y.flatten()),
                       interpolator="inverse_distance_weighting")
    assert isinstance(z, np.ndarray)

    grid = pyinterp.backends.xarray.Grid2D(xr.load_dataset(grid2d_path()).mss,
                                           geodetic=False)

    assert isinstance(grid, pyinterp.backends.xarray.Grid2D)
    w = grid.bivariate(collections.OrderedDict(lon=x.flatten(),
                                               lat=y.flatten()),
                       interpolator="inverse_distance_weighting")
    assert np.ma.fix_invalid(z).mean() != np.ma.fix_invalid(w).mean()

    with pytest.raises(TypeError):
        grid.bivariate((x.flatten(), y.flatten()))  # type: ignore

    with pytest.raises(IndexError):
        grid.bivariate(
            collections.OrderedDict(lon=x.flatten(),
                                    lat=y.flatten(),
                                    time=np.arange(3)))

    with pytest.raises(IndexError):
        grid.bivariate(
            collections.OrderedDict(longitude=x.flatten(), lat=y.flatten()))

    with pytest.raises(ValueError):
        grid.bivariate(collections.OrderedDict(lon=x.flatten(),
                                               lat=y.flatten()),
                       bounds_error=True)

    lon = pyinterp.Axis(np.linspace(0, 360, 100), is_circle=True)
    lat = pyinterp.Axis(np.linspace(-80, 80, 50), is_circle=False)
    array, _ = np.meshgrid(lon[:], lat[:])

    with pytest.raises(ValueError):
        pyinterp.Grid2D(lon, lat, array)

    grid = pyinterp.Grid2D(lon, lat, array.T)

    assert isinstance(grid, pyinterp.Grid2D)
    assert isinstance(str(grid), str)

    with pytest.raises(ValueError):
        pyinterp.Grid2D(lon, lat, array, increasing_axes='_')

    grid = pyinterp.backends.xarray.RegularGridInterpolator(
        xr.load_dataset(grid2d_path()).mss)
    z = grid(collections.OrderedDict(lon=x.flatten(), lat=y.flatten()),
             method="bilinear")
    assert isinstance(z, np.ndarray)


def test_bicubic():
    grid = pyinterp.backends.xarray.Grid2D(xr.load_dataset(grid2d_path()).mss)

    lon = np.arange(-180, 180, 1) + 1 / 3.0
    lat = np.arange(-90, 90, 1) + 1 / 3.0
    x, y = np.meshgrid(lon, lat, indexing="ij")

    z = grid.bicubic(collections.OrderedDict(lon=x.flatten(), lat=y.flatten()))
    assert isinstance(z, np.ndarray)

    for fitting_model in [
            'linear', 'bicubic', 'polynomial', 'c_spline', 'c_spline_periodic',
            'akima', 'akima_periodic', 'steffen'
    ]:
        other = grid.bicubic(collections.OrderedDict(lon=x.flatten(),
                                                     lat=y.flatten()),
                             fitting_model=fitting_model)
        assert (z - other).mean() != 0

    with pytest.raises(ValueError):
        grid.bicubic(collections.OrderedDict(lon=x.flatten(), lat=y.flatten()),
                     bounds_error=True)

    with pytest.raises(ValueError):
        grid.bicubic(collections.OrderedDict(lon=x.flatten(), lat=y.flatten()),
                     bounds_error=True,
                     boundary="sym")

    x_axis = pyinterp.Axis(np.linspace(-180, 179, 360), is_circle=True)
    y_axis = pyinterp.Axis(np.linspace(-90, 90, 181), is_circle=False)
    z_axis = pyinterp.Axis(np.linspace(0, 10, 10), is_circle=False)
    matrix, _ = np.meshgrid(x_axis[:], y_axis[:])
    grid = pyinterp.Grid2D(x_axis, y_axis, matrix.T)

    assert isinstance(grid, pyinterp.Grid2D)
    with pytest.raises(ValueError):
        pyinterp.bicubic(grid, x.flatten(), y.flatten(), fitting_model='_')
    with pytest.raises(ValueError):
        pyinterp.bicubic(grid, x.flatten(), y.flatten(), boundary='_')
    grid = pyinterp.Grid2D(x_axis.flip(inplace=False), y_axis, matrix.T)
    with pytest.raises(ValueError):
        pyinterp.bicubic(grid, x.flatten(), y.flatten())

    grid = pyinterp.Grid2D(x_axis, y_axis.flip(), matrix.T)
    with pytest.raises(ValueError):
        pyinterp.bicubic(grid, x.flatten(), y.flatten())

    matrix, _, _ = np.meshgrid(x_axis[:], y_axis[:], z_axis[:])
    grid = pyinterp.Grid3D(x_axis, y_axis, z_axis, matrix.transpose(1, 0, 2))
    with pytest.raises(ValueError):
        pyinterp.bicubic(grid, x.flatten(), y.flatten())

    grid = pyinterp.backends.xarray.RegularGridInterpolator(
        xr.load_dataset(grid2d_path()).mss)
    assert grid.ndim == 2
    assert isinstance(grid.grid, pyinterp.backends.xarray.Grid2D)
    z = grid(collections.OrderedDict(lon=x.flatten(), lat=y.flatten()),
             method="bicubic",
             bicubic_kwargs=dict(nx=3, ny=3))
    assert isinstance(z, np.ndarray)
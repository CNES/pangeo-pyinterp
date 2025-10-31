# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for 2D interpolation."""
import collections
import pickle

import numpy as np
import pytest
from pytest import Config
import xarray as xr

from . import load_grid2d, make_or_compare_reference
from .. import Axis, Grid2D, Grid3D, bicubic, core
from ..backends import xarray as xr_backend


def test_axis_identifier() -> None:
    """Test AxisIdentifier class."""
    ident = xr_backend.AxisIdentifier(xr.DataArray())
    assert ident.longitude() is None
    assert ident.latitude() is None


def test_dims_from_data_array() -> None:
    """Test dimension extraction from DataArray."""
    array = xr.DataArray()
    with pytest.raises(ValueError):
        xr_backend._dims_from_data_array(array, True, 1)
    array = xr.DataArray(data=np.zeros((2, 2), dtype='float64'))
    with pytest.raises(ValueError):
        xr_backend._dims_from_data_array(array, True, 2)
    array = xr.DataArray(data=np.zeros((2, 2), dtype='float64'),
                         coords=[('lon', np.arange(2), {
                             'units': 'degrees_east'
                         }), ('lat', np.arange(2))],
                         dims=['lon', 'lat'])
    with pytest.raises(ValueError):
        xr_backend._dims_from_data_array(array, True, 2)


def test_bivariate(pytestconfig: Config) -> None:
    """Test Grid2D with bivariate interpolation."""
    dump = pytestconfig.getoption('dump')
    grid: xr_backend.Grid2D | Grid2D | xr_backend.RegularGridInterpolator
    grid = xr_backend.Grid2D(load_grid2d().mss)

    assert isinstance(grid, xr_backend.Grid2D)
    assert isinstance(grid, Grid2D)
    other = pickle.loads(pickle.dumps(grid))
    assert isinstance(other, xr_backend.Grid2D)
    assert isinstance(grid, Grid2D)

    assert isinstance(grid.x, Axis)
    assert isinstance(grid.y, Axis)
    assert isinstance(grid.array, np.ndarray)

    x, y = np.meshgrid(
        np.arange(-180, 180, 1) + 1 / 3.0,
        np.arange(-90, 90, 1) + 1 / 3.0,
        indexing='ij',
    )

    z = grid.bivariate(collections.OrderedDict(lon=x.ravel(), lat=y.ravel()))
    assert isinstance(z, np.ndarray)
    make_or_compare_reference('bivariate_bilinear.npy', z, dump)

    z = grid.bivariate(collections.OrderedDict(lon=x.ravel(), lat=y.ravel()),
                       interpolator='nearest')
    assert isinstance(z, np.ndarray)
    make_or_compare_reference('bivariate_nearest.npy', z, dump)

    z = grid.bivariate(collections.OrderedDict(lon=x.ravel(), lat=y.ravel()),
                       interpolator='inverse_distance_weighting')
    assert isinstance(z, np.ndarray)
    make_or_compare_reference('idw.npy', z, dump)

    grid = xr_backend.Grid2D(load_grid2d().mss, geodetic=False)

    assert isinstance(grid, xr_backend.Grid2D)
    w = grid.bivariate(collections.OrderedDict(lon=x.ravel(), lat=y.ravel()),
                       interpolator='inverse_distance_weighting')
    assert np.ma.fix_invalid(z).mean() != np.ma.fix_invalid(w).mean()

    with pytest.raises(TypeError):
        grid.bivariate((x.ravel(), y.ravel()))  # type: ignore[arg-type]

    with pytest.raises(IndexError):
        grid.bivariate(
            collections.OrderedDict(lon=x.ravel(),
                                    lat=y.ravel(),
                                    time=np.arange(3)))

    with pytest.raises(IndexError):
        grid.bivariate(
            collections.OrderedDict(longitude=x.ravel(), lat=y.ravel()))

    with pytest.raises(ValueError):
        grid.bivariate(collections.OrderedDict(lon=x.ravel(), lat=y.ravel()),
                       bounds_error=True)

    lon = Axis(  # type: ignore[assignment]
        np.linspace(0, 360, 100, dtype=np.float64),
        is_circle=True,
    )
    lat = Axis(  # type: ignore[assignment]
        np.linspace(-80, 80, 50, dtype=np.float64),
        is_circle=False,
    )
    array, _ = np.meshgrid(lon[:], lat[:])

    with pytest.raises(ValueError):
        Grid2D(lon, lat, array)

    grid = Grid2D(lon, lat, array.T)

    assert isinstance(grid, Grid2D)
    assert isinstance(str(grid), str)

    with pytest.raises(ValueError):
        Grid2D(lon, lat, array, increasing_axes='_')

    grid = xr_backend.RegularGridInterpolator(  # type: ignore[assignment]
        load_grid2d().mss)
    z = grid(collections.OrderedDict(lon=x.ravel(), lat=y.ravel()),
             method='bilinear')
    assert isinstance(z, np.ndarray)
    z = grid(collections.OrderedDict(lon=x.ravel(), lat=y.ravel()),
             method='inverse_distance_weighting',
             p=1)
    assert isinstance(z, np.ndarray)

    with pytest.raises(TypeError):
        z = grid(collections.OrderedDict(lon=x.ravel(), lat=y.ravel()),
                 method='nearest',
                 p=1)

    # This is necessary in order for Dask to scatter the callable instances.
    other = pickle.loads(pickle.dumps(grid, protocol=0))
    assert isinstance(other, xr_backend.RegularGridInterpolator)


def test_bicubic(pytestconfig: Config) -> None:
    """Test Grid2D with bicubic interpolation."""
    dump = pytestconfig.getoption('dump')
    measure_coverage = pytestconfig.getoption('measure_coverage')
    step = 10 if measure_coverage else 1
    grid: (Grid2D | Grid3D | xr_backend.Grid2D
           | xr_backend.RegularGridInterpolator)
    grid = xr_backend.Grid2D(load_grid2d().mss)

    lon = np.arange(-180, 180, step) + 1 / 3
    lat = np.arange(-90, 90, step) + 1 / 3
    x, y = np.meshgrid(lon, lat, indexing='ij')

    z = grid.bicubic(collections.OrderedDict(lon=x.ravel(), lat=y.ravel()))
    assert isinstance(z, np.ndarray)

    for fitting_model in [
            'linear', 'bicubic', 'polynomial', 'c_spline',
            'c_spline_not_a_knot', 'c_spline_periodic', 'akima',
            'akima_periodic', 'steffen'
    ]:
        other = grid.bicubic(collections.OrderedDict(lon=x.ravel(),
                                                     lat=y.ravel()),
                             fitting_model=fitting_model)
        make_or_compare_reference(f'bicubic_{fitting_model}.npy', other, dump)
        assert (z - other).mean() != 0

    with pytest.raises(ValueError):
        grid.bicubic(collections.OrderedDict(lon=x.ravel(), lat=y.ravel()),
                     bounds_error=True)

    with pytest.raises(ValueError):
        grid.bicubic(collections.OrderedDict(lon=x.ravel(), lat=y.ravel()),
                     bounds_error=True,
                     boundary='sym')

    x_axis = Axis(np.linspace(-180, 179, 360), is_circle=True)
    y_axis = Axis(np.linspace(-90, 90, 181), is_circle=False)
    z_axis = Axis(np.linspace(0, 10, 10), is_circle=False)
    matrix, _ = np.meshgrid(x_axis[:], y_axis[:])
    grid = Grid2D(x_axis, y_axis, matrix.T)

    assert isinstance(grid, Grid2D)
    with pytest.raises(ValueError):
        bicubic(grid, x.ravel(), y.ravel(), fitting_model='_')
    with pytest.raises(ValueError):
        bicubic(grid, x.ravel(), y.ravel(), boundary='_')
    grid = Grid2D(x_axis.flip(inplace=False), y_axis, matrix.T)
    with pytest.raises(ValueError):
        bicubic(grid, x.ravel(), y.ravel())

    grid = Grid2D(x_axis, y_axis.flip(), matrix.T)
    with pytest.raises(ValueError):
        bicubic(grid, x.ravel(), y.ravel())

    matrix, _, _ = np.meshgrid(x_axis[:], y_axis[:], z_axis[:])
    grid = Grid3D(x_axis, y_axis, z_axis, matrix.transpose(1, 0, 2))
    with pytest.raises(ValueError):
        bicubic(grid, x.ravel(), y.ravel())

    grid = xr_backend.RegularGridInterpolator(load_grid2d().mss)
    assert grid.ndim == 2
    assert isinstance(grid.grid, xr_backend.Grid2D)
    z = grid(collections.OrderedDict(lon=x.ravel(), lat=y.ravel()),
             method='bicubic',
             bicubic_kwargs={
                 'nx': 3,
                 'ny': 3
             })
    assert isinstance(z, np.ndarray)


def test_grid_2d_int8(pytestconfig: Config) -> None:
    """Test Grid2D with int8 data type."""
    dump = pytestconfig.getoption('dump')
    measure_coverage = pytestconfig.getoption('measure_coverage')
    step = 10 if measure_coverage else 1

    grid = load_grid2d().mss
    grid.values[~np.isnan(grid.values)] = 0
    grid.values[np.isnan(grid.values)] = 1
    grid = grid.astype(np.int8)

    interpolator = xr_backend.RegularGridInterpolator(grid)
    assert isinstance(interpolator.grid._instance, core.Grid2DInt8)

    lon = np.arange(-180, 180, step) + 1 / 3
    lat = np.arange(-90, 90, step) + 1 / 3
    x, y = np.meshgrid(lon, lat, indexing='ij')

    z = interpolator(collections.OrderedDict(lon=x.ravel(), lat=y.ravel()),
                     method='nearest')
    make_or_compare_reference('nearest_int8.npy', z, dump)
    assert np.mean(z) != 0


def test_grid_2d_uint8(pytestconfig: Config) -> None:
    """Test Grid2D with uint8 data type."""
    dump = pytestconfig.getoption('dump')

    grid = load_grid2d().mss
    grid.values[~np.isnan(grid.values)] = 0
    grid.values[np.isnan(grid.values)] = 1
    grid = grid.astype(np.uint8)

    interpolator = xr_backend.RegularGridInterpolator(grid)
    assert isinstance(interpolator.grid._instance, core.Grid2DUInt8)

    lon = np.arange(-180, 180, 1) + 1 / 3.0
    lat = np.arange(-90, 90, 1) + 1 / 3.0
    x, y = np.meshgrid(lon, lat, indexing='ij')

    z = interpolator(collections.OrderedDict(lon=x.ravel(), lat=y.ravel()),
                     method='nearest')
    make_or_compare_reference('nearest_uint8.npy', z, dump)
    assert np.mean(z) != 0

# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for trivariate interpolation."""
from __future__ import annotations

from typing import TYPE_CHECKING
import os
import pickle

import pytest

try:
    import matplotlib.colors
    import matplotlib.pyplot
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False
import numpy as np

from .. import load_grid3d, make_or_compare_reference
from ... import core

if TYPE_CHECKING:
    from pytest import Config

    from ...typing import NDArray1DFloat64, NDArray2DFloat64


def plot(x: NDArray1DFloat64, y: NDArray1DFloat64, z: NDArray2DFloat64,
         filename: str) -> None:
    """Plot the interpolated data."""
    figure = matplotlib.pyplot.figure(figsize=(15, 15), dpi=150)
    value = z.mean()
    std = z.std()
    normalize = matplotlib.colors.Normalize(vmin=value - 3 * std,
                                            vmax=value + 3 * std)
    axe = figure.add_subplot(2, 1, 1)
    axe.pcolormesh(x, y, z, cmap='jet', norm=normalize, shading='auto')
    figure.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                filename),
                   bbox_inches='tight',
                   pad_inches=0.4)


def load_data(
    temporal_axis: bool = False
) -> core.TemporalGrid3DFloat64 | core.Grid3DFloat64:
    """Load test data."""
    ds = load_grid3d()
    z = np.ma.filled(np.flip(ds['tcw'].values.T, axis=1), np.nan)
    if temporal_axis:
        z_axis = core.TemporalAxis(ds['time'].values.astype('M8[h]'))
        return core.TemporalGrid3DFloat64(
            core.Axis(ds['longitude'].values, is_circle=True),
            core.Axis(np.flip(ds['latitude'].values)), z_axis, z)
    int64 = (ds['time'].values -
             np.datetime64('1900-01-01')).astype('m8[h]').astype('int64')
    return core.Grid3DFloat64(
        core.Axis(ds['longitude'].values, is_circle=True),
        core.Axis(np.flip(ds.variables['latitude'][:])), core.Axis(int64), z)


def test_grid3d_accessors() -> None:
    """Test construction and accessors of the object."""
    grid = load_data()
    assert isinstance(grid.x, core.Axis)
    assert isinstance(grid.y, core.Axis)
    assert isinstance(grid.z, core.Axis)
    assert isinstance(grid.array, np.ndarray)


def test_grid3d_pickle() -> None:
    """Serialization test."""
    grid = load_data()
    other = pickle.loads(pickle.dumps(grid))
    assert grid.x == other.x
    assert grid.y == other.y
    assert grid.z == other.z
    assert np.all(
        np.ma.fix_invalid(grid.array) == np.ma.fix_invalid(other.array))


def run_interpolator(
    step: float,
    interpolator: core.BivariateInterpolator3D,
    filename: str,
    visualize: bool,
    dump: bool,
) -> NDArray2DFloat64:
    """Test trivariate interpolation."""
    grid = load_data()
    assert isinstance(grid, core.Grid3DFloat64)
    lon = np.arange(-180, 180, step) + 1 / 3
    lat = np.arange(-90, 90, step) + 1 / 3
    time = 898500 + 3
    x, y, t = np.meshgrid(lon, lat, time, indexing='ij')
    z0 = core.trivariate_float64(grid,
                                 x.ravel(),
                                 y.ravel(),
                                 t.ravel(),
                                 interpolator,
                                 num_threads=0)
    z1 = core.trivariate_float64(grid,
                                 x.ravel(),
                                 y.ravel(),
                                 t.ravel(),
                                 interpolator,
                                 num_threads=1)
    make_or_compare_reference(filename + '.npy', z1, dump)
    shape = (len(lon), len(lat))
    z0 = np.ma.fix_invalid(z0)
    z1 = np.ma.fix_invalid(z1)
    assert np.all(z1 == z0)
    if HAVE_PLT and visualize:
        plot(x.reshape(shape), y.reshape(shape), z0.reshape(shape), filename)
    return z0


def test_trivariate_spline(pytestconfig: Config) -> None:
    """Testing of the spline interpolation."""
    measure_coverage = pytestconfig.getoption('measure_coverage')
    step = 10 if measure_coverage else 1 / 3
    grid = load_data()
    lon = np.arange(-180, 180, step) + 1 / 3
    lat = np.arange(-80, 80, step) + 1 / 3
    time = 898524 + 3
    x, y, t = np.meshgrid(lon, lat, time, indexing='ij')
    z0 = core.spline_float64(grid,
                             x.ravel(),
                             y.ravel(),
                             t.ravel(),
                             fitting_model='akima',
                             bounds_error=True,
                             num_threads=0)
    z1 = core.spline_float64(grid,
                             x.ravel(),
                             y.ravel(),
                             t.ravel(),
                             fitting_model='akima',
                             bounds_error=True,
                             num_threads=1)
    make_or_compare_reference('test_trivariate_spline.npy', z1,
                              pytestconfig.getoption('dump'))
    shape = (len(lon), len(lat))
    z0 = np.ma.fix_invalid(z0)
    z1 = np.ma.fix_invalid(z1)
    assert np.all(z1 == z0)
    if HAVE_PLT and pytestconfig.getoption('visualize'):
        plot(x.reshape(shape), y.reshape(shape), z0.reshape(shape),
             'tcw_spline.png')


def test_grid3d_bounds_error(pytestconfig: Config) -> None:
    """Test of the detection on interpolation outside bounds."""
    measure_coverage = pytestconfig.getoption('measure_coverage')
    step = 10 if measure_coverage else 1 / 3
    grid = load_data()
    assert isinstance(grid, core.Grid3DFloat64)
    interpolator = core.Bilinear3D()
    lon = np.arange(-180, 180, step) + 1 / 3
    lat = np.arange(-90, 90 + 1, step) + 1 / 3
    time = 898500 + 3
    x, y, t = np.meshgrid(lon, lat, time, indexing='ij')
    core.trivariate_float64(grid,
                            x.ravel(),
                            y.ravel(),
                            t.ravel(),
                            interpolator,
                            num_threads=0)
    with pytest.raises(ValueError):
        core.trivariate_float64(grid,
                                x.ravel(),
                                y.ravel(),
                                t.ravel(),
                                interpolator,
                                bounds_error=True,
                                num_threads=0)


def test_grid3d_z_method(pytestconfig: Config) -> None:
    """Test of the interpolation method used on Z-axis."""
    dump = pytestconfig.getoption('dump')
    measure_coverage = pytestconfig.getoption('measure_coverage')
    step = 10 if measure_coverage else 1 / 3
    grid = load_data(temporal_axis=True)
    assert isinstance(grid, core.TemporalGrid3DFloat64)
    interpolator = core.TemporalBilinear3D()
    lon = np.arange(-180, 180, step) + 1 / 3
    lat = np.arange(-90, 90 + 1, step) + 1 / 3
    time = np.array(['2002-07-02T15'], dtype='datetime64[h]').astype('int64')
    x, y, t = np.meshgrid(lon, lat, time, indexing='ij')
    z0 = core.trivariate_float64(grid,
                                 x.ravel(),
                                 y.ravel(),
                                 t.ravel(),
                                 interpolator,
                                 num_threads=0)
    z1 = core.trivariate_float64(grid,
                                 x.ravel(),
                                 y.ravel(),
                                 t.ravel(),
                                 interpolator,
                                 z_method='linear',
                                 num_threads=0)
    make_or_compare_reference('test_grid3d_z_method_linear.npy', z1, dump)
    z0 = np.ma.fix_invalid(z0)
    z1 = np.ma.fix_invalid(z1)
    assert np.all(z0 == z1)
    z1 = core.trivariate_float64(grid,
                                 x.ravel(),
                                 y.ravel(),
                                 t.ravel(),
                                 interpolator,
                                 z_method='nearest',
                                 num_threads=0)
    make_or_compare_reference('test_grid3d_z_method_nearest.npy', z1, dump)
    z1 = np.ma.fix_invalid(z1)
    assert np.all(z0 != z1)
    with pytest.raises(ValueError):
        core.trivariate_float64(grid,
                                x.ravel(),
                                y.ravel(),
                                t.ravel(),
                                interpolator,
                                z_method='NEAREST',
                                num_threads=0)


def test_grid3d_interpolator(pytestconfig: Config) -> None:
    """Testing of different interpolation methods."""
    visualize = pytestconfig.getoption('visualize')
    dump = pytestconfig.getoption('dump')
    measure_coverage = pytestconfig.getoption('measure_coverage')
    step = 10 if measure_coverage else 1 / 3
    a = run_interpolator(step, core.Nearest3D(), 'tcw_trivariate_nearest',
                         visualize, dump)
    b = run_interpolator(step, core.Bilinear3D(), 'tcw_trivariate_bilinear',
                         visualize, dump)
    c = run_interpolator(step, core.InverseDistanceWeighting3D(),
                         'tcw_trivariate_idw', visualize, dump)
    assert (a - b).std() != 0
    assert (a - c).std() != 0
    assert (b - c).std() != 0


def test_invalid_data(pytestconfig: Config) -> None:
    """Testing of the interpolation with invalid data."""
    measure_coverage = pytestconfig.getoption('measure_coverage')
    step = 10 if measure_coverage else 1 / 3
    grid = load_data(temporal_axis=True)
    assert isinstance(grid, core.TemporalGrid3DFloat64)
    interpolator = core.TemporalBilinear3D()
    lon = np.arange(-180, 180, step) + 1 / 3
    lat = np.arange(-90, 90 + 1, step) + 1 / 3
    time = np.array(['2002-07-02T15'], dtype='datetime64[h]').astype('int64')
    x, y, t = np.meshgrid(lon, lat, time, indexing='ij')
    z0 = core.trivariate_float64(grid,
                                 x.ravel(),
                                 y.ravel(),
                                 t.ravel(),
                                 interpolator,
                                 num_threads=0)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(x), size=100)
    x.ravel()[indices] = np.nan
    y.ravel()[indices] = np.nan
    t.ravel()[indices] = np.datetime64('NaT')
    z1 = core.trivariate_float64(grid,
                                 x.ravel(),
                                 y.ravel(),
                                 t.ravel(),
                                 interpolator,
                                 num_threads=0)
    mask = ~np.isnan(x.ravel())
    assert np.allclose(z1[mask], z0[mask], equal_nan=True)
    assert np.all(np.isnan(z1[~mask]))

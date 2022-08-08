# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
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


def plot(x, y, z, filename):
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


def load_data(temporal_axis=False):
    ds = load_grid3d()
    z = np.flip(ds['tcw'].values.T, axis=1)
    if temporal_axis:
        z_axis = core.TemporalAxis(ds['time'].values.astype('M8[h]'))
        return core.TemporalGrid3DFloat64(
            core.Axis(ds['longitude'].values, is_circle=True),
            core.Axis(np.flip(ds['latitude'].values)), z_axis, z.data)
    int64 = (ds['time'].values -
             np.datetime64('1900-01-01')).astype('m8[h]').astype('int64')
    return core.Grid3DFloat64(
        core.Axis(ds['longitude'].values, is_circle=True),
        core.Axis(np.flip(ds.variables['latitude'][:])), core.Axis(int64),
        z.data)


def test_grid3d_accessors():
    """Test construction and accessors of the object."""
    grid = load_data()
    assert isinstance(grid.x, core.Axis)
    assert isinstance(grid.y, core.Axis)
    assert isinstance(grid.z, core.Axis)
    assert isinstance(grid.array, np.ndarray)


def test_grid3d_pickle():
    """Serialization test."""
    grid = load_data()
    other = pickle.loads(pickle.dumps(grid))
    assert grid.x == other.x
    assert grid.y == other.y
    assert grid.z == other.z
    assert np.all(
        np.ma.fix_invalid(grid.array) == np.ma.fix_invalid(other.array))


def run_interpolator(interpolator, filename, visualize, dump):
    """Testing an interpolation method."""
    grid = load_data()
    lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
    lat = np.arange(-90, 90, 1 / 3.0) + 1 / 3.0
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


def test_trivariate_spline(pytestconfig):
    """Testing of the spline interpolation."""
    grid = load_data()
    lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
    lat = np.arange(-80, 80, 1 / 3.0) + 1 / 3.0
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


def test_grid3d_bounds_error():
    """Test of the detection on interpolation outside bounds."""
    grid = load_data()
    interpolator = core.Bilinear3D()
    lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
    lat = np.arange(-90, 90 + 1, 1 / 3.0) + 1 / 3.0
    time = 898500 + 3
    x, y, t = np.meshgrid(lon, lat, time, indexing='ij')
    core.trivariate_float64(
        grid,  # type: ignore
        x.ravel(),
        y.ravel(),
        t.ravel(),
        interpolator,
        num_threads=0)
    with pytest.raises(ValueError):
        core.trivariate_float64(
            grid,  # type: ignore
            x.ravel(),
            y.ravel(),
            t.ravel(),
            interpolator,
            bounds_error=True,
            num_threads=0)


def test_grid3d_z_method(pytestconfig):
    """Test of the interpolation method used on Z-axis."""
    dump = pytestconfig.getoption('dump')
    grid = load_data(temporal_axis=True)
    interpolator = core.TemporalBilinear3D()
    lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
    lat = np.arange(-90, 90 + 1, 1 / 3.0) + 1 / 3.0
    time = np.array(['2002-07-02T15'], dtype='datetime64[h]').astype('int64')
    x, y, t = np.meshgrid(lon, lat, time, indexing='ij')
    z0 = core.trivariate_float64(
        grid,  # type: ignore
        x.ravel(),
        y.ravel(),
        t.ravel(),
        interpolator,
        num_threads=0)
    z1 = core.trivariate_float64(
        grid,  # type: ignore
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
    z1 = core.trivariate_float64(
        grid,  # type: ignore
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
        core.trivariate_float64(
            grid,  # type: ignore
            x.ravel(),
            y.ravel(),
            t.ravel(),
            interpolator,
            z_method='NEAREST',
            num_threads=0)


def test_grid3d_interpolator(pytestconfig):
    """Testing of different interpolation methods."""
    visualize = pytestconfig.getoption('visualize')
    dump = pytestconfig.getoption('dump')
    a = run_interpolator(core.Nearest3D(), 'tcw_trivariate_nearest', visualize,
                         dump)
    b = run_interpolator(core.Bilinear3D(), 'tcw_trivariate_bilinear',
                         visualize, dump)
    c = run_interpolator(core.InverseDistanceWeighting3D(),
                         'tcw_trivariate_idw', visualize, dump)
    assert (a - b).std() != 0
    assert (a - c).std() != 0
    assert (b - c).std() != 0


def test_invalid_data():
    """Testing of the interpolation with invalid data."""
    grid = load_data(temporal_axis=True)
    interpolator = core.TemporalBilinear3D()
    lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
    lat = np.arange(-90, 90 + 1, 1 / 3.0) + 1 / 3.0
    time = np.array(['2002-07-02T15'], dtype='datetime64[h]').astype('int64')
    x, y, t = np.meshgrid(lon, lat, time, indexing='ij')
    z0 = core.trivariate_float64(
        grid,  # type: ignore
        x.ravel(),
        y.ravel(),
        t.ravel(),
        interpolator,
        num_threads=0)
    indices = np.random.randint(0, len(x), size=100)
    x.ravel()[indices] = np.nan
    y.ravel()[indices] = np.nan
    t.ravel()[indices] = np.datetime64('NaT')
    z1 = core.trivariate_float64(
        grid,  # type: ignore
        x.ravel(),
        y.ravel(),
        t.ravel(),
        interpolator,
        num_threads=0)
    mask = ~np.isnan(x.ravel())
    assert np.allclose(z1[mask], z0[mask], equal_nan=True)
    assert np.all(np.isnan(z1[~mask]))

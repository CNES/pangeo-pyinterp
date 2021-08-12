# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import os
import pickle
import pytest
import netCDF4
try:
    import matplotlib.pyplot
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False
import numpy as np
import pyinterp.core as core
from .. import grid3d_path


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
    with netCDF4.Dataset(grid3d_path()) as ds:
        z = np.flip(ds.variables['tcw'][:].T, axis=1)
        z[z.mask] = float("nan")
        z_axis = core.TemporalAxis(
            netCDF4.num2date(
                ds.variables['time'][:],
                ds.variables['time'].units,
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True).astype("datetime64[h]").astype(
                    "int64")) if temporal_axis else core.Axis(
                        ds.variables['time'][:])
        class_ = core.TemporalGrid3DFloat64 if temporal_axis else core.Grid3DFloat64

        return class_(core.Axis(ds.variables['longitude'][:], is_circle=True),
                      core.Axis(np.flip(ds.variables['latitude'][:])), z_axis,
                      z.data)


def test_grid3d_accessors():
    """Test construction and accessors of the object"""
    grid = load_data()
    assert isinstance(grid.x, core.Axis)
    assert isinstance(grid.y, core.Axis)
    assert isinstance(grid.z, core.Axis)
    assert isinstance(grid.array, np.ndarray)


def test_grid3d_pickle():
    """Serialization test"""
    grid = load_data()
    other = pickle.loads(pickle.dumps(grid))
    assert grid.x == other.x
    assert grid.y == other.y
    assert grid.z == other.z
    assert np.all(
        np.ma.fix_invalid(grid.array) == np.ma.fix_invalid(other.array))


def run_interpolator(interpolator, filename):
    """Testing an interpolation method."""
    grid = load_data()
    lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
    lat = np.arange(-90, 90, 1 / 3.0) + 1 / 3.0
    time = 898500 + 3
    x, y, t = np.meshgrid(lon, lat, time, indexing="ij")
    z0 = core.trivariate_float64(grid,
                                 x.flatten(),
                                 y.flatten(),
                                 t.flatten(),
                                 interpolator,
                                 num_threads=0)
    z1 = core.trivariate_float64(grid,
                                 x.flatten(),
                                 y.flatten(),
                                 t.flatten(),
                                 interpolator,
                                 num_threads=1)
    shape = (len(lon), len(lat))
    z0 = np.ma.fix_invalid(z0)
    z1 = np.ma.fix_invalid(z1)
    assert np.all(z1 == z0)
    if HAVE_PLT:
        plot(x.reshape(shape), y.reshape(shape), z0.reshape(shape), filename)
    return z0


def test_trivariate_spline():
    """Testing of the spline interpolation"""
    grid = load_data()
    lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
    lat = np.arange(-80, 80, 1 / 3.0) + 1 / 3.0
    time = 898524 + 3
    x, y, t = np.meshgrid(lon, lat, time, indexing='ij')
    z0 = core.spline_float64(grid,
                             x.flatten(),
                             y.flatten(),
                             t.flatten(),
                             fitting_model="akima",
                             bounds_error=True,
                             num_threads=0)
    z1 = core.spline_float64(grid,
                             x.flatten(),
                             y.flatten(),
                             t.flatten(),
                             fitting_model="akima",
                             bounds_error=True,
                             num_threads=1)
    shape = (len(lon), len(lat))
    z0 = np.ma.fix_invalid(z0)
    z1 = np.ma.fix_invalid(z1)
    assert np.all(z1 == z0)
    if HAVE_PLT:
        plot(x.reshape(shape), y.reshape(shape), z0.reshape(shape),
             "tcw_spline.png")


def test_grid3d_bounds_error():
    """Test of the detection on interpolation outside bounds"""
    grid = load_data()
    interpolator = core.Bilinear3D()
    lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
    lat = np.arange(-90, 90 + 1, 1 / 3.0) + 1 / 3.0
    time = 898500 + 3
    x, y, t = np.meshgrid(lon, lat, time, indexing="ij")
    core.trivariate_float64(grid,
                            x.flatten(),
                            y.flatten(),
                            t.flatten(),
                            interpolator,
                            num_threads=0)
    with pytest.raises(ValueError):
        core.trivariate_float64(grid,
                                x.flatten(),
                                y.flatten(),
                                t.flatten(),
                                interpolator,
                                bounds_error=True,
                                num_threads=0)


def test_grid3d_z_method():
    """Test of the interpolation method used on Z-axis"""
    grid = load_data(temporal_axis=True)
    interpolator = core.TemporalBilinear3D()
    lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
    lat = np.arange(-90, 90 + 1, 1 / 3.0) + 1 / 3.0
    time = np.array([
        netCDF4.num2date(898500 + 3,
                         "hours since 1900-01-01 00:00:0.0",
                         only_use_cftime_datetimes=False,
                         only_use_python_datetimes=True)
    ]).astype("datetime64[h]").astype("int64")
    x, y, t = np.meshgrid(lon, lat, time, indexing="ij")
    z0 = core.trivariate_float64(grid,
                                 x.flatten(),
                                 y.flatten(),
                                 t.flatten(),
                                 interpolator,
                                 num_threads=0)
    z1 = core.trivariate_float64(grid,
                                 x.flatten(),
                                 y.flatten(),
                                 t.flatten(),
                                 interpolator,
                                 z_method="linear",
                                 num_threads=0)
    z0 = np.ma.fix_invalid(z0)
    z1 = np.ma.fix_invalid(z1)
    assert np.all(z0 == z1)
    z1 = core.trivariate_float64(grid,
                                 x.flatten(),
                                 y.flatten(),
                                 t.flatten(),
                                 interpolator,
                                 z_method="nearest",
                                 num_threads=0)
    z1 = np.ma.fix_invalid(z1)
    assert np.all(z0 != z1)
    with pytest.raises(ValueError):
        core.trivariate_float64(grid,
                                x.flatten(),
                                y.flatten(),
                                t.flatten(),
                                interpolator,
                                z_method="NEAREST",
                                num_threads=0)


def test_grid3d_interpolator():
    """Testing of different interpolation methods"""
    a = run_interpolator(core.Nearest3D(), "tcw_trivariate_nearest")
    b = run_interpolator(core.Bilinear3D(), "tcw_trivariate_bilinear")
    c = run_interpolator(core.InverseDistanceWeighting3D(),
                         "tcw_trivariate_idw")
    assert (a - b).std() != 0
    assert (a - c).std() != 0
    assert (b - c).std() != 0

# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import dask.array as da
import pytest

try:
    import dask.array.stats as das
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
import numpy as np

from . import load_grid2d
from .. import Axis, Binning1D, Binning2D, geodetic


def build_binning2d_instance(dtype):
    ds = load_grid2d()

    x_axis = Axis(np.arange(-180, 180, 5), is_circle=True)
    y_axis = Axis(np.arange(-90, 95, 5))
    binning = Binning2D(x_axis, y_axis, geodetic.Spheroid(), dtype=dtype)
    assert x_axis == binning.x
    assert y_axis == binning.y
    assert isinstance(str(binning), str)

    lon, lat = np.meshgrid(ds.lon, ds.lat)
    binning.push(lon, lat, ds.mss, simple=True)
    simple_mean = binning.variable('mean')
    assert isinstance(simple_mean, np.ndarray)

    binning.clear()
    binning.push(lon, lat, ds.mss, simple=False)
    linear_mean = binning.variable('mean')
    assert isinstance(simple_mean, np.ndarray)
    assert np.any(linear_mean != simple_mean)
    assert isinstance(binning.variable('sum'), np.ndarray)
    assert isinstance(binning.variable('count'), np.ndarray)

    with pytest.raises(ValueError):
        binning.variable('_')


def test_binning2d():
    build_binning2d_instance(np.float64)
    build_binning2d_instance(np.float32), Binning2D

    with pytest.raises(ValueError):
        build_binning2d_instance(np.int8)


def test_binning2d_dask():
    x_axis = Axis(np.linspace(-180, 180, 1), is_circle=True)
    y_axis = Axis(np.linspace(-80, 80, 1))
    binning = Binning2D(x_axis, y_axis)

    x = da.full((4096 * 8, ), -180.0, dtype='f8', chunks=4096)
    y = da.full((4096 * 8, ), -80.0, dtype='f8', chunks=4096)
    z = da.random.uniform(size=4096 * 8, chunks=4096)

    binning = binning.push_delayed(x, y, z).compute()

    assert np.all(binning.variable('count') == 32768)
    assert binning.variable('sum_of_weights')[0, 0] == 32768
    assert binning.variable('mean')[0, 0] == pytest.approx(z.mean().compute())
    assert binning.variable('variance')[0, 0] == pytest.approx(
        z.std().compute()**2)  # type: ignore
    assert binning.variable('sum')[0, 0] == pytest.approx(z.sum().compute())
    if HAVE_SCIPY:
        assert binning.variable('kurtosis')[0, 0] == pytest.approx(
            das.kurtosis(z).compute())  # type: ignore
        assert binning.variable('skewness')[0, 0] == pytest.approx(
            das.skew(z).compute())  # type: ignore


def build_binning1d_instance(dtype):
    ds = load_grid2d()

    x_axis = Axis(np.arange(-180, 180, 5))
    binning = Binning1D(x_axis, dtype=dtype)
    assert x_axis == binning.x
    with pytest.raises(AttributeError):
        binning.y
    with pytest.raises(AttributeError):
        binning.wgs
    assert isinstance(str(binning), str)

    lon, lat = np.meshgrid(ds.lon, ds.lat)
    binning.push(lon, ds.mss)
    mean = binning.variable('mean')
    assert isinstance(mean, np.ndarray)
    assert isinstance(binning.variable('sum'), np.ndarray)
    assert isinstance(binning.variable('count'), np.ndarray)

    with pytest.raises(ValueError):
        binning.variable('_')


def test_binning1d():
    build_binning1d_instance(np.float64)
    build_binning1d_instance(np.float32), Binning2D

    with pytest.raises(ValueError):
        build_binning1d_instance(np.int8)


def test_binning1d_dask():
    x_axis = Axis(np.linspace(-180, 180, 1), is_circle=True)
    binning = Binning1D(x_axis)

    x = da.full((4096 * 8, ), -180.0, dtype='f8', chunks=4096)
    z = da.random.uniform(size=4096 * 8, chunks=4096)

    binning = binning.push_delayed(x, z).compute()

    assert np.all(binning.variable('count') == 32768)
    assert binning.variable('sum_of_weights')[0, 0] == 32768
    assert binning.variable('mean')[0, 0] == pytest.approx(z.mean().compute())
    assert binning.variable('variance')[0, 0] == pytest.approx(
        z.std().compute()**2)  # type: ignore
    assert binning.variable('sum')[0, 0] == pytest.approx(z.sum().compute())
    if HAVE_SCIPY:
        assert binning.variable('kurtosis')[0, 0] == pytest.approx(
            das.kurtosis(z).compute())  # type: ignore
        assert binning.variable('skewness')[0, 0] == pytest.approx(
            das.skew(z).compute())  # type: ignore

# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for binning classes."""
import dask.array as da
import pytest

try:
    import dask.array.stats as das
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
import numpy as np
from pytest import Config

from . import load_grid2d
from .. import Axis, Binning1D, Binning2D, geodetic


def build_binning2d_instance(step: int, dtype: np.dtype) -> None:
    """Build a Binning2D instance and test its methods."""
    ds = load_grid2d()

    x_axis = Axis(np.arange(-180, 180, step, dtype=np.float64), is_circle=True)
    y_axis = Axis(np.arange(-90, 95, step, dtype=np.float64))
    binning = Binning2D(x_axis, y_axis, geodetic.Spheroid(), dtype=dtype)
    assert x_axis == binning.x
    assert y_axis == binning.y
    assert isinstance(str(binning), str)

    lon, lat = np.meshgrid(ds.lon, ds.lat)
    mss = ds.mss
    if step != 1:
        # Reduce the dataset size to measure test coverage.
        lon = lon[::10, ::10]
        lat = lat[::10, ::10]
        mss = mss[::10, ::10]
    binning.push(lon, lat, mss, simple=True)
    simple_mean = binning.variable('mean')
    assert isinstance(simple_mean, np.ndarray)

    binning.clear()
    binning.push(lon, lat, mss, simple=False)
    linear_mean = binning.variable('mean')
    assert isinstance(simple_mean, np.ndarray)
    assert np.any(linear_mean != simple_mean)
    assert isinstance(binning.variable('sum'), np.ndarray)
    assert isinstance(binning.variable('count'), np.ndarray)

    with pytest.raises(ValueError):
        binning.variable('_')


def test_binning2d(pytestconfig: Config) -> None:
    """Test Binning2D class."""
    measure_coverage = pytestconfig.getoption('measure_coverage')
    step = 10 if measure_coverage else 1
    build_binning2d_instance(step, np.dtype(np.float64))
    build_binning2d_instance(step, np.dtype(np.float32))

    with pytest.raises(ValueError):
        build_binning2d_instance(step, np.dtype(np.int8))


def test_binning2d_dask(pytestconfig: Config) -> None:
    """Test Binning2D with Dask arrays."""
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
        z.std().compute()**2)
    assert binning.variable('sum')[0, 0] == pytest.approx(z.sum().compute())
    if HAVE_SCIPY:
        assert binning.variable('kurtosis')[0, 0] == pytest.approx(
            das.kurtosis(z).compute())
        assert binning.variable('skewness')[0, 0] == pytest.approx(
            das.skew(z).compute())


def build_binning1d_instance(dtype: np.dtype) -> None:
    """Build a Binning1D instance and test its methods."""
    ds = load_grid2d()

    x_axis = Axis(np.arange(-180, 180, 5, dtype=np.float64))
    binning = Binning1D(x_axis, dtype=dtype)
    assert x_axis == binning.x
    with pytest.raises(AttributeError):
        _ = binning.y  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = binning.wgs  # type: ignore[attr-defined]
    assert isinstance(str(binning), str)

    lon, _lat = np.meshgrid(ds.lon, ds.lat)
    binning.push(lon, ds.mss)
    mean = binning.variable('mean')
    assert isinstance(mean, np.ndarray)
    assert isinstance(binning.variable('sum'), np.ndarray)
    assert isinstance(binning.variable('count'), np.ndarray)

    with pytest.raises(ValueError):
        binning.variable('_')


def test_binning1d() -> None:
    """Test Binning1D class."""
    build_binning1d_instance(np.dtype(np.float64))
    build_binning1d_instance(np.dtype(np.float32))

    with pytest.raises(ValueError):
        build_binning1d_instance(np.dtype(np.int8))


def test_binning1d_dask() -> None:
    """Test Binning1D with Dask arrays."""
    x_axis = Axis(np.linspace(-180, 180, 1), is_circle=True)
    binning = Binning1D(x_axis)

    x = da.full((4096 * 8, ), -180.0, dtype='f8', chunks=4096)
    z = da.random.uniform(size=4096 * 8, chunks=4096)

    binning = binning.push_delayed(x, z).compute()

    assert np.all(binning.variable('count') == 32768)
    assert binning.variable('sum_of_weights')[0, 0] == 32768
    assert binning.variable('mean')[0, 0] == pytest.approx(z.mean().compute())
    assert binning.variable('variance')[0, 0] == pytest.approx(
        z.std().compute()**2)
    assert binning.variable('sum')[0, 0] == pytest.approx(z.sum().compute())
    if HAVE_SCIPY:
        assert binning.variable('kurtosis')[0, 0] == pytest.approx(
            das.kurtosis(z).compute())
        assert binning.variable('skewness')[0, 0] == pytest.approx(
            das.skew(z).compute())

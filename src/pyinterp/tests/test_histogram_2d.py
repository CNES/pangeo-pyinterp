# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import dask.array as da
import numpy as np
import pytest

from . import load_grid2d
from .. import Axis, Histogram2D


def build_instance(dtype):
    """Build an instance of Histogram2D with a given dtype."""
    ds = load_grid2d()

    x_axis = Axis(np.arange(-180, 180, 5), is_circle=True)
    y_axis = Axis(np.arange(-90, 95, 5))
    hist2d = Histogram2D(x_axis, y_axis, bin_counts=40, dtype=dtype)
    assert x_axis == hist2d.x
    assert y_axis == hist2d.y
    assert isinstance(str(hist2d), str)

    lon, lat = np.meshgrid(ds.lon, ds.lat)
    hist2d.push(lon, lat, ds.mss)
    mean = hist2d.variable('mean')
    assert isinstance(mean, np.ndarray)
    median = hist2d.variable('quantile', 0.5)
    assert isinstance(median, np.ndarray)
    kurtosis = hist2d.variable('kurtosis')
    assert isinstance(kurtosis, np.ndarray)
    skewness = hist2d.variable('skewness')
    assert isinstance(skewness, np.ndarray)

    hist2d.clear()
    assert np.all(hist2d.variable('count') == 0)

    with pytest.raises(ValueError):
        hist2d.variable('_')


def test_histogram2d():
    """Test Histogram2D class."""
    build_instance(np.float64)
    build_instance(np.float32)

    with pytest.raises(ValueError):
        build_instance(np.int8)


def test_dask():
    """Test Histogram2D with dask arrays."""
    x_axis = Axis(np.linspace(-180, 180, 1), is_circle=True)
    y_axis = Axis(np.linspace(-80, 80, 1))
    hist2d = Histogram2D(x_axis, y_axis)

    x = da.full((4096 * 8, ), -180.0, dtype='f8', chunks=4096)
    y = da.full((4096 * 8, ), -80.0, dtype='f8', chunks=4096)
    z = da.random.uniform(size=4096 * 8, chunks=4096)

    hist2d = hist2d.push_delayed(x, y, z).compute()

    assert np.all(hist2d.variable('count') == 32768)
    assert hist2d.variable('mean')[0, 0] == pytest.approx(z.mean().compute())
    assert hist2d.variable('variance')[0, 0] == pytest.approx(
        z.std().compute()**2, rel=1e-4, abs=1e-4)  # type: ignore

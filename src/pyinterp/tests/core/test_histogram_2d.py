# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import copy
import os
import pickle

try:
    import matplotlib.colors
    import matplotlib.pyplot
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False
import numpy as np

from .. import load_grid2d
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


def load_data():
    ds = load_grid2d()
    return ds['lon'].values, ds['lat'].values, ds['mss'].values.T


def test_histogram2d_constructor():
    x_axis = core.Axis(np.linspace(-180, 180, 10), is_circle=True)
    y_axis = core.Axis(np.linspace(-90, 90, 10))

    hist2d = core.Histogram2DFloat64(x_axis, y_axis)
    assert isinstance(hist2d.x, core.Axis)
    assert isinstance(hist2d.y, core.Axis)

    # The class must return a reference on the axes provided during
    # construction
    assert id(x_axis) == id(hist2d.x)
    assert id(y_axis) == id(hist2d.y)

    hist2d.clear()
    count = hist2d.count()
    assert isinstance(count, np.ndarray)
    assert count.size == len(x_axis) * len(y_axis)
    assert count.mean() == 0


def test_binning2d_methods(pytestconfig):
    x_axis = core.Axis(np.linspace(-180, 180, 361 // 4), is_circle=True)
    y_axis = core.Axis(np.linspace(-90, 90, 180 // 4))

    hist2d = core.Histogram2DFloat64(x_axis, y_axis, None)
    x, y, z = load_data()
    mx, my = np.meshgrid(x, y, indexing='ij')
    hist2d.push(mx.ravel(), my.ravel(), z.ravel())

    count = hist2d.count()
    assert count.max() != 0
    simple_mean = np.ma.fix_invalid(hist2d.mean())
    if HAVE_PLT and pytestconfig.getoption('visualize'):
        mx, my = np.meshgrid(x_axis[:], y_axis[:], indexing='ij')
        plot(mx, my, simple_mean, 'hist2d_mean.png')

    # Test of access to statistical variables
    assert isinstance(hist2d.max(), np.ndarray)
    assert isinstance(hist2d.min(), np.ndarray)
    assert isinstance(hist2d.variance(), np.ndarray)
    assert isinstance(hist2d.skewness(), np.ndarray)
    assert isinstance(hist2d.kurtosis(), np.ndarray)
    median = np.ma.fix_invalid(hist2d.quantile())
    assert isinstance(median, np.ndarray)
    if HAVE_PLT and pytestconfig.getoption('visualize'):
        mx, my = np.meshgrid(x_axis[:], y_axis[:], indexing='ij')
        plot(mx, my, median, 'hist2d_median.png')

    histograms = hist2d.histograms()
    assert isinstance(histograms, np.ndarray)
    assert histograms.ndim == 3
    assert histograms.dtype == np.dtype([('value', 'f8'), ('weight', 'f8')])


def test_binning2d_pickle():
    x_axis = core.Axis(np.linspace(-180, 180, 1), is_circle=True)
    y_axis = core.Axis(np.linspace(-90, 90, 1))

    hist2d = core.Histogram2DFloat64(x_axis, y_axis, None)
    hist2d.push([-180], [-90], [np.pi])  # type: ignore

    assert np.all(hist2d.count() == 1)
    assert np.all(hist2d.mean() == np.pi)
    assert np.all(hist2d.min() == np.pi)
    assert np.all(hist2d.max() == np.pi)
    assert np.all(hist2d.variance() == 0)
    assert np.all(np.isnan(hist2d.skewness()))
    assert np.all(np.isnan(hist2d.kurtosis()))

    other = pickle.loads(pickle.dumps(hist2d))

    assert np.all(other.count() == 1)
    assert np.all(other.mean() == np.pi)
    assert np.all(other.min() == np.pi)
    assert np.all(other.max() == np.pi)
    assert np.all(other.variance() == 0)
    assert np.all(np.isnan(hist2d.skewness()))
    assert np.all(np.isnan(hist2d.kurtosis()))


def test_binning2d_iadd():
    x_axis = core.Axis(np.linspace(-180, 180, 1), is_circle=True)
    y_axis = core.Axis(np.linspace(-90, 90, 1))

    hist2d = core.Histogram2DFloat64(x_axis, y_axis, None)
    hist2d.push(np.array([-180]), np.array([-90]), np.array([np.pi]))

    other = copy.copy(hist2d)
    other += hist2d

    assert np.all(hist2d.count() == 1)
    assert np.all(hist2d.mean() == np.pi)
    assert np.all(hist2d.min() == np.pi)
    assert np.all(hist2d.max() == np.pi)
    assert np.all(hist2d.variance() == 0)
    assert np.all(np.isnan(hist2d.skewness()))
    assert np.all(np.isnan(hist2d.kurtosis()))

    assert np.all(other.count() == 2)
    assert np.all(other.mean() == np.pi)
    assert np.all(other.min() == np.pi)
    assert np.all(other.max() == np.pi)
    assert np.all(other.variance() == 0)
    assert np.all(np.isnan(other.skewness()))
    assert np.all(np.isnan(other.kurtosis()))

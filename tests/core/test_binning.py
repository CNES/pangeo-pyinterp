# Copyright (c) 2020 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import copy
import os
import pickle
import unittest
import netCDF4
try:
    import matplotlib.pyplot
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False
import numpy as np
import pyinterp.core as core


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


class TestBinning2D(unittest.TestCase):
    """Test of the C+++/Python interface of the pyinterp::Binning2DFloat64
    class"""
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "dataset", "mss.nc")

    @classmethod
    def load_data(cls):
        with netCDF4.Dataset(cls.GRID) as ds:
            z = ds.variables['mss'][:].T
            z[z.mask] = float("nan")
            return ds.variables['lon'][:], ds.variables['lat'][:], z.data

    def test_binning2d_acessors(self):
        x_axis = core.Axis(np.linspace(-180, 180, 10), is_circle=True)
        y_axis = core.Axis(np.linspace(-90, 90, 10))

        binning = core.Binning2DFloat64(x_axis, y_axis)
        self.assertIsInstance(binning.x, core.Axis)
        self.assertIsInstance(binning.y, core.Axis)
        # The class must return a reference on the axes provided during
        # construction
        self.assertEqual(id(x_axis), id(binning.x))
        self.assertEqual(id(y_axis), id(binning.y))

        binning.clear()
        count = binning.count()
        self.assertIsInstance(count, np.ndarray)
        self.assertEqual(count.size, len(x_axis) * len(y_axis))
        self.assertEqual(count.mean(), 0)

    def test_binning2d_methods(self):
        x_axis = core.Axis(np.linspace(-180, 180, 361 // 4), is_circle=True)
        y_axis = core.Axis(np.linspace(-90, 90, 180 // 4))

        binning = core.Binning2DFloat64(x_axis, y_axis, None)
        x, y, z = self.load_data()
        mx, my = np.meshgrid(x, y, indexing='ij')
        binning.push(mx.flatten(), my.flatten(), z.flatten())

        count = binning.count()
        self.assertNotEqual(count.max(), 0)
        simple_mean = np.ma.fix_invalid(binning.mean())
        if HAVE_PLT:
            mx, my = np.meshgrid(x_axis[:], y_axis[:], indexing='ij')
            plot(mx, my, simple_mean, "binning2d_simple.png")

        mx, my = np.meshgrid(x, y, indexing='ij')
        binning.clear()
        binning.push(mx.flatten(), my.flatten(), z.flatten(), simple=False)

        count = binning.count()
        self.assertNotEqual(count.max(), 0)
        linear_mean = np.ma.fix_invalid(binning.mean())
        if HAVE_PLT:
            mx, my = np.meshgrid(x_axis[:], y_axis[:], indexing='ij')
            plot(mx, my, linear_mean, "binning2d_linear.png")

        self.assertFalse(np.all(linear_mean == simple_mean))

        # Test of access to statistical variables
        self.assertIsInstance(binning.kurtosis(), np.ndarray)
        self.assertIsInstance(binning.max(), np.ndarray)
        self.assertIsInstance(binning.min(), np.ndarray)
        self.assertIsInstance(binning.skewness(), np.ndarray)
        self.assertIsInstance(binning.sum(), np.ndarray)
        self.assertIsInstance(binning.sum_of_weights(), np.ndarray)
        self.assertIsInstance(binning.variance(), np.ndarray)

    def test_binning2d_pickle(self):
        x_axis = core.Axis(np.linspace(-180, 180, 1), is_circle=True)
        y_axis = core.Axis(np.linspace(-90, 90, 1))

        binning = core.Binning2DFloat64(x_axis, y_axis, None)
        binning.push([-180], [-90], [np.pi])

        self.assertTrue(np.all(binning.count() == 1))
        self.assertTrue(np.all(binning.mean() == np.pi))
        self.assertTrue(np.all(binning.min() == np.pi))
        self.assertTrue(np.all(binning.max() == np.pi))
        self.assertTrue(np.all(binning.sum() == np.pi))
        self.assertTrue(np.all(binning.sum_of_weights() == 1))
        self.assertTrue(np.all(binning.variance() == 0))
        self.assertTrue(np.all(np.isnan(binning.skewness())))
        self.assertTrue(np.all(np.isnan(binning.kurtosis())))

        other = pickle.loads(pickle.dumps(binning))

        self.assertTrue(np.all(other.count() == 1))
        self.assertTrue(np.all(other.mean() == np.pi))
        self.assertTrue(np.all(other.min() == np.pi))
        self.assertTrue(np.all(other.max() == np.pi))
        self.assertTrue(np.all(other.sum() == np.pi))
        self.assertTrue(np.all(other.sum_of_weights() == 1))
        self.assertTrue(np.all(other.variance() == 0))
        self.assertTrue(np.all(np.isnan(other.skewness())))
        self.assertTrue(np.all(np.isnan(other.kurtosis())))

    def test_binning2d_iadd(self):
        x_axis = core.Axis(np.linspace(-180, 180, 1), is_circle=True)
        y_axis = core.Axis(np.linspace(-90, 90, 1))

        binning = core.Binning2DFloat64(x_axis, y_axis, None)
        binning.push([-180], [-90], [np.pi])

        other = copy.copy(binning)
        other += binning

        self.assertTrue(np.all(binning.count() == 1))
        self.assertTrue(np.all(binning.mean() == np.pi))
        self.assertTrue(np.all(binning.min() == np.pi))
        self.assertTrue(np.all(binning.max() == np.pi))
        self.assertTrue(np.all(binning.sum() == np.pi))
        self.assertTrue(np.all(binning.sum_of_weights() == 1))
        self.assertTrue(np.all(binning.variance() == 0))
        self.assertTrue(np.all(np.isnan(other.skewness())))
        self.assertTrue(np.all(np.isnan(other.kurtosis())))

        self.assertTrue(np.all(other.count() == 2))
        self.assertTrue(np.all(other.mean() == np.pi))
        self.assertTrue(np.all(other.min() == np.pi))
        self.assertTrue(np.all(other.max() == np.pi))
        self.assertTrue(np.all(other.sum() == np.pi * 2))
        self.assertTrue(np.all(other.sum_of_weights() == 2))
        self.assertTrue(np.all(other.variance() == 0))
        self.assertTrue(np.all(np.isnan(other.skewness())))
        self.assertTrue(np.all(np.isnan(other.kurtosis())))


if __name__ == "__main__":
    unittest.main()

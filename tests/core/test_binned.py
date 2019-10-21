# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
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
    axe.pcolormesh(x, y, z, cmap='jet', norm=normalize)
    figure.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                filename),
                   bbox_inches='tight',
                   pad_inches=0.4)


class TestBinnedStatistics(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "dataset", "mss.nc")

    @classmethod
    def load_data(cls):
        with netCDF4.Dataset(cls.GRID) as ds:
            z = ds.variables['mss'][:].T
            z[z.mask] = float("nan")
            return ds.variables['lon'][:], ds.variables['lat'][:], z.data

    def test_acessors(self):
        x_axis = core.Axis(np.linspace(-180, 180, 10), is_circle=True)
        y_axis = core.Axis(np.linspace(-90, 90, 10))

        binned = core.statistics.BinnedFloat64(x_axis, y_axis)
        self.assertIsInstance(binned.x, core.Axis)
        self.assertIsInstance(binned.y, core.Axis)
        self.assertEqual(id(x_axis), id(binned.x))
        self.assertEqual(id(y_axis), id(binned.y))

        binned.clear()
        count = binned.count()
        self.assertIsInstance(count, np.ndarray)
        self.assertEqual(count.size, len(x_axis) * len(y_axis))
        self.assertEqual(count.mean(), 0)

    def test_mean(self):
        x_axis = core.Axis(np.linspace(-180, 180, 361 // 4), is_circle=True)
        y_axis = core.Axis(np.linspace(-90, 90, 180 // 4))

        binned = core.statistics.BinnedFloat64(x_axis, y_axis)
        x, y, z = self.load_data()
        mx, my = np.meshgrid(x, y, indexing='ij')
        binned.push(mx.flatten(), my.flatten(), z.flatten())

        count = binned.count()
        self.assertNotEqual(count.max(), 0)
        mean = np.ma.fix_invalid(binned.mean())
        if HAVE_PLT:
            mx, my = np.meshgrid(x_axis[:], y_axis[:], indexing='ij')
            plot(mx, my, mean, "binned_mean.png")
            plot(mx, my, count, "binned_count.png")


if __name__ == "__main__":
    unittest.main()

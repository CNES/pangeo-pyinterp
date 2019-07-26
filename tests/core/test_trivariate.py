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


class TestCase(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "dataset", "tcw.nc")

    @classmethod
    def load_data(cls):
        with netCDF4.Dataset(cls.GRID) as ds:
            z = ds.variables['tcw'][:].T
            z[z.mask] = float("nan")
            return core.Grid3DFloat64(
                core.Axis(ds.variables['longitude'][:], is_circle=True),
                core.Axis(ds.variables['latitude'][:]),
                core.Axis(ds.variables['time'][:]), z.data)


class TestGrid3D(TestCase):
    def test_init(self):
        grid = self.load_data()
        self.assertIsInstance(grid.x, core.Axis)
        self.assertIsInstance(grid.y, core.Axis)
        self.assertIsInstance(grid.z, core.Axis)
        self.assertIsInstance(grid.array, np.ndarray)

    def test_pickle(self):
        grid = self.load_data()
        other = pickle.loads(pickle.dumps(grid))
        self.assertEqual(grid.x, other.x)
        self.assertEqual(grid.y, other.y)
        self.assertEqual(grid.z, other.z)
        self.assertTrue(
            np.all(
                np.ma.fix_invalid(grid.array) == np.ma.fix_invalid(
                    other.array)))


class Trivariate(TestCase):
    def _test(self, interpolator, filename):
        grid = self.load_data()
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
        self.assertTrue(np.all(z1 == z0))
        if HAVE_PLT:
            plot(x.reshape(shape), y.reshape(shape), z0.reshape(shape),
                 filename)
        return z0

    def test_bounds_error(self):
        grid = self.load_data()
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
        with self.assertRaises(ValueError):
            core.trivariate_float64(grid,
                                    x.flatten(),
                                    y.flatten(),
                                    t.flatten(),
                                    interpolator,
                                    bounds_error=True,
                                    num_threads=0)

    def test_interpolator(self):
        a = self._test(core.Nearest3D(), "tcw_trivariate_nearest")
        b = self._test(core.Bilinear3D(), "tcw_trivariate_bilinear")
        c = self._test(core.InverseDistanceWeighting3D(), "tcw_trivariate_idw")
        self.assertTrue((a - b).std() != 0)
        self.assertTrue((a - c).std() != 0)
        self.assertTrue((b - c).std() != 0)


if __name__ == "__main__":
    unittest.main()

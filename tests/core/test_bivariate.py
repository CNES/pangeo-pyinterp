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
    """Common class managing the loading of processed data."""
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "dataset", "mss.nc")

    @classmethod
    def load_data(cls):
        with netCDF4.Dataset(cls.GRID) as ds:
            z = ds.variables['mss'][:].T
            z[z.mask] = float("nan")
            return core.Grid2DFloat64(
                core.Axis(ds.variables['lon'][:], is_circle=True),
                core.Axis(ds.variables['lat'][:]), z.data)


class TestGrid2D(TestCase):
    """Test of the C+++/Python interface of the pyinterp::Grid2DFloat64
    class"""
    def test_grid2d_init(self):
        """Test construction and accessors of the object"""
        grid = self.load_data()
        self.assertIsInstance(grid.x, core.Axis)
        self.assertIsInstance(grid.y, core.Axis)
        self.assertIsInstance(grid.array, np.ndarray)

    def test_grid2d_pickle(self):
        """Serialization test"""
        grid = self.load_data()
        other = pickle.loads(pickle.dumps(grid))
        self.assertEqual(grid.x, other.x)
        self.assertEqual(grid.y, other.y)
        self.assertTrue(
            np.all(
                np.ma.fix_invalid(grid.array) == np.ma.fix_invalid(
                    other.array)))


class TestBivariate(TestCase):
    """Test of the C+++/Python interface of the bivariate interpolator"""
    def _test(self, interpolator, filename):
        """Testing an interpolation method."""
        grid = self.load_data()
        lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
        lat = np.arange(-90, 90, 1 / 3.0) + 1 / 3.0
        x, y = np.meshgrid(lon, lat, indexing="ij")

        z0 = core.bivariate_float64(grid,
                                    x.flatten(),
                                    y.flatten(),
                                    interpolator,
                                    num_threads=0)

        z1 = core.bivariate_float64(grid,
                                    x.flatten(),
                                    y.flatten(),
                                    interpolator,
                                    num_threads=1)

        # The data from the mono-threads and multi-threads execution must be
        # identical.
        z0 = np.ma.fix_invalid(z0)
        z1 = np.ma.fix_invalid(z1)
        self.assertTrue(np.all(z1 == z0))

        if HAVE_PLT:
            plot(x, y, z0.reshape((len(lon), len(lat))), filename)

        # Out of bounds interpolation
        with self.assertRaises(ValueError):
            core.bivariate_float64(grid,
                                   x.flatten(),
                                   y.flatten(),
                                   interpolator,
                                   bounds_error=True,
                                   num_threads=0)

        return z0

    def test_bivariate_interpolator(self):
        """Testing of different interpolation methods"""
        a = self._test(core.Nearest2D(), "mss_bivariate_nearest")
        b = self._test(core.Bilinear2D(), "mss_bivariate_bilinear")
        c = self._test(core.InverseDistanceWeighting2D(), "mss_bivariate_idw")
        self.assertTrue((a - b).std() != 0)
        self.assertTrue((a - c).std() != 0)
        self.assertTrue((b - c).std() != 0)

    def test_bivariate_pickle(self):
        """Serialization of interpolator properties"""
        for item in [
                'InverseDistanceWeighting2D', 'InverseDistanceWeighting3D',
                'Bilinear2D', 'Bilinear3D', 'Nearest2D', 'Nearest3D'
        ]:
            obj = getattr(core, item)()
            self.assertIsInstance(obj, getattr(core, item))
            self.assertIsInstance(pickle.loads(pickle.dumps(obj)),
                                  getattr(core, item))


class TestBicubic(TestCase):
    """Test of the C+++/Python interface of the bicubic interpolator"""
    def test_bicubic_interpolator(self):
        """Testing of different bicubic interpolation methods"""
        grid = self.load_data()
        lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
        lat = np.arange(-90, 90, 1 / 3.0) + 1 / 3.0
        x, y = np.meshgrid(lon, lat, indexing="ij")
        z0 = core.bicubic_float64(grid,
                                  x.flatten(),
                                  y.flatten(),
                                  fitting_model=core.FittingModel.Akima,
                                  num_threads=0)
        z1 = core.bicubic_float64(grid,
                                  x.flatten(),
                                  y.flatten(),
                                  fitting_model=core.FittingModel.Akima,
                                  num_threads=1)
        z0 = np.ma.fix_invalid(z0)
        z1 = np.ma.fix_invalid(z1)
        self.assertTrue(np.all(z1 == z0))
        if HAVE_PLT:
            plot(x, y, z0.reshape((len(lon), len(lat))), "mss_akima.png")

        z0 = core.bicubic_float64(grid, x.flatten(), y.flatten())
        z0 = np.ma.fix_invalid(z0)
        self.assertFalse(np.all(z1 == z0))
        if HAVE_PLT:
            plot(x, y, z0.reshape((len(lon), len(lat))), "mss_cspline.png")

        # Out of bounds interpolation
        with self.assertRaises(ValueError):
            core.bicubic_float64(grid,
                                 x.flatten(),
                                 y.flatten(),
                                 fitting_model=core.FittingModel.Akima,
                                 bounds_error=True,
                                 num_threads=0)


if __name__ == "__main__":
    unittest.main()

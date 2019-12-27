# Copyright (c) 2020 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pickle
import unittest
import numpy as np
import pyinterp.core as core


class TestGrid4D(unittest.TestCase):
    """Test of the C+++/Python interface of the pyinterp::Grid4DFloat64
    class"""
    @staticmethod
    def f4d(x, y, z, u):
        return u * np.exp(-x**2 - y**2 - z**2)

    def load_data(self):
        x = np.arange(-1, 1, 0.2)
        y = np.arange(-1, 1, 0.2)
        z = np.arange(-1, 1, 0.2)
        u = np.arange(-1, 10, 0.2)

        mx, my, mz, mu = np.meshgrid(x, y, z, u)
        return core.Grid4DFloat64(core.Axis(x), core.Axis(y), core.Axis(z),
                                  core.Axis(u), self.f4d(mx, my, mz, mu))

    def test_grid4d_init(self):
        """Test construction and accessors of the object"""
        grid = self.load_data()
        self.assertIsInstance(grid.x, core.Axis)
        self.assertIsInstance(grid.y, core.Axis)
        self.assertIsInstance(grid.z, core.Axis)
        self.assertIsInstance(grid.u, core.Axis)
        self.assertIsInstance(grid.array, np.ndarray)

    def test_grid4d_pickle(self):
        """Serialization test"""
        grid = self.load_data()
        other = pickle.loads(pickle.dumps(grid))
        self.assertEqual(grid.x, other.x)
        self.assertEqual(grid.y, other.y)
        self.assertEqual(grid.z, other.z)
        self.assertEqual(grid.u, other.u)
        self.assertTrue(np.all(grid.array == other.array))

    def test_interpolator(self):
        grid = self.load_data()

        x = np.arange(-1, 1, 0.2)
        y = np.arange(-1, 1, 0.2)
        z = np.arange(-1, 1, 0.2)
        u = np.arange(-1, 10, 0.2)

        mx, my, mz, mu = np.meshgrid(x, y, z, u)
        expected = self.f4d(mx, my, mz, mu)

        interpolator = core.Bilinear3D()

        calculated = core.quadrivariate_float64(grid,
                                                mx.flatten(),
                                                my.flatten(),
                                                mz.flatten(),
                                                mu.flatten(),
                                                interpolator,
                                                num_threads=0,
                                                bounds_error=True)
        self.assertTrue(np.all(expected.flatten() == calculated))

        x = np.arange(-1, 1, 0.2)
        y = np.arange(-1, 1, 0.2)
        z = np.arange(-1, 1, 0.2)
        u = np.arange(-1, 10, 0.33333)

        mx, my, mz, mu = np.meshgrid(x, y, z, u)
        expected = self.f4d(mx, my, mz, mu)

        interpolator = core.Bilinear3D()

        calculated = core.quadrivariate_float64(grid,
                                                mx.flatten(),
                                                my.flatten(),
                                                mz.flatten(),
                                                mu.flatten(),
                                                interpolator,
                                                num_threads=0,
                                                bounds_error=False)
        self.assertAlmostEqual(np.nanstd(expected.flatten() - calculated), 0)


if __name__ == "__main__":
    unittest.main()

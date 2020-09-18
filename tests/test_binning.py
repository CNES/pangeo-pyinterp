# Copyright (c) 2020 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import os
import unittest
import dask.array as da
import dask.array.stats as das
import numpy as np
import xarray as xr
import pyinterp
import pyinterp.geodetic


class Nearest(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset",
                        "mss.nc")

    def init(self, dtype):
        ds = xr.load_dataset(self.GRID)

        x_axis = pyinterp.Axis(np.arange(-180, 180, 5), is_circle=True)
        y_axis = pyinterp.Axis(np.arange(-90, 95, 5))
        binning = pyinterp.Binning2D(x_axis,
                                     y_axis,
                                     pyinterp.geodetic.System(),
                                     dtype=dtype)
        self.assertEqual(x_axis, binning.x)
        self.assertEqual(y_axis, binning.y)
        self.assertIsInstance(str(binning), str)

        lon, lat = np.meshgrid(ds.lon, ds.lat)
        binning.push(lon, lat, ds.mss, simple=True)
        simple_mean = binning.variable('mean')
        self.assertIsInstance(simple_mean, np.ndarray)

        binning.clear()
        binning.push(lon, lat, ds.mss, simple=False)
        linear_mean = binning.variable('mean')
        self.assertIsInstance(simple_mean, np.ndarray)
        self.assertFalse(np.all(linear_mean == simple_mean))
        self.assertIsInstance(binning.variable("sum"), np.ndarray)
        self.assertIsInstance(binning.variable("count"), np.ndarray)

        with self.assertRaises(ValueError):
            binning.variable("_")

    def test_init(self):
        self.init(np.float64)
        self.init(np.float32)

        with self.assertRaises(ValueError):
            self.init(np.int8)

    def test_dask(self):
        x_axis = pyinterp.Axis(np.linspace(-180, 180, 1), is_circle=True)
        y_axis = pyinterp.Axis(np.linspace(-80, 80, 1))
        binning = pyinterp.Binning2D(x_axis, y_axis)

        x = da.full((4096 * 8, ), -180.0, dtype="f8", chunks=4096)
        y = da.full((4096 * 8, ), -80.0, dtype="f8", chunks=4096)
        z = da.random.uniform(size=4096 * 8, chunks=4096)

        binning = binning.push_delayed(x, y, z).compute()

        self.assertTrue(np.all(binning.variable("count") == 32768))
        self.assertEqual(binning.variable("sum_of_weights")[0, 0], 32768)
        self.assertAlmostEqual(
            binning.variable("mean")[0, 0],
            z.mean().compute())
        self.assertAlmostEqual(
            binning.variable("variance")[0, 0],
            z.std().compute()**2)
        self.assertAlmostEqual(
            binning.variable("sum")[0, 0],
            z.sum().compute())
        self.assertAlmostEqual(
            binning.variable("kurtosis")[0, 0],
            das.kurtosis(z).compute())
        self.assertAlmostEqual(
            binning.variable("skewness")[0, 0],
            das.skew(z).compute())


if __name__ == "__main__":
    unittest.main()

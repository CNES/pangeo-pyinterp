# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import os
import unittest
import numpy as np
import xarray as xr
import pyinterp.core
import pyinterp.bivariate
import pyinterp.bicubic
import pyinterp.trivariate


class Bivariate(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset",
                        "mss.nc")

    def test_biavariate(self):
        interpolator = pyinterp.bivariate.from_dataset(
            xr.open_dataset(self.GRID), "mss")
        self.assertIsInstance(interpolator.x, pyinterp.core.Axis)
        self.assertIsInstance(interpolator.y, pyinterp.core.Axis)
        self.assertIsInstance(interpolator.array, np.ndarray)

        lon = np.arange(-180, 180, 1) + 1 / 3.0
        lat = np.arange(-90, 90, 1) + 1 / 3.0
        x, y = np.meshgrid(lon, lat, indexing="ij")

        z = interpolator.evaluate(x.flatten(), y.flatten())
        self.assertIsInstance(z, np.ndarray)

    def test_bicubic(self):
        interpolator = pyinterp.bicubic.from_dataset(
            xr.open_dataset(self.GRID), "mss")
        self.assertIsInstance(interpolator.x, pyinterp.core.Axis)
        self.assertIsInstance(interpolator.y, pyinterp.core.Axis)
        self.assertIsInstance(interpolator.array, np.ndarray)

        lon = np.arange(-180, 180, 1) + 1 / 3.0
        lat = np.arange(-90, 90, 1) + 1 / 3.0
        x, y = np.meshgrid(lon, lat, indexing="ij")

        z = interpolator.evaluate(x.flatten(), y.flatten())
        self.assertIsInstance(z, np.ndarray)


class Trivariate(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset",
                        "tcw.nc")

    def test(self):
        interpolator = pyinterp.trivariate.from_dataset(
            xr.open_dataset(self.GRID), "tcw")
        self.assertIsInstance(interpolator.x, pyinterp.core.Axis)
        self.assertIsInstance(interpolator.y, pyinterp.core.Axis)
        self.assertIsInstance(interpolator.array, np.ndarray)

        lon = np.arange(-180, 180, 1) + 1 / 3.0
        lat = np.arange(-90, 90, 1) + 1 / 3.0
        time = 898500 + 3
        x, y, t = np.meshgrid(lon, lat, time, indexing="ij")

        z = interpolator.evaluate(x.flatten(), y.flatten(), t.flatten())
        self.assertIsInstance(z, np.ndarray)


if __name__ == "__main__":
    unittest.main()
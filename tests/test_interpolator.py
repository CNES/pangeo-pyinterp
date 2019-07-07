# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import collections
import os
import unittest
import pickle
import numpy as np
import xarray as xr
import pyinterp.core
import pyinterp.backends.xarray
import pyinterp.bicubic
import pyinterp.bivariate
import pyinterp.trivariate


class Bivariate(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset",
                        "mss.nc")

    def test_biavariate(self):
        interpolator = pyinterp.backends.xarray.Bivariate(
            xr.open_dataset(self.GRID), "mss")

        self.assertIsInstance(interpolator, pyinterp.bivariate.Bivariate)
        self.assertIsInstance(interpolator, pyinterp.backends.xarray.Bivariate)
        other = pickle.loads(pickle.dumps(interpolator))
        self.assertIsInstance(other, pyinterp.bivariate.Bivariate)
        self.assertIsInstance(other, pyinterp.backends.xarray.Bivariate)

        self.assertIsInstance(interpolator.x, pyinterp.core.Axis)
        self.assertIsInstance(interpolator.y, pyinterp.core.Axis)
        self.assertIsInstance(interpolator.array, np.ndarray)

        lon = np.arange(-180, 180, 1) + 1 / 3.0
        lat = np.arange(-90, 90, 1) + 1 / 3.0
        x, y = np.meshgrid(lon, lat, indexing="ij")

        z = interpolator.evaluate(
            collections.OrderedDict(lon=x.flatten(), lat=y.flatten()))
        self.assertIsInstance(z, np.ndarray)

    def test_bicubic(self):
        interpolator = pyinterp.backends.xarray.Bicubic(
            xr.open_dataset(self.GRID), "mss")

        self.assertIsInstance(interpolator, pyinterp.backends.xarray.Bicubic)
        self.assertIsInstance(interpolator, pyinterp.bicubic.Bicubic)
        other = pickle.loads(pickle.dumps(interpolator))
        self.assertIsInstance(other, pyinterp.backends.xarray.Bicubic)
        self.assertIsInstance(other, pyinterp.bicubic.Bicubic)

        self.assertIsInstance(interpolator.x, pyinterp.core.Axis)
        self.assertIsInstance(interpolator.y, pyinterp.core.Axis)
        self.assertIsInstance(interpolator.array, np.ndarray)

        lon = np.arange(-180, 180, 1) + 1 / 3.0
        lat = np.arange(-90, 90, 1) + 1 / 3.0
        x, y = np.meshgrid(lon, lat, indexing="ij")

        z = interpolator.evaluate(
            collections.OrderedDict(lon=x.flatten(), lat=y.flatten()))
        self.assertIsInstance(z, np.ndarray)


class Trivariate(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset",
                        "tcw.nc")

    def test(self):
        interpolator = pyinterp.backends.xarray.Trivariate(
            xr.open_dataset(self.GRID), "tcw")

        self.assertIsInstance(interpolator,
                              pyinterp.backends.xarray.Trivariate)
        self.assertIsInstance(interpolator, pyinterp.trivariate.Trivariate)
        other = pickle.loads(pickle.dumps(interpolator))
        self.assertIsInstance(other, pyinterp.backends.xarray.Trivariate)
        self.assertIsInstance(other, pyinterp.trivariate.Trivariate)

        self.assertIsInstance(interpolator.x, pyinterp.core.Axis)
        self.assertIsInstance(interpolator.y, pyinterp.core.Axis)
        self.assertIsInstance(interpolator.z, pyinterp.core.Axis)
        self.assertIsInstance(interpolator.array, np.ndarray)

        lon = np.arange(-180, 180, 1) + 1 / 3.0
        lat = np.arange(-90, 90, 1) + 1 / 3.0
        time = 898500 + 3
        x, y, t = np.meshgrid(lon, lat, time, indexing="ij")

        z = interpolator.evaluate(
            collections.OrderedDict(longitude=x.flatten(),
                                    latitude=y.flatten(),
                                    time=t.flatten()))
        self.assertIsInstance(z, np.ndarray)


if __name__ == "__main__":
    unittest.main()
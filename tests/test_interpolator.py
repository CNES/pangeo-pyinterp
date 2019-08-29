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


class Grid2D(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset",
                        "mss.nc")

    def test_biavariate(self):
        grid = pyinterp.backends.xarray.Grid2D(
            xr.open_dataset(self.GRID).data_vars["mss"])

        self.assertIsInstance(grid, pyinterp.backends.xarray.Grid2D)
        self.assertIsInstance(grid, pyinterp.grid.Grid2D)
        other = pickle.loads(pickle.dumps(grid))
        self.assertIsInstance(other, pyinterp.backends.xarray.Grid2D)
        self.assertIsInstance(grid, pyinterp.grid.Grid2D)

        self.assertIsInstance(grid.x, pyinterp.core.Axis)
        self.assertIsInstance(grid.y, pyinterp.core.Axis)
        self.assertIsInstance(grid.array, np.ndarray)

        lon = np.arange(-180, 180, 1) + 1 / 3.0
        lat = np.arange(-90, 90, 1) + 1 / 3.0
        x, y = np.meshgrid(lon, lat, indexing="ij")

        z = grid.bivariate(
            collections.OrderedDict(lon=x.flatten(), lat=y.flatten()))
        self.assertIsInstance(z, np.ndarray)

        with self.assertRaises(ValueError):
            grid.bivariate(collections.OrderedDict(lon=x.flatten(),
                                                   lat=y.flatten()),
                           bounds_error=True)

    def test_bicubic(self):
        grid = pyinterp.backends.xarray.Grid2D(
            xr.open_dataset(self.GRID).data_vars["mss"])

        lon = np.arange(-180, 180, 1) + 1 / 3.0
        lat = np.arange(-90, 90, 1) + 1 / 3.0
        x, y = np.meshgrid(lon, lat, indexing="ij")

        z = grid.bicubic(
            collections.OrderedDict(lon=x.flatten(), lat=y.flatten()))
        self.assertIsInstance(z, np.ndarray)

        with self.assertRaises(ValueError):
            grid.bicubic(collections.OrderedDict(lon=x.flatten(),
                                                 lat=y.flatten()),
                         bounds_error=True)

        with self.assertRaises(ValueError):
            grid.bicubic(collections.OrderedDict(lon=x.flatten(),
                                                 lat=y.flatten()),
                         bounds_error=True,
                         boundary="sym")


class Trivariate(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset",
                        "tcw.nc")

    def test(self):
        grid = pyinterp.backends.xarray.Grid3D(
            xr.open_dataset(self.GRID).data_vars["tcw"])

        self.assertIsInstance(grid, pyinterp.backends.xarray.Grid3D)
        self.assertIsInstance(grid, pyinterp.grid.Grid3D)
        other = pickle.loads(pickle.dumps(grid))
        self.assertIsInstance(other, pyinterp.backends.xarray.Grid3D)
        self.assertIsInstance(grid, pyinterp.grid.Grid3D)

        self.assertIsInstance(grid.x, pyinterp.core.Axis)
        self.assertIsInstance(grid.y, pyinterp.core.Axis)
        self.assertIsInstance(grid.z, pyinterp.core.Axis)
        self.assertIsInstance(grid.array, np.ndarray)

        lon = np.arange(-180, 180, 1) + 1 / 3.0
        lat = np.arange(-90, 90, 1) + 1 / 3.0
        time = 898500 + 3
        x, y, t = np.meshgrid(lon, lat, time, indexing="ij")

        z = grid.trivariate(
            collections.OrderedDict(longitude=x.flatten(),
                                    latitude=y.flatten(),
                                    time=t.flatten()))
        self.assertIsInstance(z, np.ndarray)

        with self.assertRaises(ValueError):
            grid.trivariate(collections.OrderedDict(longitude=x.flatten(),
                                                    latitude=y.flatten(),
                                                    time=t.flatten()),
                            bounds_error=True)


if __name__ == "__main__":
    unittest.main()

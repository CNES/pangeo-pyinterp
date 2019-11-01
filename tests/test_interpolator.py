# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import datetime
import collections
import os
import unittest
import pickle
import numpy as np
import xarray as xr
import pyinterp.backends.xarray
import pyinterp


class Grid2D(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset",
                        "mss.nc")

    def test_biavariate(self):
        grid = pyinterp.backends.xarray.Grid2D(xr.load_dataset(self.GRID).mss)

        self.assertIsInstance(grid, pyinterp.backends.xarray.Grid2D)
        self.assertIsInstance(grid, pyinterp.Grid2D)
        other = pickle.loads(pickle.dumps(grid))
        self.assertIsInstance(other, pyinterp.backends.xarray.Grid2D)
        self.assertIsInstance(grid, pyinterp.Grid2D)

        self.assertIsInstance(grid.x, pyinterp.Axis)
        self.assertIsInstance(grid.y, pyinterp.Axis)
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
        grid = pyinterp.backends.xarray.Grid2D(xr.load_dataset(self.GRID).mss)

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
        grid = pyinterp.backends.xarray.Grid3D(xr.load_dataset(self.GRID).tcw,
                                               increasing_axes=True)

        self.assertIsInstance(grid, pyinterp.backends.xarray.Grid3D)
        self.assertIsInstance(grid, pyinterp.Grid3D)
        other = pickle.loads(pickle.dumps(grid))
        self.assertIsInstance(other, pyinterp.backends.xarray.Grid3D)
        self.assertIsInstance(grid, pyinterp.Grid3D)

        self.assertIsInstance(grid.x, pyinterp.Axis)
        self.assertIsInstance(grid.y, pyinterp.Axis)
        self.assertIsInstance(grid.z, pyinterp.Axis)
        self.assertIsInstance(grid.array, np.ndarray)

        lon = np.arange(-180, 180, 1) + 1 / 3.0
        lat = np.arange(-90, 90, 1) + 1 / 3.0
        time = np.array([datetime.datetime(2002, 7, 2, 15, 0)],
                        grid.time_unit())
        x, y, t = np.meshgrid(lon, lat, time, indexing="ij")

        z = grid.trivariate(
            collections.OrderedDict(longitude=x.flatten(),
                                    latitude=y.flatten(),
                                    time=t.flatten()))
        self.assertIsInstance(z, np.ndarray)

        with self.assertRaises(ValueError):
            time = np.array([datetime.datetime(2012, 7, 2, 15, 0)],
                            grid.time_unit())
            x, y, t = np.meshgrid(lon, lat, time, indexing="ij")
            grid.trivariate(collections.OrderedDict(longitude=x.flatten(),
                                                    latitude=y.flatten(),
                                                    time=t.flatten()),
                            bounds_error=True)


class TestRTree(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset",
                        "mss.nc")

    @classmethod
    def load_data(cls):
        ds = xr.load_dataset(cls.GRID)
        z = ds.mss.T
        x, y = np.meshgrid(ds.lon.values, ds.lat.values, indexing='ij')
        mesh = pyinterp.RTree()
        mesh.packing(
            np.vstack((x.flatten(), y.flatten())).T, z.values.flatten())
        return mesh

    def test_interpolate(self):
        mesh = self.load_data()
        lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
        lat = np.arange(-90, 90, 1 / 3.0) + 1 / 3.0
        x, y = np.meshgrid(lon, lat, indexing="ij")
        coordinates = np.vstack((x.flatten(), y.flatten())).T
        mesh.query(coordinates)
        mesh.inverse_distance_weighting(coordinates)


if __name__ == "__main__":
    unittest.main()

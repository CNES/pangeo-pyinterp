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

        z = grid.bivariate(collections.OrderedDict(lon=x.flatten(),
                                                   lat=y.flatten()),
                           interpolator="nearest")
        self.assertIsInstance(z, np.ndarray)

        z = grid.bivariate(collections.OrderedDict(lon=x.flatten(),
                                                   lat=y.flatten()),
                           interpolator="inverse_distance_weighting")
        self.assertIsInstance(z, np.ndarray)

        with self.assertRaises(ValueError):
            grid.bivariate(collections.OrderedDict(lon=x.flatten(),
                                                   lat=y.flatten()),
                           bounds_error=True)

        lon = pyinterp.Axis(0, 360, 100, is_circle=True)
        lat = pyinterp.Axis(-80, 80, 50, is_circle=False)
        array, _ = np.meshgrid(lon[:], lat[:])

        with self.assertRaises(ValueError):
            pyinterp.Grid2D(lon, lat, array)

        grid = pyinterp.Grid2D(lon, lat, array.T)

        self.assertIsInstance(grid, pyinterp.Grid2D)
        self.assertIsInstance(str(grid), str)

        with self.assertRaises(ValueError):
            pyinterp.Grid2D(lon, lat, array, increasing_axes='_')

    def test_bicubic(self):
        grid = pyinterp.backends.xarray.Grid2D(xr.load_dataset(self.GRID).mss)

        lon = np.arange(-180, 180, 1) + 1 / 3.0
        lat = np.arange(-90, 90, 1) + 1 / 3.0
        x, y = np.meshgrid(lon, lat, indexing="ij")

        z = grid.bicubic(
            collections.OrderedDict(lon=x.flatten(), lat=y.flatten()))
        self.assertIsInstance(z, np.ndarray)

        for fitting_model in [
                'linear', 'polynomial', 'c_spline', 'c_spline_periodic',
                'akima', 'akima_periodic', 'steffen'
        ]:
            other = grid.bicubic(collections.OrderedDict(lon=x.flatten(),
                                                         lat=y.flatten()),
                                 fitting_model=fitting_model)
            self.assertNotEqual((z - other).mean(), 0)

        with self.assertRaises(ValueError):
            grid.bicubic(collections.OrderedDict(lon=x.flatten(),
                                                 lat=y.flatten()),
                         bounds_error=True)

        with self.assertRaises(ValueError):
            grid.bicubic(collections.OrderedDict(lon=x.flatten(),
                                                 lat=y.flatten()),
                         bounds_error=True,
                         boundary="sym")

        x_axis = pyinterp.Axis(-180, 179, 360, is_circle=True)
        y_axis = pyinterp.Axis(-90, 90, 181, is_circle=False)
        z_axis = pyinterp.Axis(0, 10, 10, is_circle=False)
        matrix, _ = np.meshgrid(x_axis[:], y_axis[:])
        grid = pyinterp.Grid2D(x_axis, y_axis, matrix.T)

        self.assertIsInstance(grid, pyinterp.Grid2D)
        with self.assertRaises(ValueError):
            pyinterp.bicubic(grid, x.flatten(), y.flatten(), fitting_model='_')
        with self.assertRaises(ValueError):
            pyinterp.bicubic(grid, x.flatten(), y.flatten(), boundary='_')
        grid = pyinterp.Grid2D(x_axis.flip(inplace=False), y_axis, matrix.T)
        with self.assertRaises(ValueError):
            pyinterp.bicubic(grid, x.flatten(), y.flatten())

        grid = pyinterp.Grid2D(x_axis, y_axis.flip(), matrix.T)
        with self.assertRaises(ValueError):
            pyinterp.bicubic(grid, x.flatten(), y.flatten())

        matrix, _, _ = np.meshgrid(x_axis[:], y_axis[:], z_axis[:])
        grid = pyinterp.Grid3D(x_axis, y_axis, z_axis,
                               matrix.transpose(1, 0, 2))
        with self.assertRaises(ValueError):
            pyinterp.bicubic(grid, x.flatten(), y.flatten())


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

        z = grid.bicubic(
            collections.OrderedDict(longitude=x.flatten()[1:2],
                                    latitude=y.flatten()[1:2],
                                    time=t.flatten()[1:2]))
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

    def init(self, dtype):
        lon = np.arange(-180, 180, 10)
        lat = np.arange(-90, 90, 10)
        lon, lat = np.meshgrid(lon, lat)
        data = lon * 0
        mesh = pyinterp.RTree(dtype=dtype)
        self.assertIsInstance(mesh, pyinterp.RTree)
        self.assertEqual(len(mesh), 0)
        self.assertFalse(bool(mesh))
        mesh.packing(
            np.vstack((lon.flatten(), lat.flatten())).T, data.flatten())
        self.assertEqual(len(mesh), len(lon.flatten()))
        self.assertTrue(bool(mesh))
        (x_min, y_min, z_min), (x_max, y_max, z_max) = mesh.bounds()
        self.assertEqual(x_min, -180)
        self.assertEqual(y_min, -90.0)
        self.assertEqual(x_max, 180.0)
        self.assertEqual(y_max, 80)
        self.assertAlmostEqual(z_min,
                               0,
                               delta=1e-6 if dtype == np.float64 else 0.5)
        self.assertAlmostEqual(z_max,
                               0,
                               delta=1e-6 if dtype == np.float64 else 0.5)
        mesh.clear()
        self.assertEqual(len(mesh), 0)
        self.assertFalse(bool(mesh))
        mesh.insert(
            np.vstack((lon.flatten(), lat.flatten())).T, data.flatten())
        self.assertEqual(len(mesh), len(lon.flatten()))
        self.assertIsInstance(pickle.loads(pickle.dumps(mesh)), pyinterp.RTree)

    def test_init(self):
        self.init(dtype=np.float32)
        self.init(dtype=np.float64)

        with self.assertRaises(ValueError):
            self.init(np.int8)

        with self.assertRaises(ValueError):
            mesh = pyinterp.RTree()
            mesh.__setstate__((1, ))

    def load_data(self):
        ds = xr.load_dataset(self.GRID)
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

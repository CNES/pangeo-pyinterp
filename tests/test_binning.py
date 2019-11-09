import os
import unittest
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


if __name__ == "__main__":
    unittest.main()

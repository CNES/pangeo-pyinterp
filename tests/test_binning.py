import os
import unittest
import numpy as np
import xarray as xr
import pyinterp
import pyinterp.geodetic


class Nearest(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset",
                        "mss.nc")

    def test_init(self):
        ds = xr.load_dataset(self.GRID)

        binning = pyinterp.Binning2D(
            pyinterp.Axis(np.arange(-180, 180, 5), is_circle=True),
            pyinterp.Axis(np.arange(-90, 95, 5)), pyinterp.geodetic.System())

        lon, lat = np.meshgrid(ds.lon, ds.lat)
        binning.push(lon, lat, ds.mss, simple=True)
        simple_mean = binning.variable('mean')
        self.assertIsInstance(simple_mean, np.ndarray)

        binning.clear()
        binning.push(lon, lat, ds.mss, simple=False)
        linear_mean = binning.variable('mean')
        self.assertIsInstance(simple_mean, np.ndarray)
        self.assertFalse(np.all(linear_mean == simple_mean))


if __name__ == "__main__":
    unittest.main()

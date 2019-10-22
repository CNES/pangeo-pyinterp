import os
import unittest
import numpy as np
import xarray as xr
import pyinterp.binning


class Nearest(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset",
                        "mss.nc")

    def test_init(self):
        ds = xr.load_dataset(self.GRID)

        binned = pyinterp.binning.NearestBivariate(
            pyinterp.Axis(np.arange(-180, 180, 5), is_circle=True),
            pyinterp.Axis(np.arange(-90, 95, 5)))

        lon, lat = np.meshgrid(ds.lon, ds.lat)
        binned.push(lon, lat, ds.mss)

        self.assertIsInstance(binned.variable('mean'), np.ndarray)


if __name__ == "__main__":
    unittest.main()

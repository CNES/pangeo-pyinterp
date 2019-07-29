import unittest
import os
import netCDF4
import numpy as np
import pyinterp.fill


class Fill(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset",
                        "mss.nc")

    @classmethod
    def _load(cls, cube=False):
        ds = netCDF4.Dataset(cls.GRID)
        x_axis = pyinterp.core.Axis(ds.variables["lon"][:], is_circle=True)
        y_axis = pyinterp.core.Axis(ds.variables["lat"][:])
        mss = ds.variables["mss"][:].T
        mss[mss.mask] = float("nan")
        if cube:
            z_axis = pyinterp.core.Axis(np.arange(2))
            mss = np.stack([mss.data] * len(z_axis)).transpose(1, 2, 0)
            return pyinterp.grid.Grid3D(x_axis, y_axis, z_axis, mss)
        return pyinterp.grid.Grid2D(x_axis, y_axis, mss.data)

    def test_loess(self):
        grid = self._load()
        filled0 = pyinterp.fill.loess(grid, num_threads=0)
        filled1 = pyinterp.fill.loess(grid, num_threads=1)
        data = np.copy(grid.array)
        data[np.isnan(data)] = 0
        filled0[np.isnan(filled0)] = 0
        filled1[np.isnan(filled1)] = 0
        self.assertEqual((filled0 - filled1).mean(), 0)
        self.assertEqual(np.ma.fix_invalid(grid.array - filled1).mean(), 0)
        self.assertNotEqual((data - filled1).mean(), 0)

    def test_gauss_seidel(self):
        grid = self._load()
        _, filled0 = pyinterp.fill.gauss_seidel(grid, num_threads=0)
        _, filled1 = pyinterp.fill.gauss_seidel(grid, num_threads=1)
        data = np.copy(grid.array)
        data[np.isnan(data)] = 0
        filled0[np.isnan(filled0)] = 0
        filled1[np.isnan(filled1)] = 0
        self.assertEqual((filled0 - filled1).mean(), 0)
        self.assertEqual(np.ma.fix_invalid(grid.array - filled1).mean(), 0)
        self.assertNotEqual((data - filled1).mean(), 0)

    def test_loess_3d(self):
        grid = self._load(True)
        filled0 = pyinterp.fill.loess(grid, num_threads=0)
        filled0[np.isnan(filled0)] = 0
        self.assertEqual((filled0[:, :, 0] - filled0[:, :, 1]).mean(), 0)

    def test_gauss_seidel_3d(self):
        grid = self._load(True)
        _, filled0 = pyinterp.fill.gauss_seidel(grid, num_threads=0)
        self.assertEqual((filled0[:, :, 0] - filled0[:, :, 1]).mean(), 0)


if __name__ == "__main__":
    unittest.main()

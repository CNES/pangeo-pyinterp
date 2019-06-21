import os
import pickle
import unittest
import netCDF4
try:
    import matplotlib.pyplot
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False
import numpy as np
import pyinterp.core as core


def plot2d(x, y, z, filename):
    figure = matplotlib.pyplot.figure(figsize=(15, 15), dpi=150)
    value = z.mean()
    std = z.std()
    normalize = matplotlib.colors.Normalize(vmin=value - 3 * std,
                                            vmax=value + 3 * std)
    axe = figure.add_subplot(2, 1, 1)
    axe.pcolormesh(x, y, z, cmap='jet', norm=normalize)
    figure.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                filename),
                   bbox_inches='tight',
                   pad_inches=0.4)


class TestBivariate2D(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mss.nc")

    @classmethod
    def _load_data(cls):
        with netCDF4.Dataset(cls.GRID) as ds:
            z = ds.variables['mss'][:].T
            z[z.mask] = float("nan")
            return core.cartesian.Bivariate(
                core.Axis(ds.variables['lon'][:], is_circle=True),
                core.Axis(ds.variables['lat'][:]), z.data)

    def _test(self, interpolator, filename):
        bivariate = self._load_data()
        lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
        lat = np.arange(-90, 90, 1 / 3.0) + 1 / 3.0
        x, y = np.meshgrid(lon, lat, indexing="ij")
        z0 = bivariate.evaluate(x.flatten(),
                                y.flatten(),
                                interpolator,
                                num_threads=0)
        z1 = bivariate.evaluate(x.flatten(),
                                y.flatten(),
                                interpolator,
                                num_threads=1)
        z0 = np.ma.fix_invalid(z0)
        z1 = np.ma.fix_invalid(z1)
        self.assertTrue(np.all(z1 == z0))
        if HAVE_PLT:
            plot2d(x, y, z0.reshape((len(lon), len(lat))), filename)

    def test_nearest(self):
        self._test(core.cartesian.Nearest2D(), "mss_cartesian_nearest")

    def test_bilinear(self):
        self._test(core.cartesian.Bilinear2D(), "mss_cartesian_bilinear")

    def test_idw(self):
        self._test(core.cartesian.InverseDistanceWeighting2D(),
                   "mss_cartesian_idw")


class TestBivariate3D(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tcw.nc")

    @classmethod
    def _load_data(cls):
        with netCDF4.Dataset(cls.GRID) as ds:
            z = ds.variables['tcw'][:].T
            z[z.mask] = float("nan")
            return core.cartesian.Trivariate(
                core.Axis(ds.variables['longitude'][:], is_circle=True),
                core.Axis(ds.variables['latitude'][:]),
                core.Axis(ds.variables['time'][:]), z.data)

    def _test(self, interpolator, filename):
        trivariate = self._load_data()
        lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
        lat = np.arange(-90, 90, 1 / 3.0) + 1 / 3.0
        time = 898500 + 3
        x, y, t = np.meshgrid(lon, lat, time, indexing="ij")
        z0 = trivariate.evaluate(x.flatten(),
                                 y.flatten(),
                                 t.flatten(),
                                 interpolator,
                                 num_threads=0)
        z1 = trivariate.evaluate(x.flatten(),
                                 y.flatten(),
                                 t.flatten(),
                                 interpolator,
                                 num_threads=1)
        z0 = np.ma.fix_invalid(z0)
        z1 = np.ma.fix_invalid(z1)
        self.assertTrue(np.all(z1 == z0))
        shape = (len(lon), len(lat))
        if HAVE_PLT:
            plot2d(x.reshape(shape), y.reshape(shape), z0.reshape(shape),
                   filename)

    def test_nearest(self):
        self._test(core.cartesian.Nearest3D(), "mss_cartesian_nearest3d")

    def test_bilinear(self):
        self._test(core.cartesian.Bilinear3D(), "mss_cartesian_bilinear3d")

    def test_idw(self):
        self._test(core.cartesian.InverseDistanceWeighting3D(),
                   "mss_cartesian_idw3d")

    # def test_pickle(self):
    #     interpolator = self._load_data()
    #     other = pickle.loads(pickle.dumps(interpolator))
    #     self.assertEqual(interpolator.x, other.x)
    #     self.assertEqual(interpolator.y, other.y)
    #     self.assertEqual(interpolator.z, other.z)
    #     self.assertTrue(
    #         np.all(
    #             np.ma.fix_invalid(interpolator.array) == np.ma.fix_invalid(
    #                 other.array)))

if __name__ == "__main__":
    unittest.main()
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


def plot(x, y, z, filename):
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


class TestCase(unittest.TestCase):
    GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mss.nc")

    @classmethod
    def load_data(cls):
        with netCDF4.Dataset(cls.GRID) as ds:
            z = ds.variables['mss'][:].T
            z[z.mask] = float("nan")
            return core.geodetic.Bivariate(
                core.Axis(ds.variables['lon'][:], is_circle=True),
                core.Axis(ds.variables['lat'][:]), z.data)


class TestBivariate(TestCase):
    def _test(self, interpolator, filename):
        bivariate = self.load_data()
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
            plot(x, y, z0.reshape((len(lon), len(lat))), filename)

    def test_nearest(self):
        self._test(core.geodetic.Nearest2D(), "mss_geodetic_nearest")

    def test_bilinear(self):
        self._test(core.geodetic.Bilinear2D(), "mss_geodetic_bilinear")

    def test_idw(self):
        self._test(core.geodetic.InverseDistanceWeighting2D(),
                   "mss_geodetic_idw")


if __name__ == "__main__":
    unittest.main()
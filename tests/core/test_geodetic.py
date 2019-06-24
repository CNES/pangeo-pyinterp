import pickle
import unittest
import math
import numpy as np
from pyinterp import core


class TextSystem(unittest.TestCase):
    def test_wgs84(self):
        wgs84 = core.geodetic.System()
        # https://fr.wikipedia.org/wiki/WGS_84
        # https://en.wikipedia.org/wiki/Geodetic_datum
        # http://earth-info.nga.mil/GandG/publications/tr8350.2/wgs84fin.pdf
        self.assertAlmostEqual(wgs84.semi_major_axis, 6378137)
        self.assertAlmostEqual(wgs84.flattening, 1 / 298.257223563)
        self.assertAlmostEqual(wgs84.semi_minor_axis(),
                               6356752.314245179497563967)
        self.assertAlmostEqual(
            math.sqrt(wgs84.first_eccentricity_squared()),
            0.081819190842622,
            delta=1e-15)
        self.assertAlmostEqual(
            math.sqrt(wgs84.second_eccentricity_squared()),
            8.2094437949696 * 1e-2,
            delta=1e-15)
        self.assertAlmostEqual(
            wgs84.equatorial_circumference() * 1e-3, 40075.017, delta=1e-3)
        self.assertAlmostEqual(
            wgs84.equatorial_circumference(False) * 1e-3,
            39940.652,
            delta=1e-3)
        self.assertAlmostEqual(
            wgs84.polar_radius_of_curvature(), 6399593.6258, delta=1e-4)
        self.assertAlmostEqual(
            wgs84.equatorial_radius_of_curvature(), 6335439.3272, delta=1e-4)
        self.assertAlmostEqual(wgs84.axis_ratio(), 0.996647189335, delta=1e-12)
        self.assertAlmostEqual(
            wgs84.linear_eccentricity(), 5.2185400842339 * 1E5, delta=1e-6)
        self.assertAlmostEqual(wgs84.mean_radius(), 6371008.7714, delta=1e-4)
        self.assertAlmostEqual(
            wgs84.authalic_radius(), 6371007.1809, delta=1e-4)
        self.assertAlmostEqual(
            wgs84.volumetric_radius(), 6371000.7900, delta=1e-4)

    def test_operator(self):
        wgs84 = core.geodetic.System()
        # https://en.wikipedia.org/wiki/Geodetic_Reference_System_1980
        grs80 = core.geodetic.System(6378137, 1 / 298.257222101)
        self.assertAlmostEqual(grs80.semi_major_axis, 6378137)
        self.assertAlmostEqual(grs80.flattening, 1 / 298.257222101)
        self.assertEqual(wgs84, wgs84)
        self.assertNotEqual(wgs84, grs80)

    def test_pickle(self):
        wgs84 = core.geodetic.System()
        self.assertEqual(wgs84, pickle.loads(pickle.dumps(wgs84)))


class TestCoordinates(unittest.TestCase):
    def test_ecef_lla(self):
        lon, lat, alt = core.geodetic.Coordinates(None).ecef_to_lla(
            [1176498.769459714], [5555043.905503586], [2895446.8901510699])
        self.assertAlmostEqual(78.042068, lon[0], delta=1e-8)
        self.assertAlmostEqual(27.173891, lat[0], delta=1e-8)
        self.assertAlmostEqual(168.0, alt[0], delta=1e-8)

    def test_lla_to_ecef(self):
        x, y, z = core.geodetic.Coordinates(None).lla_to_ecef(
            [78.042068], [27.173891], [168.0])
        self.assertAlmostEqual(1176498.769459714, x[0], delta=1e-8)
        self.assertAlmostEqual(5555043.905503586, y[0], delta=1e-8)
        self.assertAlmostEqual(2895446.8901510699, z[0], delta=1e-8)

    def test_round_trip(self):
        lon1 = np.random.uniform(-180.0, 180.0, 1000000)
        lat1 = np.random.uniform(-90.0, 90.0, 1000000)
        alt1 = np.random.uniform(-10000, 100000, 1000000)

        a = core.geodetic.Coordinates(None)
        b = core.geodetic.Coordinates(None)

        lon2, lat2, alt2 = a.transform(b, lon1, lat1, alt1, num_threads=0)

        self.assertAlmostEqual((lon1 - lon2).mean(), 0, delta=1e-12)
        self.assertAlmostEqual((lat1 - lat2).mean(), 0, delta=1e-12)
        self.assertAlmostEqual((alt1 - alt2).mean(), 0, delta=1e-11)

    def test_pickle(self):
        a = core.geodetic.Coordinates(None)
        b = pickle.loads(pickle.dumps(a))
        self.assertTrue(np.all(a.__getstate__() == b.__getstate__()))


if __name__ == "__main__":
    unittest.main()

import unittest
import pyinterp.geodetic


class System(unittest.TestCase):
    def test_init(self):
        wgs = pyinterp.geodetic.System()
        self.assertIsInstance(wgs, pyinterp.geodetic.System)
        with self.assertRaises(TypeError):
            wgs = pyinterp.geodetic.System(12.0)
        with self.assertRaises(TypeError):
            wgs = pyinterp.geodetic.System((12.0, 3.0, 5))
        wgs = pyinterp.geodetic.System((1, 1))
        self.assertIsInstance(wgs, pyinterp.geodetic.System)


class Coordinates(unittest.TestCase):
    def test_init(self):
        wgs = pyinterp.geodetic.Coordinates()
        self.assertIsInstance(wgs, pyinterp.geodetic.Coordinates)
        wgs = pyinterp.geodetic.Coordinates(pyinterp.geodetic.System())
        self.assertIsInstance(wgs, pyinterp.geodetic.Coordinates)


if __name__ == "__main__":
    unittest.main()

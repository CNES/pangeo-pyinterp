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


class Point2D(unittest.TestCase):
    def test_init(self):
        # pt = pyinterp.geodetic.Point2D()
        # self.assertEqual(pt.lon, 0)
        # self.assertEqual(pt.lat, 0)
        pt = pyinterp.geodetic.Point2D(1, 2)
        self.assertEqual(pt.lon, 1)
        self.assertEqual(pt.lat, 2)


class Box2D(unittest.TestCase):
    def test_init(self):
        box = pyinterp.geodetic.Box2D()
        # self.assertEqual(box.min_corner.lon, 0)
        # self.assertEqual(box.min_corner.lat, 0)
        # self.assertEqual(box.max_corner.lon, 0)
        # self.assertEqual(box.max_corner.lat, 0)

        box = pyinterp.geodetic.Box2D.entire_earth()
        self.assertEqual(box.min_corner.lon, -180)
        self.assertEqual(box.min_corner.lat, -90)
        self.assertEqual(box.max_corner.lon, 180)
        self.assertEqual(box.max_corner.lat, 90)

        box = pyinterp.geodetic.Box2D(pyinterp.geodetic.Point2D(1, 2),
                                      pyinterp.geodetic.Point2D(3, 4))
        self.assertEqual(box.min_corner.lon, 1)
        self.assertEqual(box.min_corner.lat, 2)
        self.assertEqual(box.max_corner.lon, 3)
        self.assertEqual(box.max_corner.lat, 4)


if __name__ == "__main__":
    unittest.main()

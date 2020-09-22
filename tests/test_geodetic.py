# Copyright (c) 2020 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
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


class Point(unittest.TestCase):
    def test_init(self):
        pt = pyinterp.geodetic.Point(1, 2)
        self.assertEqual(pt.lon, 1)
        self.assertEqual(pt.lat, 2)

    def test_deprecated(self):
        with self.assertWarns(PendingDeprecationWarning):
            pyinterp.geodetic.Point2D()


class Box(unittest.TestCase):
    def test_init(self):
        box = pyinterp.geodetic.Box()
        # self.assertEqual(box.min_corner.lon, 0)
        # self.assertEqual(box.min_corner.lat, 0)
        # self.assertEqual(box.max_corner.lon, 0)
        # self.assertEqual(box.max_corner.lat, 0)

        box = pyinterp.geodetic.Box.whole_earth()
        self.assertEqual(box.min_corner.lon, -180)
        self.assertEqual(box.min_corner.lat, -90)
        self.assertEqual(box.max_corner.lon, 180)
        self.assertEqual(box.max_corner.lat, 90)

        box = pyinterp.geodetic.Box(pyinterp.geodetic.Point(1, 2),
                                    pyinterp.geodetic.Point(3, 4))
        self.assertEqual(box.min_corner.lon, 1)
        self.assertEqual(box.min_corner.lat, 2)
        self.assertEqual(box.max_corner.lon, 3)
        self.assertEqual(box.max_corner.lat, 4)

    def test_deprecated(self):
        with self.assertWarns(PendingDeprecationWarning):
            pyinterp.geodetic.Box2D()


class Polygon(unittest.TestCase):
    def test_init(self):
        outer = [
            pyinterp.geodetic.Point(0, 0),
            pyinterp.geodetic.Point(0, 5),
            pyinterp.geodetic.Point(5, 5),
            pyinterp.geodetic.Point(5, 0),
            pyinterp.geodetic.Point(0, 0)
        ]
        polygon = pyinterp.geodetic.Polygon(outer)
        assert polygon.wkt() == 'POLYGON((0 0,0 5,5 5,5 0,0 0))'
        inners = [
            pyinterp.geodetic.Point(1, 1),
            pyinterp.geodetic.Point(4, 1),
            pyinterp.geodetic.Point(4, 4),
            pyinterp.geodetic.Point(1, 4),
            pyinterp.geodetic.Point(1, 1)
        ]
        polygon = pyinterp.geodetic.Polygon(outer, [inners])
        assert polygon.wkt() == \
            'POLYGON((0 0,0 5,5 5,5 0,0 0),(1 1,4 1,4 4,1 4,1 1))'

        assert pyinterp.geodetic.Polygon.read_wkt(
            'POLYGON((0 0,0 5,5 5,5 0,0 0),(1 1,4 1,4 4,1 4,1 1))').wkt(
            ) == 'POLYGON((0 0,0 5,5 5,5 0,0 0),(1 1,4 1,4 4,1 4,1 1))'

        with self.assertRaises(ValueError):
            inners.append(5)
            polygon = pyinterp.geodetic.Polygon(outer, [inners])

        with self.assertRaises(TypeError):
            polygon = pyinterp.geodetic.Polygon(outer, [1])

        with self.assertRaises(ValueError):
            polygon = pyinterp.geodetic.Polygon([1])


if __name__ == "__main__":
    unittest.main()

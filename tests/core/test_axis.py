# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pickle
import unittest
import numpy as np
import pyinterp.core as core

MERCATOR_LATITUDES = np.array([
    -89.000000, -88.908818, -88.809323, -88.700757, -88.582294, -88.453032,
    -88.311987, -88.158087, -87.990161, -87.806932, -87.607008, -87.388869,
    -87.150861, -86.891178, -86.607851, -86.298736, -85.961495, -85.593582,
    -85.192224, -84.754402, -84.276831, -83.755939, -83.187844, -82.568330,
    -81.892820, -81.156357, -80.353575, -79.478674, -78.525397, -77.487013,
    -76.356296, -75.125518, -73.786444, -72.330344, -70.748017, -69.029837,
    -67.165823, -65.145744, -62.959262, -60.596124, -58.046413, -55.300856,
    -52.351206, -49.190700, -45.814573, -42.220632, -38.409866, -34.387043,
    -30.161252, -25.746331, -21.161107, -16.429384, -11.579629, -6.644331,
    -1.659041, 3.338836, 8.311423, 13.221792, 18.035297, 22.720709, 27.251074,
    31.604243, 35.763079, 39.715378, 43.453560, 46.974192, 50.277423,
    53.366377, 56.246554, 58.925270, 61.411164, 63.713764, 65.843134,
    67.809578, 69.623418, 71.294813, 72.833637, 74.249378, 75.551083,
    76.747318, 77.846146, 78.855128, 79.781321, 80.631294, 81.411149,
    82.126535, 82.782681, 83.384411, 83.936179, 84.442084, 84.905904,
    85.331111, 85.720897, 86.078198, 86.405707, 86.705898, 86.981044,
    87.233227, 87.464359, 87.676195, 87.870342, 88.048275, 88.211348,
    88.360799, 88.497766, 88.623291, 88.738328, 88.843755, 88.940374
],
                              dtype=np.float64)


class TextAxis(unittest.TestCase):
    def test_accessor(self):
        lon = np.linspace(0, 359, 360)
        a = core.Axis(lon, is_circle=False, is_radian=False)
        b = core.Axis(0, 359, 360, is_circle=False, is_radian=False)
        self.assertEqual(a, b)
        self.assertFalse(a != b)
        self.assertEqual(str(a), str(b))
        self.assertEqual(
            str(a), "Axis([0, ..., 359], is_circle=false, is_radian=false)")
        self.assertEqual(a.increment(), 1)
        self.assertTrue(a.is_ascending())
        self.assertEqual(a.front(), 0)
        self.assertEqual(a.back(), 359)
        self.assertTrue(
            np.all(
                a.find_index(np.arange(0, 359, 360) +
                             0.01) == np.arange(0, 359, 360)))
        self.assertTrue(a.is_regular())
        self.assertEqual(a.min_value(), 0)
        self.assertEqual(a.max_value(), 359)
        self.assertEqual(a[0], 0)
        self.assertTrue(np.all(a[:] == np.arange(0, 360)))
        self.assertEqual(len(a), 360)

    def test_pickle(self):
        a = core.Axis(0, 359, 360, is_circle=False, is_radian=False)
        b = pickle.loads(pickle.dumps(a))
        self.assertEqual(a, b)

        a = core.Axis(MERCATOR_LATITUDES, is_circle=False, is_radian=False)
        b = pickle.loads(pickle.dumps(a))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
# Copyright (c) 2020 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import datetime
import unittest
import numpy as np
import pyinterp


class TemporalAxis(unittest.TestCase):
    def test_constructor(self):
        start = datetime.datetime(2000, 1, 1)
        values = np.array([
            start + datetime.timedelta(seconds=index) for index in range(86400)
        ],
                          dtype="datetime64[us]")
        axis = pyinterp.TemporalAxis(values)
        self.assertTrue(isinstance(str(axis), str))
        self.assertEqual(axis.increment(), np.timedelta64(1000000, "us"))
        self.assertEqual(axis.front(), np.datetime64('2000-01-01'))
        self.assertEqual(axis.back(), np.datetime64('2000-01-01T23:59:59'))
        self.assertEqual(axis[0], np.datetime64('2000-01-01'))
        self.assertEqual(axis.min_value(), np.datetime64('2000-01-01'))
        self.assertEqual(axis.max_value(),
                         np.datetime64('2000-01-01T23:59:59'))
        self.assertTrue(
            np.all(
                axis.find_index(np.array(['2000-01-01', '2000-02-01'],
                                         dtype="datetime64"),
                                bounded=False) == [0, -1]))
        self.assertTrue(
            np.all(
                axis.find_index(np.array(['2000-01-01', '2000-02-01'],
                                         dtype="datetime64"),
                                bounded=True) == [0, 86399]))
        axis.flip(inplace=True)
        self.assertEqual(axis.increment(), np.timedelta64(-1000000, "us"))
        self.assertEqual(axis.back(), np.datetime64('2000-01-01'))
        self.assertEqual(axis.front(), np.datetime64('2000-01-01T23:59:59'))
        self.assertEqual(axis.min_value(), np.datetime64('2000-01-01'))
        self.assertEqual(axis.max_value(),
                         np.datetime64('2000-01-01T23:59:59'))
        self.assertTrue(
            np.all(
                axis.find_index(np.array(['2000-01-01', '2000-02-01'],
                                         dtype="datetime64"),
                                bounded=False) == [86399, -1]))

        self.assertTrue(
            np.all(
                axis.find_indexes(
                    np.array(['2000-01-01', '2000-02-01'], dtype="datetime64"))
                == [[86398, 86399], [-1, -1]]))

        axis = pyinterp.TemporalAxis(values.astype("datetime64[s]"))
        with self.assertWarns(UserWarning):
            axis.safe_cast(values)


if __name__ == "__main__":
    unittest.main()

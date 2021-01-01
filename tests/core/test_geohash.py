# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import numpy
import pytest
import pyinterp.core.geohash as geohash

testcases = [["77mkh2hcj7mz", -26.015434642, -26.173663656],
             ["wthnssq3w00x", 29.291182895, 118.331595326],
             ["z3jsmt1sde4r", 51.400326027, 154.228244707],
             ["18ecpnqdg4s1", -86.976900779, -106.90988479],
             ["u90suzhjqv2s", 51.49934315, 23.417648894],
             ["k940p3ewmmyq", -39.365655496, 25.636144008],
             ["6g4wv2sze6ms", -26.934429639, -52.496991862],
             ["jhfyx4dqnczq", -62.123898484, 49.178194037],
             ["j80g4mkqz3z9", -89.442648795, 68.659722351],
             ["hq9z7cjwrcw4", -52.156511416, 13.883626414]]


def test_string_numpy():
    strs = numpy.array([item[0] for item in testcases], dtype="S")
    lons, lats = geohash.decode(strs, round=True)
    assert numpy.all(
        numpy.abs(lons - numpy.array([item[2] for item in testcases])) < 1e-6)
    assert numpy.all(
        numpy.abs(lats - numpy.array([item[1] for item in testcases])) < 1e-6)

    strs = numpy.array([item[0] for item in testcases], dtype="U")
    with pytest.raises(ValueError):
        geohash.decode(strs, round=True)
    strs = numpy.array([item[0] for item in testcases],
                       dtype="S").reshape(5, 2)
    with pytest.raises(ValueError):
        geohash.decode(strs, round=True)
    strs = numpy.array([b"0" * 24], dtype="S")
    with pytest.raises(ValueError):
        geohash.decode(strs, round=True)
    strs = numpy.array([item[0] for item in testcases], dtype="S")
    strs = numpy.vstack((strs[:5], strs[5:]))
    indexes = geohash.where(strs)
    assert isinstance(indexes, dict)

    with pytest.raises(ValueError):
        indexes = geohash.where(strs.astype("U"))

    strs = numpy.array([item[0] for item in testcases], dtype="S")
    strs.reshape(1, 2, 5)
    with pytest.raises(ValueError):
        indexes = geohash.where(strs)

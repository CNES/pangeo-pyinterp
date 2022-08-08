# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import numpy
import pytest

from ...core import GeoHash, geohash

testcases = [['77mkh2hcj7mz', -26.015434642, -26.173663656],
             ['wthnssq3w00x', 29.291182895, 118.331595326],
             ['z3jsmt1sde4r', 51.400326027, 154.228244707],
             ['18ecpnqdg4s1', -86.976900779, -106.90988479],
             ['u90suzhjqv2s', 51.49934315, 23.417648894],
             ['k940p3ewmmyq', -39.365655496, 25.636144008],
             ['6g4wv2sze6ms', -26.934429639, -52.496991862],
             ['jhfyx4dqnczq', -62.123898484, 49.178194037],
             ['j80g4mkqz3z9', -89.442648795, 68.659722351],
             ['hq9z7cjwrcw4', -52.156511416, 13.883626414]]


def test_string_numpy():
    strs = numpy.array([item[0] for item in testcases], dtype='S')
    lons, lats = geohash.decode(strs, round=True)
    assert numpy.all(
        numpy.abs(  # type: ignore
            lons - numpy.array([item[2] for item in testcases])) < 1e-6)
    assert numpy.all(
        numpy.abs(  # type: ignore
            lats - numpy.array([item[1] for item in testcases])) < 1e-6)

    strs = numpy.array([item[0] for item in testcases], dtype='U')
    with pytest.raises(ValueError):
        geohash.decode(strs, round=True)
    strs = numpy.array([item[0] for item in testcases],
                       dtype='S').reshape(5, 2)
    with pytest.raises(ValueError):
        geohash.decode(strs, round=True)
    strs = numpy.array([b'0' * 24], dtype='S')
    with pytest.raises(ValueError):
        geohash.decode(strs, round=True)
    strs = numpy.array([item[0] for item in testcases], dtype='S')
    strs = numpy.vstack((strs[:5], strs[5:]))
    indexes = geohash.where(strs)
    assert isinstance(indexes, dict)

    with pytest.raises(ValueError):
        indexes = geohash.where(strs.astype('U'))

    strs = numpy.array([item[0] for item in testcases], dtype='S')
    strs.reshape(1, 2, 5)
    with pytest.raises(ValueError):
        indexes = geohash.where(strs)


def test_bounding_boxes():
    bboxes = geohash.bounding_boxes(precision=1)
    assert len(bboxes) == 32
    for bbox in bboxes:
        code = GeoHash.from_string(bbox.decode())
        case = geohash.bounding_boxes(code.bounding_box(), precision=1)
        assert len(case) == 1
        assert case[0] == bbox

        case = geohash.bounding_boxes(code.bounding_box(), precision=3)
        assert len(case) == 2**10
        assert all(item.startswith(bbox) for item in case)

    with pytest.raises(MemoryError):
        geohash.bounding_boxes(precision=12)


def test_bounding_zoom():
    bboxes = geohash.bounding_boxes(precision=1)
    assert len(bboxes) == 32

    zoom_in = geohash.transform(bboxes, precision=3)
    assert len(zoom_in) == 2**10 * 32
    assert numpy.all(
        numpy.sort(geohash.transform(zoom_in, precision=1)) == numpy.sort(
            bboxes))


def test_class():
    for code, lat, lon in testcases:
        instance = GeoHash.from_string(code)
        assert str(instance) == code
        point = instance.center()
        assert lat == pytest.approx(point.lat, abs=1e-6)
        assert lon == pytest.approx(point.lon, abs=1e-6)

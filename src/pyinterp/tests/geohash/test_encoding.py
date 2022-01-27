# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import json

import numpy
import pytest

from ... import GeoHash, geohash
from .. import geohash_path


def test_encoding_decoding():
    with open(geohash_path(), 'r') as stream:
        cases = json.load(stream)
    lon = numpy.array([item[3] for item in cases])
    lat = numpy.array([item[2] for item in cases])

    int_hashs = geohash.int64.encode(lon, lat)
    assert numpy.all(
        numpy.array([item[0] for item in cases], dtype='uint64') == int_hashs)
    decoded_lon, decoded_lat = geohash.int64.decode(int_hashs, round=True)
    assert numpy.all(numpy.abs(lat - decoded_lat) < 1e-7)  # type: ignore
    assert numpy.all(numpy.abs(lon - decoded_lon) < 1e-7)  # type: ignore

    str_hashs = geohash.encode(lon, lat)
    assert numpy.all([item[1] for item in cases] == str_hashs.astype("U"))
    decoded_lon, decoded_lat = geohash.decode(str_hashs, round=True)
    assert numpy.all(numpy.abs(lat - decoded_lat) < 1e-6)  # type: ignore
    assert numpy.all(numpy.abs(lon - decoded_lon) < 1e-6)  # type: ignore

    for item in cases:
        code = GeoHash.from_string(item[1])
        point = code.center()
        assert pytest.approx(point.lon, item[3])
        assert pytest.approx(point.lat, item[3])
        assert str(code) == item[1]

# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import json

import numpy
import pytest

from .. import geohash_path
from ... import GeoHash, geohash


def test_encoding_decoding():
    with open(geohash_path()) as stream:
        cases = json.load(stream)
    lon = numpy.array([item[3] for item in cases])
    lat = numpy.array([item[2] for item in cases])

    int_hashs = geohash.int64.encode(lon, lat)
    assert numpy.all(
        numpy.array([item[0] for item in cases], dtype='uint64') == int_hashs)
    decoded_lon, decoded_lat = geohash.int64.decode(int_hashs, round=True)
    assert numpy.all(numpy.abs(lat - decoded_lat) < 1e-7)  # type: ignore
    assert numpy.all(numpy.abs(lon - decoded_lon) < 1e-7)  # type: ignore

    # Encode with longitude [0, 360] and latitude [-90, 90]
    lon_0_360 = lon % 360
    int_hashs = geohash.int64.encode(lon_0_360, lat)
    assert numpy.all(
        numpy.array([item[0] for item in cases], dtype='uint64') == int_hashs)

    str_hashs = geohash.encode(lon, lat)
    assert numpy.all([item[1] for item in cases] == str_hashs.astype('U'))
    decoded_lon, decoded_lat = geohash.decode(str_hashs, round=True)
    assert numpy.all(numpy.abs(lat - decoded_lat) < 1e-6)  # type: ignore
    assert numpy.all(numpy.abs(lon - decoded_lon) < 1e-6)  # type: ignore

    # Encode with longitude [0, 360] and latitude [-90, 90]
    str_hashs = geohash.encode(lon_0_360, lat)
    assert numpy.all([item[1] for item in cases] == str_hashs.astype('U'))

    for item in cases:
        code = GeoHash.from_string(item[1])
        point = code.center()
        assert item[3] == pytest.approx(point.lon, abs=1e-6)
        assert item[2] == pytest.approx(point.lat, abs=1e-6)
        assert str(code) == item[1]

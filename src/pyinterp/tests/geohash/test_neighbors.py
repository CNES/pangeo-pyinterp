# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import json

import numpy as np

from .. import geohash_neighbors_path
from ... import GeoHash, geodetic, geohash


def test_neighbors():
    with open(geohash_neighbors_path()) as stream:
        cases = json.load(stream)

    for (lat, lon, hash_int, bits, hash_int_neighbors, hash_str,
         hash_str_neighbors) in cases:
        point = geodetic.Point(lon, lat)
        hash = geohash.int64.encode(np.array([lon], dtype='float64'),
                                    np.array([lat], dtype='float64'), bits)
        assert hash_int == hash[0]
        assert list(geohash.int64.neighbors(hash_int,
                                            bits)) == hash_int_neighbors

        code = GeoHash(lon, lat, len(hash_str))
        assert [str(item) for item in code.neighbors()] == hash_str_neighbors

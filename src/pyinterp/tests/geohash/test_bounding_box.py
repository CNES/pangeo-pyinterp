# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import json

from .. import geohash_bbox_path
from ... import GeoHash, geodetic


def test_bbox():
    with open(geohash_bbox_path()) as stream:
        cases = json.load(stream)
    for hash_str, (min_lat, max_lat, min_lng, max_lng) in cases:
        point = GeoHash.from_string(hash_str).center()
        assert geodetic.Box(geodetic.Point(min_lng, min_lat),
                            geodetic.Point(max_lng, max_lat)).covered_by(point)

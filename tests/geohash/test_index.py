# Copyright (c) 2020 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pytest
import numpy
import pyinterp.geohash as geohash
import pyinterp.geodetic as geodetic


def test_index():
    # Create dummy data and populate the index
    data = dict(
        (key, key) for key in geohash.string.bounding_boxes(precision=3))
    store = geohash.storage.UnQlite(":mem:", mode="w")
    idx = geohash.index.init_geohash(store)
    assert(len(idx) == 0)
    idx.update(data)
    assert(len(idx) == 32768)
    assert idx.store == store
    assert str(idx) == "<GeoHash precision=3>"
    assert idx.encode(numpy.array([185.874942713]),
                      numpy.array([-84.529178182]),
                      normalize=True)[0] == b"00u"
    assert idx.encode(numpy.array([185.874942713]),
                      numpy.array([-84.529178182]),
                      normalize=False)[0] == b"00f"

    box = geodetic.Box(geodetic.Point(-40, -40), geodetic.Point(40, 40))
    boxes = list(geohash.string.bounding_boxes(box, precision=3))
    assert idx.box(box) == boxes

    idx.extend({b"00u": (None, )})
    assert idx.store[b"00u"] == [b"00u", (None, )]

    with pytest.raises(RuntimeError):
        idx = geohash.index.init_geohash(store)

    idx = geohash.index.open_geohash(store)
    assert idx.precision == 3

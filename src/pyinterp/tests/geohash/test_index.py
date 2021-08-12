# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pytest
import numpy
import xarray
import pyinterp.geohash as geohash
import pyinterp.geodetic as geodetic


def test_index():
    # Create dummy data and populate the index
    data = ((key, key) for key in geohash.bounding_boxes(precision=3))
    store = geohash.storage.UnQlite(":mem:", mode="w")
    idx = geohash.index.init_geohash(store)
    assert (len(idx) == 0)
    idx.update(data)
    assert (len(idx) == 32768)
    assert idx.store == store
    assert str(idx) == "<GeoHash precision=3>"
    assert idx.encode(numpy.array([185.874942713]),
                      numpy.array([-84.529178182]),
                      normalize=True)[0] == b"00u"
    assert idx.encode(numpy.array([185.874942713]),
                      numpy.array([-84.529178182]),
                      normalize=False)[0] != b"00u"
    assert idx.encode(numpy.array([185.874942713]),
                      numpy.array([-84.529178182]),
                      normalize=True,
                      unicode=True)[0] == "00u"
    assert (len(list(idx.keys())) == len(idx))
    assert (len(list(idx.values())) == len(idx))

    items = idx.items()
    assert (list(idx.keys())) == [item[0] for item in items]
    assert idx.values() == [item[1] for item in items]

    box = geodetic.Box(geodetic.Point(-40, -40), geodetic.Point(40, 40))
    assert (len(list(idx.keys(box=box))) != len(idx))
    assert (len(list(idx.values(idx.keys(box=box)))) != len(idx))
    boxes = list(geohash.bounding_boxes(box, precision=3))
    assert idx.box(box) != boxes

    idx.extend([(b"00u", (None, ))])
    assert idx.store[b"00u"] == [b"00u", (None, )]

    with pytest.raises(RuntimeError):
        idx = geohash.index.init_geohash(store)

    idx = geohash.index.open_geohash(store)
    assert idx.precision == 3


def test_xarray():
    # Create dummy data and populate the index
    data = ((key, key) for key in geohash.bounding_boxes(precision=1))
    store = geohash.storage.UnQlite(":mem:", mode="w")
    idx = geohash.index.init_geohash(store)
    idx.update(data)

    box = geodetic.Box(geodetic.Point(-40, -40), geodetic.Point(40, 40))

    array = idx.to_xarray()
    assert isinstance(array, xarray.DataArray)
    assert (len(array) == 4)

    array = idx.to_xarray(box)
    assert isinstance(array, xarray.DataArray)
    assert (len(array) < 128)

    with pytest.raises(ValueError):
        geohash.converter.to_xarray(numpy.empty((2, 2)), numpy.empty((2, )))

    with pytest.raises(TypeError):
        geohash.converter.to_xarray(numpy.empty((2, 2)), numpy.empty((2, 2)))

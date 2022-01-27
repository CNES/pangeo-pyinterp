# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pytest

from .. import geodetic


def test_system():
    wgs = geodetic.System()
    assert isinstance(wgs, geodetic.System)
    with pytest.raises(TypeError):
        wgs = geodetic.System(12.0)  # type: ignore
    with pytest.raises(TypeError):
        wgs = geodetic.System((12.0, 3.0, 5))  # type: ignore
    wgs = geodetic.System((1, 1))
    assert isinstance(wgs, geodetic.System)
    assert str(wgs) == "System(1.0, 1.0)"


def test_coordinates():
    wgs = geodetic.Coordinates()
    assert isinstance(wgs, geodetic.Coordinates)
    wgs = geodetic.Coordinates(geodetic.System())
    assert isinstance(wgs, geodetic.Coordinates)


def test_point():
    pt = geodetic.Point(1, 2)
    assert pt.lon == 1
    assert pt.lat == 2


def test_box():
    box = geodetic.Box()
    assert isinstance(box, geodetic.Box)

    box = geodetic.Box.whole_earth()
    assert box.min_corner.lon == -180
    assert box.min_corner.lat == -90
    assert box.max_corner.lon == 180
    assert box.max_corner.lat == 90

    box = geodetic.Box(geodetic.Point(1, 2), geodetic.Point(3, 4))
    assert box.min_corner.lon == 1
    assert box.min_corner.lat == 2
    assert box.max_corner.lon == 3
    assert box.max_corner.lat == 4


def test_polygon():
    outer = [
        geodetic.Point(0, 0),
        geodetic.Point(0, 5),
        geodetic.Point(5, 5),
        geodetic.Point(5, 0),
        geodetic.Point(0, 0)
    ]
    polygon = geodetic.Polygon(outer)
    assert polygon.wkt() == 'POLYGON((0 0,0 5,5 5,5 0,0 0))'
    inners = [
        geodetic.Point(1, 1),
        geodetic.Point(4, 1),
        geodetic.Point(4, 4),
        geodetic.Point(1, 4),
        geodetic.Point(1, 1)
    ]
    polygon = geodetic.Polygon(outer, [inners])
    assert polygon.wkt() == \
        'POLYGON((0 0,0 5,5 5,5 0,0 0),(1 1,4 1,4 4,1 4,1 1))'

    assert geodetic.Polygon.read_wkt(
        'POLYGON((0 0,0 5,5 5,5 0,0 0),(1 1,4 1,4 4,1 4,1 1))').wkt(
        ) == 'POLYGON((0 0,0 5,5 5,5 0,0 0),(1 1,4 1,4 4,1 4,1 1))'

    with pytest.raises(ValueError):
        inners.append(5)  # type: ignore
        polygon = geodetic.Polygon(outer, [inners])

    with pytest.raises(TypeError):
        polygon = geodetic.Polygon(outer, [1])  # type: ignore

    with pytest.raises(ValueError):
        polygon = geodetic.Polygon([1])  # type: ignore

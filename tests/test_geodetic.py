# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pytest
import pyinterp.geodetic


def test_system():
    wgs = pyinterp.geodetic.System()
    assert isinstance(wgs, pyinterp.geodetic.System)
    with pytest.raises(TypeError):
        wgs = pyinterp.geodetic.System(12.0)
    with pytest.raises(TypeError):
        wgs = pyinterp.geodetic.System((12.0, 3.0, 5))
    wgs = pyinterp.geodetic.System((1, 1))
    assert isinstance(wgs, pyinterp.geodetic.System)
    assert str(wgs) == "System(1.0, 1.0)"


def test_coordinates():
    wgs = pyinterp.geodetic.Coordinates()
    assert isinstance(wgs, pyinterp.geodetic.Coordinates)
    wgs = pyinterp.geodetic.Coordinates(pyinterp.geodetic.System())
    assert isinstance(wgs, pyinterp.geodetic.Coordinates)


def test_point():
    pt = pyinterp.geodetic.Point(1, 2)
    assert pt.lon == 1
    assert pt.lat == 2


def test_point_2d():
    with pytest.warns(PendingDeprecationWarning):
        pyinterp.geodetic.Point2D()


def test_box():
    box = pyinterp.geodetic.Box()
    assert isinstance(box, pyinterp.geodetic.Box)

    box = pyinterp.geodetic.Box.whole_earth()
    assert box.min_corner.lon == -180
    assert box.min_corner.lat == -90
    assert box.max_corner.lon == 180
    assert box.max_corner.lat == 90

    box = pyinterp.geodetic.Box(pyinterp.geodetic.Point(1, 2),
                                pyinterp.geodetic.Point(3, 4))
    assert box.min_corner.lon == 1
    assert box.min_corner.lat == 2
    assert box.max_corner.lon == 3
    assert box.max_corner.lat == 4


def test_box2d():
    with pytest.warns(PendingDeprecationWarning):
        pyinterp.geodetic.Box2D()


def test_polygon():
    outer = [
        pyinterp.geodetic.Point(0, 0),
        pyinterp.geodetic.Point(0, 5),
        pyinterp.geodetic.Point(5, 5),
        pyinterp.geodetic.Point(5, 0),
        pyinterp.geodetic.Point(0, 0)
    ]
    polygon = pyinterp.geodetic.Polygon(outer)
    assert polygon.wkt() == 'POLYGON((0 0,0 5,5 5,5 0,0 0))'
    inners = [
        pyinterp.geodetic.Point(1, 1),
        pyinterp.geodetic.Point(4, 1),
        pyinterp.geodetic.Point(4, 4),
        pyinterp.geodetic.Point(1, 4),
        pyinterp.geodetic.Point(1, 1)
    ]
    polygon = pyinterp.geodetic.Polygon(outer, [inners])
    assert polygon.wkt() == \
        'POLYGON((0 0,0 5,5 5,5 0,0 0),(1 1,4 1,4 4,1 4,1 1))'

    assert pyinterp.geodetic.Polygon.read_wkt(
        'POLYGON((0 0,0 5,5 5,5 0,0 0),(1 1,4 1,4 4,1 4,1 1))').wkt(
        ) == 'POLYGON((0 0,0 5,5 5,5 0,0 0),(1 1,4 1,4 4,1 4,1 1))'

    with pytest.raises(ValueError):
        inners.append(5)
        polygon = pyinterp.geodetic.Polygon(outer, [inners])

    with pytest.raises(TypeError):
        polygon = pyinterp.geodetic.Polygon(outer, [1])

    with pytest.raises(ValueError):
        polygon = pyinterp.geodetic.Polygon([1])

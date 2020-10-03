# Copyright (c) 2020 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pickle
import pytest
import math
import numpy as np
from pyinterp import core


def test_system_wgs84():
    """Checking expected WGS-84 properties"""
    wgs84 = core.geodetic.System()
    # https://fr.wikipedia.org/wiki/WGS_84
    # https://en.wikipedia.org/wiki/Geodetic_datum
    # http://earth-info.nga.mil/GandG/publications/tr8350.2/wgs84fin.pdf
    assert 6378137 == pytest.approx(wgs84.semi_major_axis)
    assert 1 / 298.257223563 == pytest.approx(wgs84.flattening)
    assert 6356752.314245179497563967 == pytest.approx(wgs84.semi_minor_axis())
    assert 0.081819190842622 == pytest.approx(math.sqrt(
        wgs84.first_eccentricity_squared()),
                                              abs=1e-15)
    assert 8.2094437949696 * 1e-2 == pytest.approx(math.sqrt(
        wgs84.second_eccentricity_squared()),
                                                   abs=1e-15)
    assert 40075.017 == pytest.approx(wgs84.equatorial_circumference() * 1e-3,
                                      abs=1e-3)
    assert 39940.652 == pytest.approx(wgs84.equatorial_circumference(False) *
                                      1e-3,
                                      abs=1e-3)
    assert 6399593.6258 == pytest.approx(wgs84.polar_radius_of_curvature(),
                                         abs=1e-4)
    assert 6335439.3272 == pytest.approx(
        wgs84.equatorial_radius_of_curvature(), abs=1e-4)
    assert 0.996647189335 == pytest.approx(wgs84.axis_ratio(), abs=1e-12)
    assert 5.2185400842339 * 1E5 == pytest.approx(wgs84.linear_eccentricity(),
                                                  abs=1e-6)
    assert 6371008.7714 == pytest.approx(wgs84.mean_radius(), abs=1e-4)
    assert 6371007.1809 == pytest.approx(wgs84.authalic_radius(), abs=1e-4)
    assert 6371000.7900 == pytest.approx(wgs84.volumetric_radius(), abs=1e-4)


def test_system_operators():
    """Test operators"""
    wgs84 = core.geodetic.System()
    # https://en.wikipedia.org/wiki/Geodetic_Reference_System_1980
    grs80 = core.geodetic.System(6378137, 1 / 298.257222101)
    assert 6378137 == pytest.approx(grs80.semi_major_axis)
    assert 1 / 298.257222101 == pytest.approx(grs80.flattening)
    assert wgs84 == wgs84
    assert wgs84 != grs80


def test_system_pickle():
    """Serialization test"""
    wgs84 = core.geodetic.System()
    assert wgs84 == pickle.loads(pickle.dumps(wgs84))


def test_coordinates_ecef_lla():
    """ECEF/LLA Conversion Test"""
    lon, lat, alt = core.geodetic.Coordinates(None).ecef_to_lla(
        [1176498.769459714], [5555043.905503586], [2895446.8901510699])
    assert lon[0] == pytest.approx(78.042068, abs=1e-8)
    assert lat[0] == pytest.approx(27.173891, abs=1e-8)
    assert alt[0] == pytest.approx(168.0, abs=1e-8)


def test_coordinates_lla_to_ecef():
    """LLA/ECEF Conversion Test"""
    x, y, z = core.geodetic.Coordinates(None).lla_to_ecef([78.042068],
                                                          [27.173891], [168.0])
    assert x[0] == pytest.approx(1176498.769459714, abs=1e-8)
    assert y[0] == pytest.approx(5555043.905503586, abs=1e-8)
    assert z[0] == pytest.approx(2895446.8901510699, abs=1e-8)


def test_coordinates_round_trip():
    """Check algorithm precision"""
    lon1 = np.random.uniform(-180.0, 180.0, 1000000)
    lat1 = np.random.uniform(-90.0, 90.0, 1000000)
    alt1 = np.random.uniform(-10000, 100000, 1000000)

    a = core.geodetic.Coordinates(None)
    b = core.geodetic.Coordinates(None)

    lon2, lat2, alt2 = a.transform(b, lon1, lat1, alt1, num_threads=0)

    assert 0 == pytest.approx((lon1 - lon2).mean(), abs=1e-12)
    assert 0 == pytest.approx((lat1 - lat2).mean(), abs=1e-12)
    assert 0 == pytest.approx((alt1 - alt2).mean(), abs=1e-10)


def test_coordinates_pickle():
    """Serialization test"""
    a = core.geodetic.Coordinates(None)
    b = pickle.loads(pickle.dumps(a))
    assert np.all(a.__getstate__() == b.__getstate__())


def test_point():
    """Test construction and accessors of the object"""
    pt = core.geodetic.Point(12, 24)
    assert pt.lon == 12
    assert pt.lat == 24
    assert str(pt) == "(12, 24)"
    assert repr(pt) == "(12, 24)"
    pt.lon = 55
    assert pt.lon == 55
    pt.lat = 33
    assert pt.lat == 33
    point = core.geodetic.Point.read_wkt("POINT(-2 2)")
    assert point.wkt() == "POINT(-2 2)"


def test_point_pickle():
    """Serialization tests"""
    a = core.geodetic.Point(1, 2)
    b = pickle.loads(pickle.dumps(a))
    assert a.lon == b.lon
    assert a.lat == b.lat
    assert a == b
    assert not a != b
    assert id(a) != id(b)


def test_box():
    """Test construction and accessors of the object"""
    min_corner = core.geodetic.Point(0, 1)
    max_corner = core.geodetic.Point(2, 3)

    box = core.geodetic.Box(min_corner, max_corner)
    assert str(box) == "((0, 1), (2, 3))"
    assert box.min_corner.lon == 0
    assert box.min_corner.lat == 1
    assert box.max_corner.lon == 2
    assert box.max_corner.lat == 3

    assert box.covered_by(min_corner)
    assert box.covered_by(max_corner)
    assert box.covered_by(core.geodetic.Point(1, 2))
    assert not box.covered_by(core.geodetic.Point(0, 0))

    flags = box.covered_by([1, 0], [2, 0])
    assert np.all(flags == [1, 0])

    box.min_corner, box.max_corner = max_corner, min_corner
    assert box.min_corner.lon == 2
    assert box.min_corner.lat == 3
    assert box.max_corner.lon == 0
    assert box.max_corner.lat == 1

    assert box.wkt() == "POLYGON((2 3,2 1,0 1,0 3,2 3))"
    box = core.geodetic.Box.read_wkt("POLYGON((2 3,2 1,0 1,0 3,2 3))")
    assert repr(box) == "((2, 3), (0, 1))"

    box = core.geodetic.Box.whole_earth()
    assert repr(box) == "((-180, -90), (180, 90))"


def test_box_pickle():
    """Serialization tests"""
    min_corner = core.geodetic.Point(0, 1)
    max_corner = core.geodetic.Point(2, 3)
    a = core.geodetic.Box(min_corner, max_corner)
    b = pickle.loads(pickle.dumps(a))
    assert a.min_corner.lon == b.min_corner.lon
    assert a.min_corner.lat == b.min_corner.lat
    assert a.max_corner.lon == b.max_corner.lon
    assert a.max_corner.lat == b.max_corner.lat
    assert a == b
    assert not a != b


def test_polygon():
    polygon = core.geodetic.Polygon.read_wkt("POLYGON((0 0,0 7,4 2,2 0,0 0))")
    assert repr(polygon) == "(((0, 0), (0, 7), (4, 2), (2, 0), (0, 0)))"
    assert polygon.envelope() == core.geodetic.Box(core.geodetic.Point(0, 0),
                                                   core.geodetic.Point(4, 7))
    polygon = core.geodetic.Polygon([
        core.geodetic.Point(0, 0),
        core.geodetic.Point(0, 5),
        core.geodetic.Point(5, 5),
        core.geodetic.Point(5, 0),
        core.geodetic.Point(0, 0)
    ], [[
        core.geodetic.Point(1, 1),
        core.geodetic.Point(4, 1),
        core.geodetic.Point(4, 4),
        core.geodetic.Point(1, 4),
        core.geodetic.Point(1, 1)
    ]])
    assert repr(polygon) == "(((0, 0), (0, 5), (5, 5), (5, 0), (0, 0)), " \
        "((1, 1), (4, 1), (4, 4), (1, 4), (1, 1)))"
    assert polygon.wkt() == "POLYGON((0 0,0 5,5 5,5 0,0 0)," \
        "(1 1,4 1,4 4,1 4,1 1))"


def test_polygon_pickle():
    for item in [
            "POLYGON((0 0,0 7,4 2,2 0,0 0))",
            "POLYGON((0 0,0 5,5 5,5 0,0 0),(1 1,4 1,4 4,1 4,1 1))"
    ]:
        polygon = core.geodetic.Polygon.read_wkt(item)
        other = pickle.loads(pickle.dumps(polygon))
        assert polygon == other
        assert not polygon != other
        assert id(polygon) != id(other)

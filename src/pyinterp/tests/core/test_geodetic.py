# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import json
import math
import pickle

import numpy as np
import pytest

from .. import polygon_path
from ... import core
from ...core import geodetic


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
        np.array([1176498.769459714]), np.array([5555043.905503586]),
        np.array([2895446.8901510699]))
    assert lon[0] == pytest.approx(78.042068, abs=1e-8)
    assert lat[0] == pytest.approx(27.173891, abs=1e-8)
    assert alt[0] == pytest.approx(168.0, abs=1e-8)


def test_coordinates_lla_to_ecef():
    """LLA/ECEF Conversion Test"""
    x, y, z = core.geodetic.Coordinates(None).lla_to_ecef(
        np.array([78.042068]), np.array([27.173891]), np.array([168.0]))
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


def test_point_distance():
    acropolis = core.geodetic.Point(23.725750, 37.971536)
    ulb = core.geodetic.Point(4.3826169, 50.8119483)
    assert 2088389.07865908 == pytest.approx(acropolis.distance(
        ulb, strategy="andoyer"),
                                             abs=1e-6)
    assert 2088384.36439399 == pytest.approx(acropolis.distance(
        ulb, strategy="thomas"),
                                             abs=1e-6)
    assert 2088384.36606831 == pytest.approx(acropolis.distance(
        ulb, strategy="vincenty"),
                                             abs=1e-6)
    assert acropolis.distance(ulb,
                              strategy="thomas") == acropolis.distance(ulb)
    with pytest.raises(ValueError):
        acropolis.distance(ulb, strategy="Thomas")


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

    assert box.distance(box) == 0
    assert box.distance(min_corner) == 0
    assert box.distance(core.geodetic.Point(1, 1)) != 0

    assert box.covered_by(min_corner)
    assert box.covered_by(max_corner)
    assert box.covered_by(core.geodetic.Point(1, 2))
    assert not box.covered_by(core.geodetic.Point(0, 0))

    flags = box.covered_by(np.array([1, 0]), np.array([2, 0]))
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
    assert polygon.distance(polygon) == 0
    assert polygon.distance(core.geodetic.Point(0, 0)) == 0
    assert polygon.distance(core.geodetic.Point(10, 10)) != 0

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


def test_polygon_covered_by():
    with open(polygon_path()) as stream:
        points = json.load(stream)
    lon = np.arange(0, 360, 10)
    lat = np.arange(-90, 90.5, 10)
    mx, my = np.meshgrid(lon, lat)
    polygon = core.geodetic.Polygon(
        [core.geodetic.Point(*item) for item in points])
    mask1 = polygon.covered_by(mx.ravel(), my.ravel()).reshape(mx.shape)
    mask2 = polygon.covered_by(mx.ravel(), my.ravel(),
                               num_threads=1).reshape(mx.shape)
    assert np.all(mask2 == mask1)
    ix, iy = np.where(mask1 == 1)
    assert np.all(ix == [
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7,
        7, 7, 7, 7, 7, 8, 8, 8, 8, 8
    ])
    assert np.all(iy == [
        0, 1, 30, 31, 32, 33, 34, 35, 0, 1, 30, 31, 32, 33, 34, 35, 0, 1, 31,
        32, 33, 34, 35, 0, 1, 32, 33, 34, 35, 0, 1, 33, 34, 35
    ])
    # Switch to [-180, 180[ input range
    mx = (mx + 180) % 360 - 180
    mask2 = polygon.covered_by(mx.ravel(), my.ravel()).reshape(mx.shape)
    assert np.all(mask2 == mask1)


def test_coordinate_distance():
    lon = np.arange(0, 360, 10)
    lat = np.arange(-90, 90.5, 10)
    mx, my = np.meshgrid(lon, lat)
    d1 = core.geodetic.coordinate_distances(mx.ravel(),
                                            my.ravel(),
                                            mx.ravel() + 1,
                                            my.ravel() + 1,
                                            strategy="vincenty",
                                            num_threads=1)
    d0 = core.geodetic.coordinate_distances(mx.ravel(),
                                            my.ravel(),
                                            mx.ravel() + 1,
                                            my.ravel() + 1,
                                            strategy="vincenty",
                                            num_threads=0)
    assert np.all(d0 == d1)
    d0 = d0.reshape(mx.shape)
    for iy in range(d0.shape[0]):
        assert np.all(np.abs((d0[iy, :] - d0[iy, 0]) <= 1e-6))
    for ix in range(d0.shape[1]):
        delta = np.abs(d0[:, ix] - d0[0, ix])
        assert np.all(delta[delta != 0] > 1e3)


def test_crossover():
    """Calculate the location of a crossover"""
    lon1 = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.float64)
    lat1 = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.float64)
    lon2 = np.array(lon1[:])
    lat2 = np.array(lat1[::-1])

    crossover = core.geodetic.Crossover(core.geodetic.Linestring(lon1, lat1),
                                        core.geodetic.Linestring(lon2, lat2))
    assert isinstance(crossover, core.geodetic.Crossover)
    assert crossover.exists()
    coordinates = crossover.search()
    assert coordinates is not None
    assert pytest.approx(coordinates.lon) == 4
    assert pytest.approx(coordinates.lat, rel=1e-3) == 4.0018282189756835
    assert crossover.nearest(coordinates) == (4, 3)

    other = pickle.loads(pickle.dumps(crossover))
    assert list(crossover.half_orbit_1) == list(other.half_orbit_1)
    assert list(crossover.half_orbit_2) == list(other.half_orbit_2)


def test_merged_point():
    """Try to compute a crossover from overlay tracks"""
    lon1 = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.float64)
    lat1 = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.float64)

    crossover = core.geodetic.Crossover(core.geodetic.Linestring(lon1, lat1),
                                        core.geodetic.Linestring(lon1, lat1))
    assert isinstance(crossover, core.geodetic.Crossover)
    assert crossover.exists()

    with pytest.raises(RuntimeError):
        crossover.search()

    crossover = core.geodetic.Crossover(
        core.geodetic.Linestring(lon1, lat1),
        core.geodetic.Linestring(lon1 + 10, lat1 + 10))
    assert not crossover.exists()
    assert crossover.search() is None


def test_missing_crossover():
    """Try to calculate a crossing point when the entry passes do not cross.
    """
    x1 = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.float64)
    y1 = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.float64)
    x2 = np.flip(x1, axis=0)
    y2 = np.flip(x2, axis=0)

    crossover = core.geodetic.Crossover(core.geodetic.Linestring(x1, y1),
                                        core.geodetic.Linestring(x2, y2))
    assert isinstance(crossover, core.geodetic.Crossover)
    assert crossover.exists()


def test_case_crossover_shift():
    """Calculate a crossover between two tracks."""
    lon1 = np.array([300, 350, 40, 90], dtype=np.float64)
    lat1 = np.array([0, 1, 2, 3], dtype=np.float64)
    lon2 = np.array(lon1[:])
    lat2 = np.array(lat1[::-1])

    crossover = core.geodetic.Crossover(core.geodetic.Linestring(lon1, lat1),
                                        core.geodetic.Linestring(lon2, lat2))
    assert isinstance(crossover, core.geodetic.Crossover)
    assert crossover.exists()
    coordinates = crossover.search()
    assert coordinates is not None
    assert pytest.approx(coordinates.lon) == 15
    assert pytest.approx(coordinates.lat, rel=1e-3) == 1.6551107341906504
    assert crossover.nearest(coordinates, predicate=None) == (2, 1)

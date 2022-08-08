# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import json
import math
import pickle

import numpy as np
import pytest

from .. import multipolygon_path, polygon_path
from ... import core
from ...core import geodetic


def test_system_wgs84():
    """Checking expected WGS-84 properties."""
    wgs84 = core.geodetic.Spheroid()
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
    """Test operators."""
    wgs84 = core.geodetic.Spheroid()
    # https://en.wikipedia.org/wiki/Geodetic_Reference_System_1980
    grs80 = core.geodetic.Spheroid(6378137, 1 / 298.257222101)
    assert 6378137 == pytest.approx(grs80.semi_major_axis)
    assert 1 / 298.257222101 == pytest.approx(grs80.flattening)
    assert wgs84 == wgs84
    assert wgs84 != grs80


def test_system_pickle():
    """Serialization test."""
    wgs84 = core.geodetic.Spheroid()
    assert wgs84 == pickle.loads(pickle.dumps(wgs84))


def test_coordinates_ecef_lla():
    """ECEF/LLA Conversion Test."""
    lon, lat, alt = core.geodetic.Coordinates(None).ecef_to_lla(
        np.array([1176498.769459714]), np.array([5555043.905503586]),
        np.array([2895446.8901510699]))
    assert lon[0] == pytest.approx(78.042068, abs=1e-8)
    assert lat[0] == pytest.approx(27.173891, abs=1e-8)
    assert alt[0] == pytest.approx(168.0, abs=1e-8)


def test_coordinates_lla_to_ecef():
    """LLA/ECEF Conversion Test."""
    x, y, z = core.geodetic.Coordinates(None).lla_to_ecef(
        np.array([78.042068]), np.array([27.173891]), np.array([168.0]))
    assert x[0] == pytest.approx(1176498.769459714, abs=1e-8)
    assert y[0] == pytest.approx(5555043.905503586, abs=1e-8)
    assert z[0] == pytest.approx(2895446.8901510699, abs=1e-8)


def test_coordinates_round_trip():
    """Check algorithm precision."""
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
    """Serialization test."""
    a = core.geodetic.Coordinates(None)
    b = pickle.loads(pickle.dumps(a))
    assert np.all(a.__getstate__() == b.__getstate__())


def test_point():
    """Test construction and accessors of the object."""
    pt = core.geodetic.Point(12, 24)
    assert pt.lon == 12
    assert pt.lat == 24
    assert str(pt) == '(12, 24)'
    assert repr(pt) == '(12, 24)'
    pt.lon = 55
    assert pt.lon == 55
    pt.lat = 33
    assert pt.lat == 33
    point = core.geodetic.Point.read_wkt('POINT(-2 2)')
    assert point.wkt() == 'POINT(-2 2)'
    assert point.to_geojson() == {'type': 'Point', 'coordinates': [-2, 2]}


def test_point_distance():
    acropolis = core.geodetic.Point(23.725750, 37.971536)
    ulb = core.geodetic.Point(4.3826169, 50.8119483)
    assert 2088389.07865908 == pytest.approx(acropolis.distance(
        ulb, strategy='andoyer'),
                                             abs=1e-6)
    assert 2088384.36439399 == pytest.approx(acropolis.distance(
        ulb, strategy='thomas'),
                                             abs=1e-6)
    assert 2088384.36606831 == pytest.approx(acropolis.distance(
        ulb, strategy='vincenty'),
                                             abs=1e-6)
    assert acropolis.distance(ulb,
                              strategy='thomas') == acropolis.distance(ulb)
    with pytest.raises(ValueError):
        acropolis.distance(ulb, strategy='Thomas')


def test_point_pickle():
    """Serialization tests."""
    a = core.geodetic.Point(1, 2)
    b = pickle.loads(pickle.dumps(a))
    assert a.lon == b.lon
    assert a.lat == b.lat
    assert a == b
    assert not a != b
    assert id(a) != id(b)


def test_box():
    """Test construction and accessors of the object."""
    min_corner = core.geodetic.Point(0, 1)
    max_corner = core.geodetic.Point(2, 3)

    box = core.geodetic.Box(min_corner, max_corner)
    assert str(box) == '((0, 1), (2, 3))'
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

    assert box.wkt() == 'POLYGON((2 3,2 1,0 1,0 3,2 3))'
    assert box.to_geojson() == {
        'type': 'Polygon',
        'coordinates': [[[2, 3], [2, 1], [0, 1], [0, 3], [2, 3]]]
    }
    box = core.geodetic.Box.read_wkt('POLYGON((2 3,2 1,0 1,0 3,2 3))')
    assert repr(box) == '((2, 3), (0, 1))'

    box = core.geodetic.Box.whole_earth()
    assert repr(box) == '((-180, -90), (180, 90))'


def test_box_pickle():
    """Serialization tests."""
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
    polygon = core.geodetic.Polygon.read_wkt('POLYGON((0 0,0 7,4 2,2 0,0 0))')
    assert repr(polygon) == '(((0, 0), (0, 7), (4, 2), (2, 0), (0, 0)))'
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

    assert repr(polygon) == '(((0, 0), (0, 5), (5, 5), (5, 0), (0, 0)), ' \
        '((1, 1), (4, 1), (4, 4), (1, 4), (1, 1)))'
    assert polygon.wkt() == 'POLYGON((0 0,0 5,5 5,5 0,0 0),' \
        '(1 1,4 1,4 4,1 4,1 1))'
    assert polygon.to_geojson() == {
        'type':
        'Polygon',
        'coordinates': [[[0, 0], [0, 5], [5, 5], [5, 0], [0, 0]],
                        [[1, 1], [4, 1], [4, 4], [1, 4], [1, 1]]]
    }


def test_polygon_pickle():
    for item in [
            'POLYGON((0 0,0 7,4 2,2 0,0 0))',
            'POLYGON((0 0,0 5,5 5,5 0,0 0),(1 1,4 1,4 4,1 4,1 1))'
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


def test_multipolygon():
    with open(multipolygon_path()) as stream:
        coordinates = json.load(stream)

    polygons = []
    multipolygon = core.geodetic.MultiPolygon()
    for points in coordinates:
        polygon = core.geodetic.Polygon(
            [core.geodetic.Point(*item) for item in points])
        polygons.append(polygon)
        multipolygon.append(polygon)
    assert len(multipolygon) == len(polygons)
    assert polygons[1] in multipolygon
    assert list(multipolygon) == polygons
    assert isinstance(multipolygon.envelope(), geodetic.Box)
    assert multipolygon.covered_by(polygons[0].outer[0])

    lon = np.fromiter((item[0] for item in coordinates[0]), dtype='float64')
    lat = np.fromiter((item[1] for item in coordinates[0]), dtype='float64')

    assert np.all(multipolygon.covered_by(lon, lat) == 1)

    assert multipolygon.distance(multipolygon) == 0
    assert multipolygon.distance(polygons[-1]) == 0
    assert multipolygon.distance(polygons[0].outer[0]) == 0

    assert multipolygon.area() != 0

    with pytest.raises(IndexError):
        multipolygon[len(polygons)]

    assert isinstance(multipolygon.wkt(), str)


def test_coordinate_distance():
    lon = np.arange(0, 360, 10)
    lat = np.arange(-90, 90.5, 10)
    mx, my = np.meshgrid(lon, lat)
    d1 = core.geodetic.coordinate_distances(mx.ravel(),
                                            my.ravel(),
                                            mx.ravel() + 1,
                                            my.ravel() + 1,
                                            strategy='vincenty',
                                            num_threads=1)
    d0 = core.geodetic.coordinate_distances(mx.ravel(),
                                            my.ravel(),
                                            mx.ravel() + 1,
                                            my.ravel() + 1,
                                            strategy='vincenty',
                                            num_threads=0)
    assert np.all(d0 == d1)
    d0 = d0.reshape(mx.shape)
    for iy in range(d0.shape[0]):
        assert np.all(np.abs((d0[iy, :] - d0[iy, 0]) <= 1e-6))
    for ix in range(d0.shape[1]):
        delta = np.abs(d0[:, ix] - d0[0, ix])
        assert np.all(delta[delta != 0] > 1e3)


def test_crossover():
    """Calculate the location of a crossover."""
    lon1 = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.float64)
    lat1 = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.float64)
    lon2 = np.array(lon1[:])
    lat2 = np.array(lat1[::-1])

    crossover = core.geodetic.Crossover(core.geodetic.LineString(lon1, lat1),
                                        core.geodetic.LineString(lon2, lat2))
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

    for flag in (True, False):
        crossover_properties = core.geodetic.calculate_crossover(
            lon1, lat1, lon2, lat2, cartesian_plane=flag)
        assert crossover_properties is not None
        point, (ix1, ix2) = crossover_properties
        assert (ix1, ix2) == (3, 3) if flag else (4, 3)
        assert pytest.approx(point.lon) == 4
        assert pytest.approx(point.lat, rel=1e-3) == 4.0018282189756835


def test_merged_point():
    """Try to compute a crossover from overlay tracks."""
    lon1 = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.float64)
    lat1 = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.float64)

    crossover = core.geodetic.Crossover(core.geodetic.LineString(lon1, lat1),
                                        core.geodetic.LineString(lon1, lat1))
    assert isinstance(crossover, core.geodetic.Crossover)
    assert crossover.exists()

    with pytest.raises(RuntimeError):
        crossover.search()

    crossover = core.geodetic.Crossover(
        core.geodetic.LineString(lon1, lat1),
        core.geodetic.LineString(lon1 + 10, lat1 + 10))
    assert not crossover.exists()
    assert crossover.search() is None

    for flag in (True, False):
        with pytest.raises(RuntimeError):
            core.geodetic.calculate_crossover(lon1,
                                              lat1,
                                              lon1,
                                              lat1,
                                              cartesian_plane=flag)
        crossover_properties = core.geodetic.calculate_crossover(
            lon1, lat1, lon1 + 10, lat1 + 10, cartesian_plane=flag)


def test_missing_crossover():
    """Try to calculate a crossing point when the entry passes do not cross."""
    x1 = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.float64)
    y1 = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.float64)
    x2 = np.flip(x1, axis=0)
    y2 = np.flip(x2, axis=0)

    crossover = core.geodetic.Crossover(core.geodetic.LineString(x1, y1),
                                        core.geodetic.LineString(x2, y2))
    assert isinstance(crossover, core.geodetic.Crossover)
    assert crossover.exists()

    for flag in (True, False):
        crossover_properties = core.geodetic.calculate_crossover(
            x1, y1, x2, y2, cartesian_plane=flag)
        assert crossover_properties is not None


def test_case_crossover_shift():
    """Calculate a crossover between two tracks."""
    lon1 = np.array([300, 350, 40, 90], dtype=np.float64)
    lat1 = np.array([0, 1, 2, 3], dtype=np.float64)
    lon2 = np.array(lon1[:])
    lat2 = np.array(lat1[::-1])

    crossover = core.geodetic.Crossover(core.geodetic.LineString(lon1, lat1),
                                        core.geodetic.LineString(lon2, lat2))
    assert isinstance(crossover, core.geodetic.Crossover)
    assert crossover.exists()
    coordinates = crossover.search()
    assert coordinates is not None
    assert pytest.approx(coordinates.lon) == 15
    assert pytest.approx(coordinates.lat, rel=1e-3) == 1.6551107341906504
    assert crossover.nearest(coordinates, predicate=None) == (2, 1)

    for flag in (True, False):
        crossover_properties = core.geodetic.calculate_crossover(
            lon1, lat1, lon2, lat2, cartesian_plane=flag)
        assert crossover_properties is not None
        point, (ix1, ix2) = crossover_properties
        assert (ix1, ix2) == (1, 1) if flag else (2, 1)
        assert pytest.approx(point.lon) == 15
        assert pytest.approx(point.lat,
                             rel=1e-3) == (1.5 if flag else 1.6551107341906504)


def test_bbox_from_geojson():
    """Calculate a bounding box from a geojson coordinate array."""
    bbox = core.geodetic.Box.from_geojson([100.0, 0.0, 105.0, 1.0])
    assert bbox.max_corner == core.geodetic.Point(105.0, 1.0)
    assert bbox.min_corner == core.geodetic.Point(100.0, 0.0)


def test_polygon_from_geojson():
    """Calculate a polygon from a geojson coordinate array."""
    polygon = core.geodetic.Polygon.from_geojson([
        [[100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0]],
    ])
    assert polygon.outer == geodetic.LineString([
        core.geodetic.Point(100.0, 0.0),
        core.geodetic.Point(101.0, 0.0),
        core.geodetic.Point(101.0, 1.0),
        core.geodetic.Point(100.0, 1.0),
        core.geodetic.Point(100.0, 0.0),
    ])
    assert polygon.inners == []
    assert polygon.num_interior_rings() == 0

    coordinates = [
        [[100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0]],
        [[100.2, 0.2], [100.8, 0.2], [100.8, 0.8], [100.2, 0.8], [100.2, 0.2]],
    ]
    polygon = core.geodetic.Polygon.from_geojson(coordinates)
    assert polygon.to_geojson() == {
        'type': 'Polygon',
        'coordinates': coordinates,
    }
    assert polygon.outer == geodetic.LineString([
        core.geodetic.Point(100.0, 0.0),
        core.geodetic.Point(101.0, 0.0),
        core.geodetic.Point(101.0, 1.0),
        core.geodetic.Point(100.0, 1.0),
        core.geodetic.Point(100.0, 0.0),
    ])
    assert polygon.inners == [
        geodetic.LineString([
            core.geodetic.Point(100.2, 0.2),
            core.geodetic.Point(100.8, 0.2),
            core.geodetic.Point(100.8, 0.8),
            core.geodetic.Point(100.2, 0.8),
            core.geodetic.Point(100.2, 0.2),
        ])
    ]
    assert polygon.num_interior_rings() == 1


def test_multipolygon_from_geojson():
    """Calculate a multi-polygon from a geojson coordinate array."""
    coordinates = [
        [
            [
                [102.0, 2.0],
                [103.0, 2.0],
                [103.0, 3.0],
                [102.0, 3.0],
                [102.0, 2.0],
            ],
        ],
        [
            [
                [100.0, 0.0],
                [101.0, 0.0],
                [101.0, 1.0],
                [100.0, 1.0],
                [100.0, 0.0],
            ],
            [
                [100.2, 0.2],
                [100.8, 0.2],
                [100.8, 0.8],
                [100.2, 0.8],
                [100.2, 0.2],
            ],
        ],
    ]
    multipolygon = core.geodetic.MultiPolygon.from_geojson(coordinates)
    assert multipolygon.num_interior_rings() == 1
    assert multipolygon.to_geojson() == {
        'type': 'MultiPolygon',
        'coordinates': coordinates
    }
    assert multipolygon[0] == core.geodetic.Polygon([
        core.geodetic.Point(102.0, 2.0),
        core.geodetic.Point(103.0, 2.0),
        core.geodetic.Point(103.0, 3.0),
        core.geodetic.Point(102.0, 3.0),
        core.geodetic.Point(102.0, 2.0),
    ])
    assert multipolygon[1] == core.geodetic.Polygon([
        core.geodetic.Point(100.0, 0.0),
        core.geodetic.Point(101.0, 0.0),
        core.geodetic.Point(101.0, 1.0),
        core.geodetic.Point(100.0, 1.0),
        core.geodetic.Point(100.0, 0.0),
    ], [[
        core.geodetic.Point(100.2, 0.2),
        core.geodetic.Point(100.8, 0.2),
        core.geodetic.Point(100.8, 0.8),
        core.geodetic.Point(100.2, 0.8),
        core.geodetic.Point(100.2, 0.2),
    ]])


def test_union_polygon():
    a = core.geodetic.Polygon.read_wkt(
        'POLYGON((2 1.3,2.4 1.7,2.8 1.8,3.4 1.2,3.7 1.6,3.4 2,4.1 3,5.3 2.6,'
        '5.4 1.2,4.9 0.8,2.9 0.7,2 1.3)(4.0 2.0,4.2 1.4,4.8 1.9,4.4 2.2,'
        '4.0 2.0))')
    b = core.geodetic.Polygon.read_wkt(
        'POLYGON((4.0 -0.5,3.5 1.0,2.0 1.5,3.5 2.0,4.0 3.5,4.5 2.0,6.0 1.5,'
        '4.5 1.0,4.0 -0.5))')
    union = a.union(b)
    assert len(union) == 1


def test_linestring():
    a = core.geodetic.LineString.from_geojson([[1, 2], [2, 3], [3, 4]])
    assert a.wkt() == 'LINESTRING(1 2,2 3,3 4)'
    assert a == core.geodetic.LineString.read_wkt('LINESTRING(1 2,2 3,3 4)')
    assert a == pickle.loads(pickle.dumps(a))
    assert a.to_geojson() == {
        'type': 'LineString',
        'coordinates': [[1, 2], [2, 3], [3, 4]],
    }
    assert len(a) == 3
    assert a[0] == core.geodetic.Point(1, 2)
    assert a[1] == core.geodetic.Point(2, 3)
    assert a[2] == core.geodetic.Point(3, 4)
    with pytest.raises(IndexError):
        a[3]
    assert list(a) == [
        core.geodetic.Point(1, 2),
        core.geodetic.Point(2, 3),
        core.geodetic.Point(3, 4),
    ]
    assert a == core.geodetic.LineString(np.array([1, 2, 3], dtype=float),
                                         np.array([2, 3, 4], dtype=float))
    assert a == a.__copy__()
    b = core.geodetic.LineString()
    b.append(core.geodetic.Point(1, 2))
    b.append(core.geodetic.Point(2, 3))
    b.append(core.geodetic.Point(3, 4))
    assert a == b


def test_curvilinear_distance():
    """Test the curvilinear distance calculation."""
    lon = np.array([
        -9.72270435, -9.58849868, -9.45429341, -9.32009001, -9.18588996,
        -9.05169472, -8.91750577, -8.78332458, -8.64915263, -8.51499137,
        -8.38084228, -8.24670684, -8.1125865, -7.97848274, -7.84439702,
        -7.71033081, -7.57628556, -7.44226275, -7.30826383, -7.17429026,
        -7.04034349, -6.906425, -6.77253622, -6.63867861, -6.50485362,
        -6.37106269, -6.23730728, -6.10358883, -5.96990877, -5.83626855,
        -5.70266959, -5.56911334, -5.43560123, -5.30213467, -5.1687151,
        -5.03534394, -4.9020226, -4.7687525, -4.63553505, -4.50237166,
        -4.36926374, -4.23621268, -4.10321989, -3.97028675, -3.83741466,
        -3.70460501, -3.57185917, -3.43917852, -3.30656444, -3.1740183
    ])
    lat = np.array([
        -77.66299965, -77.66292905, -77.66279284, -77.66259101, -77.66232357,
        -77.66199051, -77.66159184, -77.66112757, -77.66059769, -77.66000222,
        -77.65934117, -77.65861453, -77.65782234, -77.65696458, -77.65604129,
        -77.65505246, -77.65399812, -77.65287828, -77.65169296, -77.65044218,
        -77.64912596, -77.64774431, -77.64629727, -77.64478485, -77.64320707,
        -77.64156397, -77.63985556, -77.63808189, -77.63624296, -77.63433883,
        -77.63236951, -77.63033504, -77.62823545, -77.62607078, -77.62384106,
        -77.62154633, -77.61918663, -77.616762, -77.61427247, -77.61171809,
        -77.6090989, -77.60641495, -77.60366627, -77.60085292, -77.59797493,
        -77.59503237, -77.59202528, -77.5889537, -77.5858177, -77.58261731
    ])
    ls = geodetic.LineString(lon, lat)

    distance = [0.0]
    for ix in range(1, 50):
        distance.append(
            geodetic.Point(lon[ix - 1],
                           lat[ix - 1]).distance(geodetic.Point(
                               lon[ix], lat[ix]),
                                                 strategy='thomas'))

    assert np.all((ls.curvilinear_distance(strategy='thomas') -
                   np.cumsum(np.array(distance))) == np.zeros(50))

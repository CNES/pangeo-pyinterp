# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import os
import pickle

import numpy
import pytest

try:
    import matplotlib.colors
    import matplotlib.pyplot
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False

from .. import geodetic
from ..tests import load_grid2d, make_or_compare_reference


def plot(x, y, z, filename):
    figure = matplotlib.pyplot.figure(figsize=(15, 15), dpi=150)
    z = numpy.ma.fix_invalid(z)
    value = z.mean()
    std = z.std()
    normalize = matplotlib.colors.Normalize(vmin=value - 3 * std,
                                            vmax=value + 3 * std)
    axe = figure.add_subplot(2, 1, 1)
    axe.pcolormesh(x, y, z, cmap='jet', norm=normalize, shading='auto')
    figure.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                filename),
                   bbox_inches='tight',
                   pad_inches=0.4)


def load_data():
    ds = load_grid2d()
    z = ds.mss.T
    mesh = geodetic.RTree()
    x, y = numpy.meshgrid(ds.lon.values, ds.lat.values, indexing='ij')
    mesh.packing(x.ravel(), y.ravel(), z.values.ravel())
    return mesh


def test_spheroid():
    wgs = geodetic.Spheroid()
    assert isinstance(wgs, geodetic.Spheroid)
    with pytest.raises(TypeError):
        wgs = geodetic.Spheroid(12.0)  # type: ignore
    with pytest.raises(TypeError):
        wgs = geodetic.Spheroid((12.0, 3.0, 5))  # type: ignore
    wgs = geodetic.Spheroid((1, 1))
    assert isinstance(wgs, geodetic.Spheroid)
    assert str(wgs) == 'Spheroid(1.0, 1.0)'


def test_coordinates():
    wgs = geodetic.Coordinates()
    assert isinstance(wgs, geodetic.Coordinates)
    wgs = geodetic.Coordinates(geodetic.Spheroid())
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


def test_multipolygon():
    multipolygon = geodetic.MultiPolygon()
    multipolygon.append(
        geodetic.Polygon([
            geodetic.Point(0, 0),
            geodetic.Point(0, 5),
            geodetic.Point(5, 5),
            geodetic.Point(5, 0),
            geodetic.Point(0, 0)
        ]))
    assert len(multipolygon) == 1
    assert pickle.loads(pickle.dumps(multipolygon)) == multipolygon

    assert multipolygon.wkt() == \
        'MULTIPOLYGON(((0 0,0 5,5 5,5 0,0 0)))'

    assert geodetic.MultiPolygon.read_wkt(
        'MULTIPOLYGON(((0 0,0 5,5 5,5 0,0 0)))').wkt(
        ) == 'MULTIPOLYGON(((0 0,0 5,5 5,5 0,0 0)))'


def test_rtree(pytestconfig):
    mesh = load_data()
    lon = numpy.arange(-180, 180, 1) + 1 / 3.0
    lat = numpy.arange(-90, 90, 1) + 1 / 3.0
    x, y = numpy.meshgrid(lon, lat, indexing='ij')
    data, _ = mesh.query(x.ravel(), y.ravel())
    data, _ = mesh.inverse_distance_weighting(x.ravel(), y.ravel())
    if HAVE_PLT and pytestconfig.getoption('visualize'):
        plot(x, y, data.reshape((len(lon), len(lat))),
             'mss_geodetic_rtree_idw.png')
    data, _ = mesh.radial_basis_function(x.ravel(), y.ravel())
    if HAVE_PLT and pytestconfig.getoption('visualize'):
        plot(x, y, data.reshape((len(lon), len(lat))),
             'mss_geodetic_rtree_rbf.png')
    data, _ = mesh.window_function(x.ravel(), y.ravel(), radius=2_000_000)
    if HAVE_PLT and pytestconfig.getoption('visualize'):
        plot(x, y, data.reshape((len(lon), len(lat))),
             'mss_geodetic_rtree_wf.png')

    with pytest.raises(ValueError):
        mesh.radial_basis_function(x.ravel(),
                                   y.ravel(),
                                   epsilon=1,
                                   rbf='cubic')
    with pytest.raises(ValueError):
        mesh.radial_basis_function(x.ravel(), y.ravel(), rbf='X')
    with pytest.raises(ValueError):
        mesh.window_function(x.ravel(), y.ravel(), radius=1, wf='cubic')
    with pytest.raises(ValueError):
        mesh.window_function(x.ravel(),
                             y.ravel(),
                             radius=1,
                             wf='parzen',
                             arg=-1)
    with pytest.raises(ValueError):
        mesh.window_function(x.ravel(),
                             y.ravel(),
                             radius=1,
                             wf='lanczos',
                             arg=0)
    with pytest.raises(ValueError):
        mesh.window_function(x.ravel(),
                             y.ravel(),
                             radius=1,
                             wf='blackman',
                             arg=2)

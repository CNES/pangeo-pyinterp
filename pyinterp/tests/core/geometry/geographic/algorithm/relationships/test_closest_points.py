# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for closest_points algorithm."""

from __future__ import annotations

import numpy as np

from .......core.geometry.geographic import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    Ring,
    Segment,
    Spheroid,
)
from .......core.geometry.geographic.algorithms import Strategy, closest_points


def test_closest_points_polygon_to_polygon() -> None:
    """Test closest_points between two separate polygons."""
    x1 = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    y1 = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    x2 = np.array([5.0, 5.0, 6.0, 6.0, 5.0])
    y2 = np.array([5.0, 6.0, 6.0, 5.0, 5.0])
    poly2 = Polygon(Ring(x2, y2))

    result = closest_points(poly1, poly2)

    # Result should be a segment
    assert isinstance(result, Segment)


def test_closest_points_multipoint_to_multipoint() -> None:
    """Test closest_points between two multipoints."""
    points1 = [Point(0.0, 0.0), Point(5.0, 5.0)]
    mp1 = MultiPoint(points1)

    points2 = [Point(10.0, 10.0), Point(15.0, 15.0)]
    mp2 = MultiPoint(points2)

    result = closest_points(mp1, mp2)

    # Result should be a segment
    assert isinstance(result, Segment)


def test_closest_points_linestring_to_linestring() -> None:
    """Test closest_points between two linestrings."""
    x1 = np.array([0.0, 5.0, 10.0])
    y1 = np.array([0.0, 0.0, 0.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([0.0, 5.0, 10.0])
    y2 = np.array([10.0, 10.0, 10.0])
    ls2 = LineString(x2, y2)

    result = closest_points(ls1, ls2)

    # Result should be a segment
    assert isinstance(result, Segment)


def test_closest_points_multilinestring_to_multilinestring() -> None:
    """Test closest_points between two multilinestrings."""
    ls1 = LineString(np.array([0.0, 5.0]), np.array([0.0, 5.0]))
    ls2 = LineString(np.array([10.0, 15.0]), np.array([10.0, 15.0]))
    mls1 = MultiLineString([ls1, ls2])

    ls3 = LineString(np.array([0.0, 5.0]), np.array([10.0, 15.0]))
    ls4 = LineString(np.array([10.0, 15.0]), np.array([0.0, 5.0]))
    mls2 = MultiLineString([ls3, ls4])

    result = closest_points(mls1, mls2)

    # Result should be a segment
    assert isinstance(result, Segment)


def test_closest_points_multipolygon_to_multipolygon() -> None:
    """Test closest_points between two multipolygons."""
    x1 = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    y1 = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    x2 = np.array([5.0, 5.0, 6.0, 6.0, 5.0])
    y2 = np.array([5.0, 6.0, 6.0, 5.0, 5.0])
    poly2 = Polygon(Ring(x2, y2))

    mp1 = MultiPolygon([poly1])
    mp2 = MultiPolygon([poly2])

    result = closest_points(mp1, mp2)

    # Result should be a segment
    assert isinstance(result, Segment)


def test_closest_points_polygon_to_linestring() -> None:
    """Test closest_points between polygon and linestring."""
    x = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    y = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    polygon = Polygon(Ring(x, y))

    x_ls = np.array([15.0, 20.0])
    y_ls = np.array([5.0, 15.0])
    linestring = LineString(x_ls, y_ls)

    result = closest_points(polygon, linestring)

    # Result should be a segment
    assert isinstance(result, Segment)


def test_closest_points_multipoint_to_polygon() -> None:
    """Test closest_points between multipoint and polygon."""
    x = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    y = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    polygon = Polygon(Ring(x, y))

    points = [Point(15.0, 5.0), Point(20.0, 15.0)]
    mp = MultiPoint(points)

    result = closest_points(mp, polygon)

    # Result should be a segment
    assert isinstance(result, Segment)


def test_closest_points_linestring_to_polygon() -> None:
    """Test closest_points between linestring and polygon."""
    x_ls = np.array([15.0, 20.0])
    y_ls = np.array([5.0, 15.0])
    linestring = LineString(x_ls, y_ls)

    x = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    y = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    polygon = Polygon(Ring(x, y))

    result = closest_points(linestring, polygon)

    # Result should be a segment
    assert isinstance(result, Segment)


def test_closest_points_with_spheroid() -> None:
    """Test closest_points with custom spheroid."""
    x1 = np.array([0.0, 5.0, 10.0])
    y1 = np.array([0.0, 0.0, 0.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([0.0, 5.0, 10.0])
    y2 = np.array([10.0, 10.0, 10.0])
    ls2 = LineString(x2, y2)

    spheroid = Spheroid()

    result = closest_points(ls1, ls2, spheroid=spheroid)

    # Result should be a segment
    assert isinstance(result, Segment)


def test_closest_points_with_strategies() -> None:
    """Test closest_points with different strategies."""
    x1 = np.array([0.0, 5.0, 10.0])
    y1 = np.array([0.0, 0.0, 0.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([0.0, 5.0, 10.0])
    y2 = np.array([10.0, 10.0, 10.0])
    ls2 = LineString(x2, y2)

    # Test with all strategies
    result_vincenty = closest_points(ls1, ls2, strategy=Strategy.VINCENTY)
    result_thomas = closest_points(ls1, ls2, strategy=Strategy.THOMAS)
    result_andoyer = closest_points(ls1, ls2, strategy=Strategy.ANDOYER)
    result_karney = closest_points(ls1, ls2, strategy=Strategy.KARNEY)

    # All strategies should return segments
    for result in [
        result_vincenty,
        result_thomas,
        result_andoyer,
        result_karney,
    ]:
        assert isinstance(result, Segment)


def test_closest_points_equator_linestrings() -> None:
    """Test closest_points between linestrings on equator."""
    x1 = np.array([0.0, 10.0])
    y1 = np.array([0.0, 0.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([0.0, 10.0])
    y2 = np.array([5.0, 5.0])
    ls2 = LineString(x2, y2)

    result = closest_points(ls1, ls2)

    # Result should be a segment
    assert isinstance(result, Segment)


def test_closest_points_meridian_linestrings() -> None:
    """Test closest_points between linestrings on meridian."""
    x1 = np.array([0.0, 0.0])
    y1 = np.array([0.0, 10.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([5.0, 5.0])
    y2 = np.array([0.0, 10.0])
    ls2 = LineString(x2, y2)

    result = closest_points(ls1, ls2)

    # Result should be a segment
    assert isinstance(result, Segment)


def test_closest_points_polar_polygons() -> None:
    """Test closest_points with geometries in polar regions."""
    x1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    y1 = np.array([80.0, 85.0, 85.0, 80.0, 80.0])
    poly1 = Polygon(Ring(x1, y1))

    x2 = np.array([10.0, 10.0, 15.0, 15.0, 10.0])
    y2 = np.array([80.0, 85.0, 85.0, 80.0, 80.0])
    poly2 = Polygon(Ring(x2, y2))

    result = closest_points(poly1, poly2)

    # Result should be a segment
    assert isinstance(result, Segment)


def test_closest_points_antimeridian_linestrings() -> None:
    """Test closest_points with geometries crossing antimeridian."""
    x1 = np.array([170.0, 190.0])
    y1 = np.array([0.0, 0.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([170.0, 190.0])
    y2 = np.array([10.0, 10.0])
    ls2 = LineString(x2, y2)

    result = closest_points(ls1, ls2)

    # Result should be a segment
    assert isinstance(result, Segment)


def test_closest_points_returns_segment_type() -> None:
    """Test that closest_points always returns a Segment."""
    x1 = np.array([0.0, 5.0, 10.0])
    y1 = np.array([0.0, 0.0, 0.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([0.0, 5.0, 10.0])
    y2 = np.array([10.0, 10.0, 10.0])
    ls2 = LineString(x2, y2)

    result = closest_points(ls1, ls2)

    # Must be a Segment
    assert isinstance(result, Segment)
    # Segment should have valid endpoints
    assert hasattr(result, "a")
    assert hasattr(result, "b")

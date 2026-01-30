# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for coordinate conversion algorithms."""

from __future__ import annotations

import numpy as np

from .......core.geometry import cartesian, geographic
from .......core.geometry.cartesian.algorithms import convert_to_geographic
from .......core.geometry.geographic.algorithms import convert_to_cartesian


def test_convert_point_to_geographic() -> None:
    """Test converting cartesian Point to geographic Point."""
    cart_point = cartesian.Point(1.0, 2.0)

    geo_point = convert_to_geographic(cart_point)

    assert isinstance(geo_point, geographic.Point)
    assert geo_point.lon == 1.0
    assert geo_point.lat == 2.0


def test_convert_segment_to_geographic() -> None:
    """Test converting cartesian Segment to geographic Segment."""
    cart_segment = cartesian.Segment((0.0, 0.0), (10.0, 10.0))

    geo_segment = convert_to_geographic(cart_segment)

    assert isinstance(geo_segment, geographic.Segment)


def test_convert_box_to_geographic() -> None:
    """Test converting cartesian Box to geographic Box."""
    cart_box = cartesian.Box((0.0, 0.0), (10.0, 10.0))

    geo_box = convert_to_geographic(cart_box)

    assert isinstance(geo_box, geographic.Box)
    min_corner = geo_box.min_corner
    max_corner = geo_box.max_corner
    assert min_corner.lon == 0.0
    assert min_corner.lat == 0.0
    assert max_corner.lon == 10.0
    assert max_corner.lat == 10.0


def test_convert_linestring_to_geographic() -> None:
    """Test converting cartesian LineString to geographic LineString."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    cart_linestring = cartesian.LineString(x, y)

    geo_linestring = convert_to_geographic(cart_linestring)

    assert isinstance(geo_linestring, geographic.LineString)
    assert len(geo_linestring) == len(cart_linestring)


def test_convert_ring_to_geographic() -> None:
    """Test converting cartesian Ring to geographic Ring."""
    x = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    cart_ring = cartesian.Ring(x, y)

    geo_ring = convert_to_geographic(cart_ring)

    assert isinstance(geo_ring, geographic.Ring)
    assert len(geo_ring) == len(cart_ring)


def test_convert_polygon_to_geographic() -> None:
    """Test converting cartesian Polygon to geographic Polygon."""
    x = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    y = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    cart_polygon = cartesian.Polygon(cartesian.Ring(x, y))

    geo_polygon = convert_to_geographic(cart_polygon)

    assert isinstance(geo_polygon, geographic.Polygon)


def test_convert_polygon_with_hole_to_geographic() -> None:
    """Test converting cartesian Polygon with hole to geographic Polygon."""
    x_outer = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    y_outer = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    outer = cartesian.Ring(x_outer, y_outer)

    x_inner = np.array([2.0, 8.0, 8.0, 2.0, 2.0])
    y_inner = np.array([2.0, 2.0, 8.0, 8.0, 2.0])
    inner = cartesian.Ring(x_inner, y_inner)

    cart_polygon = cartesian.Polygon(outer, [inner])

    geo_polygon = convert_to_geographic(cart_polygon)

    assert isinstance(geo_polygon, geographic.Polygon)
    assert len(geo_polygon.inners) == 1


def test_convert_multipoint_to_geographic() -> None:
    """Test converting cartesian MultiPoint to geographic MultiPoint."""
    points = [
        cartesian.Point(0.0, 0.0),
        cartesian.Point(1.0, 1.0),
        cartesian.Point(2.0, 2.0),
    ]
    cart_multipoint = cartesian.MultiPoint(points)

    geo_multipoint = convert_to_geographic(cart_multipoint)

    assert isinstance(geo_multipoint, geographic.MultiPoint)
    assert len(geo_multipoint) == len(cart_multipoint)


def test_convert_multilinestring_to_geographic() -> None:
    """Test converting MultiLineString to geographic."""
    x1 = np.array([0.0, 1.0])
    y1 = np.array([0.0, 1.0])
    line1 = cartesian.LineString(x1, y1)

    x2 = np.array([1.0, 2.0])
    y2 = np.array([1.0, 0.0])
    line2 = cartesian.LineString(x2, y2)

    cart_multilinestring = cartesian.MultiLineString([line1, line2])

    geo_multilinestring = convert_to_geographic(cart_multilinestring)

    assert isinstance(geo_multilinestring, geographic.MultiLineString)
    assert len(geo_multilinestring) == len(cart_multilinestring)


def test_convert_multipolygon_to_geographic() -> None:
    """Test converting cartesian MultiPolygon to geographic MultiPolygon."""
    x1 = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    y1 = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    poly1 = cartesian.Polygon(cartesian.Ring(x1, y1))

    x2 = np.array([2.0, 2.0, 3.0, 3.0, 2.0])
    y2 = np.array([2.0, 3.0, 3.0, 2.0, 2.0])
    poly2 = cartesian.Polygon(cartesian.Ring(x2, y2))

    cart_multipolygon = cartesian.MultiPolygon([poly1, poly2])

    geo_multipolygon = convert_to_geographic(cart_multipolygon)

    assert isinstance(geo_multipolygon, geographic.MultiPolygon)
    assert len(geo_multipolygon) == len(cart_multipolygon)


def test_convert_empty_linestring_to_geographic() -> None:
    """Test converting empty cartesian LineString to geographic LineString."""
    cart_linestring = cartesian.LineString()

    geo_linestring = convert_to_geographic(cart_linestring)

    assert isinstance(geo_linestring, geographic.LineString)
    assert len(geo_linestring) == 0


def test_convert_roundtrip_consistency() -> None:
    """Test that conversion maintains coordinate values."""
    # Start with cartesian
    cart_point = cartesian.Point(123.456, 78.901)

    # Convert to geographic
    geo_point = convert_to_geographic(cart_point)

    # Convert back to cartesian
    cart_point_restored = convert_to_cartesian(geo_point)

    assert isinstance(cart_point_restored, cartesian.Point)
    assert abs(cart_point_restored.x - cart_point.x) < 1e-10
    assert abs(cart_point_restored.y - cart_point.y) < 1e-10

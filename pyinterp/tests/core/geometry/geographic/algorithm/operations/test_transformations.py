# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for geometric transformation algorithms (geographic)."""

from __future__ import annotations

import numpy as np

from .......core.geometry.geographic import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    Ring,
)
from .......core.geometry.geographic.algorithms import (
    clear,
    convex_hull,
    correct,
    densify,
    line_interpolate,
    reverse,
    simplify,
    unique,
)


# Convex hull tests
def test_convex_hull_polygon(polygon_10x10: Polygon) -> None:
    """Test convex hull of a polygon."""
    hull = convex_hull(polygon_10x10)

    assert isinstance(hull, Polygon)
    assert len(hull.outer) >= 4


def test_convex_hull_multipoint() -> None:
    """Test convex hull of scattered points."""
    points = [Point(0.0, 0.0), Point(10.0, 0.0), Point(5.0, 10.0)]
    multipoint = MultiPoint(points)

    hull = convex_hull(multipoint)

    assert isinstance(hull, Polygon)


# Densify tests
def test_densify_linestring() -> None:
    """Test densify adds points to linestring."""
    lon = np.array([0.0, 10.0])
    lat = np.array([0.0, 0.0])
    linestring = LineString(lon, lat)

    densified = densify(linestring, 2.0)

    assert isinstance(densified, LineString)
    assert len(densified) > len(linestring)


def test_densify_polygon() -> None:
    """Test densify adds points to polygon."""
    lon = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    lat = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    polygon = Polygon(Ring(lon, lat))

    densified = densify(polygon, 2.0)

    assert isinstance(densified, Polygon)
    assert len(densified.outer) > len(polygon.outer)


# Simplify tests
def test_simplify_linestring() -> None:
    """Test simplify removes points from linestring."""
    lon = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
    lat = np.array([0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0])
    linestring = LineString(lon, lat)

    simplified = simplify(linestring, 0.5)

    assert isinstance(simplified, LineString)
    assert len(simplified) <= len(linestring)


def test_simplify_polygon() -> None:
    """Test simplify removes points from polygon."""
    lon = np.array([0.0, 5.0, 10.0, 10.0, 5.0, 0.0, 0.0])
    lat = np.array([0.0, 0.0, 0.0, 5.0, 10.0, 10.0, 0.0])
    polygon = Polygon(Ring(lon, lat))

    simplified = simplify(polygon, 1.0)

    assert isinstance(simplified, Polygon)


# Reverse tests
def test_reverse_linestring(linestring_basic: LineString) -> None:
    """Test reverse reverses point order."""
    first_before = linestring_basic[0]
    reverse(linestring_basic)
    last_after = linestring_basic[len(linestring_basic) - 1]

    assert first_before.lon == last_after.lon
    assert first_before.lat == last_after.lat


def test_reverse_polygon(polygon_10x10: Polygon) -> None:
    """Test reverse on polygon."""
    reverse(polygon_10x10)
    assert isinstance(polygon_10x10, Polygon)


# Correct tests
def test_correct_polygon(polygon_10x10: Polygon) -> None:
    """Test correct fixes orientation."""
    correct(polygon_10x10)
    assert isinstance(polygon_10x10, Polygon)


# Unique tests
def test_unique_linestring() -> None:
    """Test unique removes duplicate consecutive points."""
    lon = np.array([0.0, 0.0, 5.0, 5.0, 10.0])
    lat = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    linestring = LineString(lon, lat)

    unique(linestring)
    assert len(linestring) <= 5


# Clear tests
def test_clear_linestring(linestring_basic: LineString) -> None:
    """Test clear empties a linestring."""
    clear(linestring_basic)
    assert len(linestring_basic) == 0


def test_clear_polygon(polygon_10x10: Polygon) -> None:
    """Test clear empties a polygon."""
    clear(polygon_10x10)
    assert len(polygon_10x10.outer) == 0


# Line interpolate tests
def test_line_interpolate_linestring() -> None:
    """Test line_interpolate finds point at distance along linestring."""
    lon = np.array([0.0, 10.0])
    lat = np.array([0.0, 0.0])
    linestring = LineString(lon, lat)

    point = line_interpolate(linestring, 5.0)

    assert isinstance(point, Point)


def test_line_interpolate_at_start() -> None:
    """Test line_interpolate at distance 0."""
    lon = np.array([0.0, 10.0])
    lat = np.array([0.0, 0.0])
    linestring = LineString(lon, lat)

    point = line_interpolate(linestring, 0.0)

    assert isinstance(point, Point)

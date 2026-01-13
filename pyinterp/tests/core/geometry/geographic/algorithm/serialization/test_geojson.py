# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for GeoJSON serialization and deserialization (geographic)."""

import json

import numpy as np

from .......core.geometry.geographic import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    Ring,
)
from .......core.geometry.geographic.algorithms import from_geojson, to_geojson


def test_point_to_geojson() -> None:
    """Test Point to GeoJSON serialization."""
    point = Point(1.5, 2.5)
    geojson_str = to_geojson(point)

    # Parse and validate structure
    geojson = json.loads(geojson_str)
    assert geojson["type"] == "Point"
    assert geojson["coordinates"] == [1.5, 2.5]


def test_point_from_geojson() -> None:
    """Test Point from GeoJSON deserialization."""
    geojson_str = '{"type": "Point", "coordinates": [1.5, 2.5]}'
    point = from_geojson(geojson_str)

    assert isinstance(point, Point)
    assert point.lon == 1.5
    assert point.lat == 2.5


def test_point_roundtrip() -> None:
    """Test Point GeoJSON round-trip conversion."""
    original = Point(3.14, 2.71)
    geojson_str = to_geojson(original)
    restored = from_geojson(geojson_str)

    assert isinstance(restored, Point)
    assert restored.lon == original.lon
    assert restored.lat == original.lat


def test_linestring_to_geojson(linestring_basic: LineString) -> None:
    """Test LineString to GeoJSON serialization."""
    geojson_str = to_geojson(linestring_basic)

    geojson = json.loads(geojson_str)
    assert geojson["type"] == "LineString"
    assert isinstance(geojson["coordinates"], list)
    assert len(geojson["coordinates"]) == len(linestring_basic)


def test_linestring_from_geojson() -> None:
    """Test LineString from GeoJSON deserialization."""
    geojson_str = """
    {
        "type": "LineString",
        "coordinates": [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
    }
    """
    linestring = from_geojson(geojson_str)

    assert isinstance(linestring, LineString)
    assert len(linestring) == 3


def test_linestring_roundtrip() -> None:
    """Test LineString GeoJSON round-trip conversion."""
    lon = np.array([0.0, 1.0, 2.0, 3.0])
    lat = np.array([0.0, 1.0, 0.5, 0.0])
    original = LineString(lon, lat)

    geojson_str = to_geojson(original)
    restored = from_geojson(geojson_str)

    assert isinstance(restored, LineString)
    assert len(restored) == len(original)


def test_polygon_to_geojson(polygon_1x1: Polygon) -> None:
    """Test Polygon to GeoJSON serialization."""
    geojson_str = to_geojson(polygon_1x1)

    geojson = json.loads(geojson_str)
    assert geojson["type"] == "Polygon"
    assert isinstance(geojson["coordinates"], list)
    # First element is outer ring
    assert len(geojson["coordinates"]) >= 1


def test_polygon_with_hole_to_geojson(polygon_with_hole: Polygon) -> None:
    """Test Polygon with hole to GeoJSON serialization."""
    geojson_str = to_geojson(polygon_with_hole)

    geojson = json.loads(geojson_str)
    assert geojson["type"] == "Polygon"
    # Should have outer ring and inner ring (hole)
    assert len(geojson["coordinates"]) == 2


def test_polygon_from_geojson() -> None:
    """Test Polygon from GeoJSON deserialization."""
    geojson_str = """
    {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
        ]
    }
    """
    polygon = from_geojson(geojson_str)

    assert isinstance(polygon, Polygon)


def test_polygon_with_hole_from_geojson() -> None:
    """Test Polygon with hole from GeoJSON deserialization."""
    geojson_str = """
    {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
            [[2.0, 2.0], [2.0, 8.0], [8.0, 8.0], [8.0, 2.0], [2.0, 2.0]]
        ]
    }
    """
    polygon = from_geojson(geojson_str)

    assert isinstance(polygon, Polygon)
    assert len(polygon.inners) == 1


def test_polygon_roundtrip(polygon_10x10: Polygon) -> None:
    """Test Polygon GeoJSON round-trip conversion."""
    geojson_str = to_geojson(polygon_10x10)
    restored = from_geojson(geojson_str)

    assert isinstance(restored, Polygon)


def test_polygon_with_hole_roundtrip(polygon_with_hole: Polygon) -> None:
    """Test Polygon with hole GeoJSON round-trip conversion."""
    geojson_str = to_geojson(polygon_with_hole)
    restored = from_geojson(geojson_str)

    assert isinstance(restored, Polygon)
    assert len(restored.inners) == len(polygon_with_hole.inners)


def test_multipoint_to_geojson(multipoint_basic: MultiPoint) -> None:
    """Test MultiPoint to GeoJSON serialization."""
    geojson_str = to_geojson(multipoint_basic)

    geojson = json.loads(geojson_str)
    assert geojson["type"] == "MultiPoint"
    assert isinstance(geojson["coordinates"], list)
    assert len(geojson["coordinates"]) == len(multipoint_basic)


def test_multipoint_from_geojson() -> None:
    """Test MultiPoint from GeoJSON deserialization."""
    geojson_str = """
    {
        "type": "MultiPoint",
        "coordinates": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    }
    """
    multipoint = from_geojson(geojson_str)

    assert isinstance(multipoint, MultiPoint)
    assert len(multipoint) == 3


def test_multipoint_roundtrip(multipoint_simple: MultiPoint) -> None:
    """Test MultiPoint GeoJSON round-trip conversion."""
    geojson_str = to_geojson(multipoint_simple)
    restored = from_geojson(geojson_str)

    assert isinstance(restored, MultiPoint)
    assert len(restored) == len(multipoint_simple)


def test_multilinestring_to_geojson(
    multilinestring_basic: MultiLineString,
) -> None:
    """Test MultiLineString to GeoJSON serialization."""
    geojson_str = to_geojson(multilinestring_basic)

    geojson = json.loads(geojson_str)
    assert geojson["type"] == "MultiLineString"
    assert isinstance(geojson["coordinates"], list)
    assert len(geojson["coordinates"]) == len(multilinestring_basic)


def test_multilinestring_from_geojson() -> None:
    """Test MultiLineString from GeoJSON deserialization."""
    geojson_str = """
    {
        "type": "MultiLineString",
        "coordinates": [
            [[0.0, 0.0], [1.0, 1.0]],
            [[2.0, 2.0], [3.0, 3.0]]
        ]
    }
    """
    multilinestring = from_geojson(geojson_str)

    assert isinstance(multilinestring, MultiLineString)
    assert len(multilinestring) == 2


def test_multilinestring_roundtrip(
    multilinestring_simple: MultiLineString,
) -> None:
    """Test MultiLineString GeoJSON round-trip conversion."""
    geojson_str = to_geojson(multilinestring_simple)
    restored = from_geojson(geojson_str)

    assert isinstance(restored, MultiLineString)
    assert len(restored) == len(multilinestring_simple)


def test_multipolygon_to_geojson(multipolygon_simple: MultiPolygon) -> None:
    """Test MultiPolygon to GeoJSON serialization."""
    geojson_str = to_geojson(multipolygon_simple)

    geojson = json.loads(geojson_str)
    assert geojson["type"] == "MultiPolygon"
    assert isinstance(geojson["coordinates"], list)
    assert len(geojson["coordinates"]) == len(multipolygon_simple)


def test_multipolygon_from_geojson() -> None:
    """Test MultiPolygon from GeoJSON deserialization."""
    geojson_str = """
    {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
            [[[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0], [2.0, 2.0]]]
        ]
    }
    """
    multipolygon = from_geojson(geojson_str)

    assert isinstance(multipolygon, MultiPolygon)
    assert len(multipolygon) == 2


def test_multipolygon_roundtrip(multipolygon_complex: MultiPolygon) -> None:
    """Test MultiPolygon GeoJSON round-trip conversion."""
    geojson_str = to_geojson(multipolygon_complex)
    restored = from_geojson(geojson_str)

    assert isinstance(restored, MultiPolygon)
    assert len(restored) == len(multipolygon_complex)


def test_empty_geometries_to_geojson() -> None:
    """Test serialization of empty geometries."""
    empty_point = Point()
    empty_linestring = LineString()
    empty_polygon = Polygon(Ring(np.array([]), np.array([])))
    empty_multipoint = MultiPoint()
    empty_multilinestring = MultiLineString()
    empty_multipolygon = MultiPolygon()

    # All should produce valid GeoJSON
    for geom in [
        empty_point,
        empty_linestring,
        empty_polygon,
        empty_multipoint,
        empty_multilinestring,
        empty_multipolygon,
    ]:
        geojson_str = to_geojson(geom)  # type:ignore[arg-type]
        # Should be valid JSON
        geojson = json.loads(geojson_str)
        assert "type" in geojson
        assert "coordinates" in geojson


def test_geojson_with_whitespace() -> None:
    """Test parsing GeoJSON with extra whitespace."""
    geojson_str = """
    {
        "type":    "Point",
        "coordinates":   [  1.0  ,  2.0  ]
    }
    """
    point = from_geojson(geojson_str)

    assert isinstance(point, Point)
    assert point.lon == 1.0
    assert point.lat == 2.0


def test_geojson_coordinate_precision() -> None:
    """Test GeoJSON handles high-precision coordinates."""
    point = Point(1.123456789012345, 2.987654321098765)
    geojson_str = to_geojson(point)
    restored = from_geojson(geojson_str)

    assert isinstance(restored, Point)
    # Check precision is preserved (within floating-point limits)
    assert abs(restored.lon - point.lon) < 1e-10
    assert abs(restored.lat - point.lat) < 1e-10


def test_geojson_antimeridian() -> None:
    """Test GeoJSON with coordinates crossing the antimeridian."""
    lon = np.array([175.0, 180.0, -175.0])
    lat = np.array([0.0, 0.0, 0.0])
    linestring = LineString(lon, lat)

    geojson_str = to_geojson(linestring)
    restored = from_geojson(geojson_str)

    assert isinstance(restored, LineString)
    assert len(restored) == len(linestring)


def test_geojson_polar_coordinates() -> None:
    """Test GeoJSON with coordinates near poles."""
    lon = np.array([0.0, 90.0, 180.0, -90.0, 0.0])
    lat = np.array([89.0, 89.0, 89.0, 89.0, 89.0])
    ring = Ring(lon, lat)
    polygon = Polygon(ring)

    geojson_str = to_geojson(polygon)
    restored = from_geojson(geojson_str)

    assert isinstance(restored, Polygon)

# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for WKT (Well-Known Text) serialization and deserialization."""

import numpy as np

from .......core.geometry.cartesian import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from .......core.geometry.cartesian.algorithms import from_wkt, to_wkt


def test_point_to_wkt() -> None:
    """Test Point to WKT serialization."""
    point = Point(1.5, 2.5)
    wkt = to_wkt(point)

    assert wkt.startswith("POINT")
    assert "1.5" in wkt
    assert "2.5" in wkt


def test_point_from_wkt() -> None:
    """Test Point from WKT deserialization."""
    wkt = "POINT(1.5 2.5)"
    point = from_wkt(wkt)

    assert isinstance(point, Point)
    assert point.x == 1.5
    assert point.y == 2.5


def test_point_roundtrip() -> None:
    """Test Point WKT round-trip conversion."""
    original = Point(3.14, 2.71)
    wkt = to_wkt(original)
    restored = from_wkt(wkt)

    assert isinstance(restored, Point)
    assert abs(restored.x - original.x) < 1e-10
    assert abs(restored.y - original.y) < 1e-10


def test_linestring_to_wkt(linestring_basic: LineString) -> None:
    """Test LineString to WKT serialization."""
    wkt = to_wkt(linestring_basic)

    assert wkt.startswith("LINESTRING")
    assert "(" in wkt
    assert ")" in wkt


def test_linestring_from_wkt() -> None:
    """Test LineString from WKT deserialization."""
    wkt = "LINESTRING(0 0, 1 1, 2 0)"
    linestring = from_wkt(wkt)

    assert isinstance(linestring, LineString)
    assert len(linestring) == 3


def test_linestring_roundtrip() -> None:
    """Test LineString WKT round-trip conversion."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 0.5, 0.0])
    original = LineString(x, y)

    wkt = to_wkt(original)
    restored = from_wkt(wkt)

    assert isinstance(restored, LineString)
    assert len(restored) == len(original)


def test_polygon_to_wkt(polygon_1x1: Polygon) -> None:
    """Test Polygon to WKT serialization."""
    wkt = to_wkt(polygon_1x1)

    assert wkt.startswith("POLYGON")
    assert "(" in wkt
    assert ")" in wkt


def test_polygon_with_hole_to_wkt(polygon_with_hole: Polygon) -> None:
    """Test Polygon with hole to WKT serialization."""
    wkt = to_wkt(polygon_with_hole)

    assert wkt.startswith("POLYGON")
    # Should contain both outer and inner rings
    assert wkt.count("(") >= 2


def test_polygon_from_wkt() -> None:
    """Test Polygon from WKT deserialization."""
    wkt = "POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))"
    polygon = from_wkt(wkt)

    assert isinstance(polygon, Polygon)


def test_polygon_with_hole_from_wkt() -> None:
    """Test Polygon with hole from WKT deserialization."""
    wkt = "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0), (2 2, 8 2, 8 8, 2 8, 2 2))"
    polygon = from_wkt(wkt)

    assert isinstance(polygon, Polygon)
    assert len(polygon.inners) == 1


def test_polygon_roundtrip(polygon_10x10: Polygon) -> None:
    """Test Polygon WKT round-trip conversion."""
    wkt = to_wkt(polygon_10x10)
    restored = from_wkt(wkt)

    assert isinstance(restored, Polygon)


def test_polygon_with_hole_roundtrip(polygon_with_hole: Polygon) -> None:
    """Test Polygon with hole WKT round-trip conversion."""
    wkt = to_wkt(polygon_with_hole)
    restored = from_wkt(wkt)

    assert isinstance(restored, Polygon)
    assert len(restored.inners) == len(polygon_with_hole.inners)


def test_multipoint_to_wkt(multipoint_basic: MultiPoint) -> None:
    """Test MultiPoint to WKT serialization."""
    wkt = to_wkt(multipoint_basic)

    assert wkt.startswith("MULTIPOINT")


def test_multipoint_from_wkt() -> None:
    """Test MultiPoint from WKT deserialization."""
    wkt = "MULTIPOINT((0 0), (1 1), (2 2))"
    multipoint = from_wkt(wkt)

    assert isinstance(multipoint, MultiPoint)
    assert len(multipoint) == 3


def test_multipoint_alternate_syntax_from_wkt() -> None:
    """Test MultiPoint from WKT with alternate syntax (no inner parens)."""
    wkt = "MULTIPOINT(0 0, 1 1, 2 2)"
    multipoint = from_wkt(wkt)

    assert isinstance(multipoint, MultiPoint)
    assert len(multipoint) == 3


def test_multipoint_roundtrip(multipoint_simple: MultiPoint) -> None:
    """Test MultiPoint WKT round-trip conversion."""
    wkt = to_wkt(multipoint_simple)
    restored = from_wkt(wkt)

    assert isinstance(restored, MultiPoint)
    assert len(restored) == len(multipoint_simple)


def test_multilinestring_to_wkt(
    multilinestring_basic: MultiLineString,
) -> None:
    """Test MultiLineString to WKT serialization."""
    wkt = to_wkt(multilinestring_basic)

    assert wkt.startswith("MULTILINESTRING")


def test_multilinestring_from_wkt() -> None:
    """Test MultiLineString from WKT deserialization."""
    wkt = "MULTILINESTRING((0 0, 1 1), (2 2, 3 3))"
    multilinestring = from_wkt(wkt)

    assert isinstance(multilinestring, MultiLineString)
    assert len(multilinestring) == 2


def test_multilinestring_roundtrip(
    multilinestring_simple: MultiLineString,
) -> None:
    """Test MultiLineString WKT round-trip conversion."""
    wkt = to_wkt(multilinestring_simple)
    restored = from_wkt(wkt)

    assert isinstance(restored, MultiLineString)
    assert len(restored) == len(multilinestring_simple)


def test_multipolygon_to_wkt(multipolygon_simple: MultiPolygon) -> None:
    """Test MultiPolygon to WKT serialization."""
    wkt = to_wkt(multipolygon_simple)

    assert wkt.startswith("MULTIPOLYGON")


def test_multipolygon_from_wkt() -> None:
    """Test MultiPolygon from WKT deserialization."""
    wkt = (
        "MULTIPOLYGON(((0 0, 0 1, 1 1, 1 0, 0 0)), "
        "((2 2, 2 3, 3 3, 3 2, 2 2)))"
    )
    multipolygon = from_wkt(wkt)

    assert isinstance(multipolygon, MultiPolygon)
    assert len(multipolygon) == 2


def test_multipolygon_roundtrip(multipolygon_complex: MultiPolygon) -> None:
    """Test MultiPolygon WKT round-trip conversion."""
    wkt = to_wkt(multipolygon_complex)
    restored = from_wkt(wkt)

    assert isinstance(restored, MultiPolygon)
    assert len(restored) == len(multipolygon_complex)


def test_wkt_case_insensitive() -> None:
    """Test WKT parsing is case-insensitive."""
    wkt_upper = "POINT(1.5 2.5)"
    wkt_lower = "point(1.5 2.5)"
    wkt_mixed = "PoInT(1.5 2.5)"

    point_upper = from_wkt(wkt_upper)
    point_lower = from_wkt(wkt_lower)
    point_mixed = from_wkt(wkt_mixed)

    assert isinstance(point_upper, Point)
    assert isinstance(point_lower, Point)
    assert isinstance(point_mixed, Point)


def test_wkt_with_extra_whitespace() -> None:
    """Test WKT parsing with extra whitespace."""
    wkt = "POINT  (  1.5   2.5  ) "
    point = from_wkt(wkt)

    assert isinstance(point, Point)
    assert point.x == 1.5
    assert point.y == 2.5


def test_wkt_high_precision() -> None:
    """Test WKT handles high-precision coordinates."""
    point = Point(1.123456789012345, 2.987654321098765)
    wkt = to_wkt(point)
    restored = from_wkt(wkt)

    assert isinstance(restored, Point)
    # Check precision is preserved (within WKT serialization limits)
    assert abs(restored.x - point.x) < 1e-5
    assert abs(restored.y - point.y) < 1e-5


def test_wkt_negative_coordinates() -> None:
    """Test WKT with negative coordinates."""
    point = Point(-123.45, -67.89)
    wkt = to_wkt(point)
    restored = from_wkt(wkt)

    assert isinstance(restored, Point)
    assert abs(restored.x - point.x) < 1e-10
    assert abs(restored.y - point.y) < 1e-10


def test_wkt_large_coordinates() -> None:
    """Test WKT with large coordinate values."""
    point = Point(1000000.0, 2000000.0)
    wkt = to_wkt(point)
    restored = from_wkt(wkt)

    assert isinstance(restored, Point)
    assert abs(restored.x - point.x) < 1e-6
    assert abs(restored.y - point.y) < 1e-6

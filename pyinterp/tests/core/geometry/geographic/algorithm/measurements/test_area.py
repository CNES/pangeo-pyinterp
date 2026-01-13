# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for area algorithm."""

from .......core.geometry.geographic import (
    Box,
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
from .......core.geometry.geographic.algorithms import Strategy, area


def test_strategy_enum() -> None:
    """Test Strategy enum values."""
    assert hasattr(Strategy, "ANDOYER")
    assert hasattr(Strategy, "KARNEY")
    assert hasattr(Strategy, "THOMAS")
    assert hasattr(Strategy, "VINCENTY")

    # Check enum values
    assert Strategy.ANDOYER is not None
    assert Strategy.KARNEY is not None
    assert Strategy.THOMAS is not None
    assert Strategy.VINCENTY is not None

    # Enum should be comparable
    assert Strategy.VINCENTY == Strategy.VINCENTY
    assert Strategy.ANDOYER != Strategy.KARNEY


def test_area_point() -> None:
    """Test area of a point (always 0)."""
    point = Point(10.0, 20.0)
    result = area(point)

    # Points have zero area
    assert result == 0.0

    # All strategies should return 0 for points
    assert area(point, strategy=Strategy.ANDOYER) == 0.0
    assert area(point, strategy=Strategy.KARNEY) == 0.0
    assert area(point, strategy=Strategy.THOMAS) == 0.0
    assert area(point, strategy=Strategy.VINCENTY) == 0.0


def test_area_box(box_1x1: Box, box_10x10: Box) -> None:
    """Test area calculation for a box."""
    result = area(box_1x1)

    # Area should be positive
    assert result > 0

    # Approximately 12,364 km² (1 degree x 1 degree at equator)
    # Convert to m²: ~12,364,000,000 m²
    assert 12.3e9 < result < 12.4e9

    # Larger box should have larger area
    large_area = area(box_10x10)
    assert (
        large_area > result * 90
    )  # Should be ~100x but accounting for curvature


def test_area_box_different_strategies(box_1x1: Box) -> None:
    """Test that different strategies give similar but not identical results."""
    area_andoyer = area(box_1x1, strategy=Strategy.ANDOYER)
    area_karney = area(box_1x1, strategy=Strategy.KARNEY)
    area_thomas = area(box_1x1, strategy=Strategy.THOMAS)
    area_vincenty = area(box_1x1, strategy=Strategy.VINCENTY)

    # All should be positive
    assert area_andoyer > 0
    assert area_karney > 0
    assert area_thomas > 0
    assert area_vincenty > 0

    # All should be within 1% of each other for this simple case
    mean_area = (area_andoyer + area_karney + area_thomas + area_vincenty) / 4
    assert abs(area_andoyer - mean_area) / mean_area < 0.01
    assert abs(area_karney - mean_area) / mean_area < 0.01
    assert abs(area_thomas - mean_area) / mean_area < 0.01
    assert abs(area_vincenty - mean_area) / mean_area < 0.01


def test_area_ring(ring_square_1x1: Ring, box_1x1: Box) -> None:
    """Test area calculation for a ring."""
    result = area(ring_square_1x1)

    # Area should be positive (counter-clockwise winding)
    assert result > 0

    # Should be similar to the box area (same bounds)
    box_area = area(box_1x1)
    # Ring and box should have very similar areas
    assert abs(result - box_area) / box_area < 0.01


def test_area_polygon_simple(polygon_1x1: Polygon, box_1x1: Box) -> None:
    """Test area calculation for a simple polygon."""
    result = area(polygon_1x1)

    # Area should be positive (counter-clockwise winding)
    assert result > 0

    # Should be similar to the box/ring area
    box_area = area(box_1x1)
    assert abs(result - box_area) / box_area < 0.01


def test_area_polygon_with_hole(
    polygon_10x10: Polygon, polygon_with_hole: Polygon, ring_square_inner: Ring
) -> None:
    """Test area calculation for a polygon with a hole."""
    area_no_hole = area(polygon_10x10)
    area_with_hole = area(polygon_with_hole)

    # Area with hole should be smaller
    assert area_with_hole < area_no_hole

    # The difference should be approximately the area of the hole
    hole_area = area(Polygon(ring_square_inner))
    assert abs((area_no_hole - area_with_hole) - hole_area) / hole_area < 0.01


def test_area_multipolygon(
    multipolygon_simple: MultiPolygon, polygon_pair: tuple[Polygon, Polygon]
) -> None:
    """Test area calculation for a multipolygon."""
    multi_area = area(multipolygon_simple)

    # Area should be sum of individual polygons
    poly1, poly2 = polygon_pair
    area1 = area(poly1)
    area2 = area(poly2)

    # The areas should be approximately equal (same size squares)
    assert abs(area1 - area2) / area1 < 0.1

    # Multi area should be sum of individual areas
    assert abs(multi_area - (area1 + area2)) / multi_area < 0.01


def test_area_with_custom_spheroid(box_1x1: Box) -> None:
    """Test area calculation with custom spheroid."""
    # Default (WGS84)
    area_wgs84 = area(box_1x1)

    # Custom spheroid with WGS84 parameters (semi-major axis and flattening)
    # WGS84: a=6378137.0, 1/f=298.257223563, so f≈0.0033528
    custom_spheroid = Spheroid(6378137.0, 1.0 / 298.257223563)
    area_custom = area(box_1x1, spheroid=custom_spheroid)

    # Areas should be identical (using exact WGS84 parameters)
    assert abs(area_wgs84 - area_custom) / area_wgs84 < 1e-10


def test_area_at_poles(box_polar: Box, box_equator: Box) -> None:
    """Test area calculation near poles."""
    north_area = area(box_polar)
    equator_area = area(box_equator)

    # Polar box should have smaller area due to convergence of meridians
    assert north_area < equator_area


def test_area_across_antimeridian(
    box_antimeridian: Box, box_10x10: Box
) -> None:
    """Test area calculation for geometry crossing anti-meridian."""
    result = area(box_antimeridian)

    # Should have positive area (10 degrees lon x 10 degrees lat)
    assert result > 0

    # Should be similar to a 10x10 box at equator
    reference_area = area(box_10x10)

    # Areas should be similar (within 5% due to slight latitude difference)
    assert abs(result - reference_area) / reference_area < 0.05


def test_area_zero_size_geometries(
    box_zero: Box, ring_empty: Ring, polygon_empty: Polygon
) -> None:
    """Test area of zero-size geometries."""
    assert area(box_zero) == 0.0
    assert area(ring_empty) == 0.0
    assert area(polygon_empty) == 0.0


def test_area_segment(segment_basic: Segment) -> None:
    """Test area calculation for a segment."""
    result = area(segment_basic)

    # Area of a segment should be zero
    assert result == 0.0

    # Test with different strategies
    assert area(segment_basic, strategy=Strategy.ANDOYER) == 0.0
    assert area(segment_basic, strategy=Strategy.KARNEY) == 0.0
    assert area(segment_basic, strategy=Strategy.THOMAS) == 0.0
    assert area(segment_basic, strategy=Strategy.VINCENTY) == 0.0


def test_area_linestring(linestring_basic: LineString) -> None:
    """Test area calculation for a linestring."""
    result = area(linestring_basic)

    # Area of a linestring should be zero
    assert result == 0.0

    # Test with different strategies
    assert area(linestring_basic, strategy=Strategy.ANDOYER) == 0.0
    assert area(linestring_basic, strategy=Strategy.KARNEY) == 0.0
    assert area(linestring_basic, strategy=Strategy.THOMAS) == 0.0
    assert area(linestring_basic, strategy=Strategy.VINCENTY) == 0.0


def test_area_multipoint(multipoint_basic: MultiPoint) -> None:
    """Test area calculation for a multipoint."""
    result = area(multipoint_basic)

    # Area of a multipoint should be zero
    assert result == 0.0

    # Test with different strategies
    assert area(multipoint_basic, strategy=Strategy.ANDOYER) == 0.0
    assert area(multipoint_basic, strategy=Strategy.KARNEY) == 0.0
    assert area(multipoint_basic, strategy=Strategy.THOMAS) == 0.0
    assert area(multipoint_basic, strategy=Strategy.VINCENTY) == 0.0


def test_area_multilinestring(multilinestring_basic: MultiLineString) -> None:
    """Test area calculation for a multilinestring."""
    result = area(multilinestring_basic)

    # Area of a multilinestring should be zero
    assert result == 0.0

    # Test with different strategies
    assert area(multilinestring_basic, strategy=Strategy.ANDOYER) == 0.0
    assert area(multilinestring_basic, strategy=Strategy.KARNEY) == 0.0
    assert area(multilinestring_basic, strategy=Strategy.THOMAS) == 0.0
    assert area(multilinestring_basic, strategy=Strategy.VINCENTY) == 0.0

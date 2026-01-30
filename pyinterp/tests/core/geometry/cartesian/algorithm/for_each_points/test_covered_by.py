# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for for_each_point_X algorithms (Cartesian)."""

from __future__ import annotations

import numpy as np

from .......core.geometry.cartesian import (
    Box,
    LineString,
    MultiPoint,
    MultiPolygon,
    Polygon,
    Ring,
)
from .......core.geometry.cartesian.algorithms import for_each_point_covered_by


class TestForEachPointCoveredBy:
    """Tests for for_each_point_covered_by algorithm."""

    def test_for_each_point_covered_by_multipoint_box(self) -> None:
        """Test for_each_point_covered_by with MultiPoint and Box."""
        # Create multipoint with points inside and outside box
        x = np.array([1.0, 2.5, 10.0])
        y = np.array([1.0, 2.5, 10.0])
        source = MultiPoint(x, y)

        # Create a box from (0, 0) to (5, 5)
        container = Box((0.0, 0.0), (5.0, 5.0))

        result = for_each_point_covered_by(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 3
        assert result[0]  # (1.0, 1.0) is covered
        assert result[1]  # (2.5, 2.5) is covered
        assert not result[2]  # (10.0, 10.0) is not covered

    def test_for_each_point_covered_by_multipoint_polygon(self) -> None:
        """Test for_each_point_covered_by with MultiPoint and Polygon."""
        # Create multipoint with various points
        x = np.array([1.0, 2.5, 10.0, 5.0])
        y = np.array([1.0, 2.5, 10.0, 5.0])
        source = MultiPoint(x, y)

        # Create a square polygon from (0, 0) to (5, 5)
        ring_x = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
        ring_y = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
        container = Polygon(Ring(ring_x, ring_y))

        result = for_each_point_covered_by(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 4
        assert result[0]  # Inside polygon
        assert result[1]  # Inside polygon
        assert not result[2]  # Outside polygon
        assert result[3]  # On border of polygon

    def test_for_each_point_covered_by_linestring_box(self) -> None:
        """Test for_each_point_covered_by with LineString and Box."""
        # Create a linestring
        x = np.array([0.5, 2.5, 4.5])
        y = np.array([0.5, 2.5, 4.5])
        source = LineString(x, y)

        # Create a box
        container = Box((0.0, 0.0), (5.0, 5.0))

        result = for_each_point_covered_by(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 3
        # All points should be covered
        assert np.all(result)

    def test_for_each_point_covered_by_ring_polygon(self) -> None:
        """Test for_each_point_covered_by with Ring and Polygon."""
        # Create a ring (closed linestring)
        ring_x = np.array([1.0, 2.0, 2.0, 1.0, 1.0])
        ring_y = np.array([1.0, 1.0, 2.0, 2.0, 1.0])
        source = Ring(ring_x, ring_y)

        # Create a larger polygon container
        container_ring_x = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
        container_ring_y = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
        container = Polygon(Ring(container_ring_x, container_ring_y))

        result = for_each_point_covered_by(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 5
        # All points of the ring should be covered
        assert np.all(result)

    def test_for_each_point_covered_by_empty_result(self) -> None:
        """Test for_each_point_covered_by with no points covered."""
        # Create multipoint far from container
        x = np.array([100.0, 200.0])
        y = np.array([100.0, 200.0])
        source = MultiPoint(x, y)

        # Create a box at origin
        container = Box((0.0, 0.0), (5.0, 5.0))

        result = for_each_point_covered_by(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 2
        assert not np.any(result)

    def test_for_each_point_covered_by_all_covered(self) -> None:
        """Test for_each_point_covered_by with all points covered."""
        # Create multipoint entirely within box
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        source = MultiPoint(x, y)

        # Create a large box
        container = Box((0.0, 0.0), (10.0, 10.0))

        result = for_each_point_covered_by(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 4
        assert np.all(result)

    def test_for_each_point_covered_by_multipolygon(self) -> None:
        """Test for_each_point_covered_by with MultiPolygon container."""
        # Create multipoint
        x = np.array([1.0, 3.0, 7.0])
        y = np.array([1.0, 3.0, 7.0])
        source = MultiPoint(x, y)

        # Create first polygon (0-2, 0-2)
        ring1_x = np.array([0.0, 0.0, 2.0, 2.0, 0.0])
        ring1_y = np.array([0.0, 2.0, 2.0, 0.0, 0.0])
        poly1 = Polygon(Ring(ring1_x, ring1_y))

        # Create second polygon (6-8, 6-8)
        ring2_x = np.array([6.0, 6.0, 8.0, 8.0, 6.0])
        ring2_y = np.array([6.0, 8.0, 8.0, 6.0, 6.0])
        poly2 = Polygon(Ring(ring2_x, ring2_y))

        container = MultiPolygon([poly1, poly2])

        result = for_each_point_covered_by(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 3
        assert result[0]  # (1.0, 1.0) in poly1
        assert not result[1]  # (3.0, 3.0) not in any polygon
        assert result[2]  # (7.0, 7.0) in poly2

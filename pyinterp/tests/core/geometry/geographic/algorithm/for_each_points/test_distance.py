# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for for_each_point_X algorithms (Geographic)."""

from __future__ import annotations

import numpy as np
import pytest

from .......core.geometry.geographic import (
    Box,
    LineString,
    MultiPoint,
    Polygon,
    Ring,
    Spheroid,
)
from .......core.geometry.geographic.algorithms import for_each_point_distance


class TestForEachPointDistance:
    """Tests for for_each_point_distance algorithm (geographic)."""

    def test_for_each_point_distance_multipoint_box(self) -> None:
        """Test for_each_point_distance with MultiPoint and Box."""
        # Create multipoint
        x = np.array([0.0, 2.5, 10.0])
        y = np.array([0.0, 2.5, 10.0])
        source = MultiPoint(x, y)

        # Create a box from (0, 0) to (5, 5)
        container = Box((0.0, 0.0), (5.0, 5.0))

        result = for_each_point_distance(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) == 3
        # Points inside/on boundary should have distance 0
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        # Point outside should have positive distance
        assert result[2] > 0.0

    def test_for_each_point_distance_multipoint_polygon(self) -> None:
        """Test for_each_point_distance with MultiPoint and Polygon."""
        # Create multipoint
        x = np.array([2.5, 6.0])
        y = np.array([2.5, 6.0])
        source = MultiPoint(x, y)

        # Create a square polygon from (0, 0) to (5, 5)
        ring_x = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
        ring_y = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
        container = Polygon(Ring(ring_x, ring_y))

        result = for_each_point_distance(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) == 2
        # Point inside should have distance 0
        assert result[0] == pytest.approx(0.0)
        # Point outside should have positive distance
        assert result[1] > 0.0

    def test_for_each_point_distance_linestring_box(self) -> None:
        """Test for_each_point_distance with LineString and Box."""
        # Create a linestring
        x = np.array([0.5, 2.5, 7.5])
        y = np.array([0.5, 2.5, 7.5])
        source = LineString(x, y)

        # Create a box
        container = Box((0.0, 0.0), (5.0, 5.0))

        result = for_each_point_distance(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) == 3
        # Points inside or on boundary should have distance 0
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        # Point outside should have positive distance
        assert result[2] > 0.0

    def test_for_each_point_distance_ring_polygon(self) -> None:
        """Test for_each_point_distance with Ring and Polygon."""
        # Create a ring
        ring_x = np.array([1.0, 2.0, 2.0, 1.0, 1.0])
        ring_y = np.array([1.0, 1.0, 2.0, 2.0, 1.0])
        source = Ring(ring_x, ring_y)

        # Create a larger polygon
        container_ring_x = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
        container_ring_y = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
        container = Polygon(Ring(container_ring_x, container_ring_y))

        result = for_each_point_distance(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) == 5
        # All points should be inside, so all distances should be 0
        assert np.allclose(result, 0.0)

    def test_for_each_point_distance_all_outside(self) -> None:
        """Test for_each_point_distance with all points outside."""
        # Create multipoint far from container
        x = np.array([100.0, 150.0, 170.0])
        y = np.array([100.0, 150.0, 170.0])
        source = MultiPoint(x, y)

        # Create a box at origin
        container = Box((0.0, 0.0), (5.0, 5.0))

        result = for_each_point_distance(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) == 3
        # All distances should be positive
        assert np.all(result > 0.0)

    def test_for_each_point_distance_all_inside(self) -> None:
        """Test for_each_point_distance with all points inside."""
        # Create multipoint entirely within box
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        source = MultiPoint(x, y)

        # Create a large box
        container = Box((0.0, 0.0), (10.0, 10.0))

        result = for_each_point_distance(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) == 4
        # All distances should be 0
        assert np.allclose(result, 0.0)

    def test_for_each_point_distance_with_spheroid(self) -> None:
        """Test for_each_point_distance with explicit spheroid."""
        # Create multipoint
        x = np.array([0.0, 2.5])
        y = np.array([0.0, 2.5])
        source = MultiPoint(x, y)

        # Create a box
        container = Box((0.0, 0.0), (5.0, 5.0))

        # Test with default spheroid
        result_default = for_each_point_distance(source, container)

        # Test with explicit WGS84 spheroid
        wgs84 = Spheroid()  # WGS84 is the default
        result_wgs84 = for_each_point_distance(
            source, container, spheroid=wgs84
        )

        assert isinstance(result_default, np.ndarray)
        assert isinstance(result_wgs84, np.ndarray)
        # Results should be similar
        assert np.allclose(result_default, result_wgs84)

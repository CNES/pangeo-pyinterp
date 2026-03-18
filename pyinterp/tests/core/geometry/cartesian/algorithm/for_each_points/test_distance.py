# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for for_each_point_X algorithms (Cartesian)."""

from __future__ import annotations

import numpy as np
import pytest

from .......core.geometry.cartesian import (
    Box,
    LineString,
    MultiPoint,
    Polygon,
    Ring,
)
from .......core.geometry.cartesian.algorithms import (
    for_each_point_distance,
    for_each_point_pairwise_distance,
)


class TestForEachPointDistance:
    """Tests for for_each_point_distance algorithm."""

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
        x = np.array([100.0, 200.0, 300.0])
        y = np.array([100.0, 200.0, 300.0])
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


class TestForEachPointPairwiseDistance:
    """Tests for for_each_point_pairwise_distance algorithm."""

    def test_for_each_point_pairwise_distance_multipoint(self) -> None:
        """Test pairwise distances for multipoints."""
        geometry1 = MultiPoint(
            np.array([0.0, 1.0, 4.0]), np.array([0.0, 1.0, 4.0])
        )
        geometry2 = MultiPoint(
            np.array([0.0, 2.0, 4.0]), np.array([0.0, 2.0, 7.0])
        )

        result = for_each_point_pairwise_distance(geometry1, geometry2)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) == 3
        assert np.allclose(result, np.array([0.0, np.sqrt(2.0), 3.0]))

    def test_for_each_point_pairwise_distance_linestring(self) -> None:
        """Test pairwise distances for linestrings."""
        geometry1 = LineString(np.array([0.0, 2.0]), np.array([0.0, 2.0]))
        geometry2 = LineString(np.array([0.0, 2.0]), np.array([0.0, 5.0]))

        result = for_each_point_pairwise_distance(geometry1, geometry2)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) == 2
        assert np.allclose(result, np.array([0.0, 3.0]))

    def test_for_each_point_pairwise_distance_ring(self) -> None:
        """Test pairwise distances for rings."""
        geometry1 = Ring(
            np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 1.0, 0.0, 0.0]),
        )
        geometry2 = Ring(
            np.array([0.0, 0.0, 2.0, 2.0, 0.0]),
            np.array([0.0, 1.0, 1.0, 0.0, 0.0]),
        )

        result = for_each_point_pairwise_distance(geometry1, geometry2)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) == 5
        assert np.allclose(result, np.array([0.0, 0.0, 1.0, 1.0, 0.0]))

    def test_for_each_point_pairwise_distance_size_mismatch(self) -> None:
        """Test pairwise distances with different number of points."""
        geometry1 = MultiPoint(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        geometry2 = MultiPoint(np.array([0.0]), np.array([0.0]))

        with pytest.raises(
            ValueError,
            match="Source and target geometries must have the same "
            "number of points",
        ):
            for_each_point_pairwise_distance(geometry1, geometry2)

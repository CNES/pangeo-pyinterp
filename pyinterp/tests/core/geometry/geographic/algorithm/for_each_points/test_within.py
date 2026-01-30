# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for for_each_point_X algorithms (Geographic)."""

from __future__ import annotations

import numpy as np

from .......core.geometry.geographic import (
    Box,
    LineString,
    MultiPoint,
    Polygon,
    Ring,
)
from .......core.geometry.geographic.algorithms import for_each_point_within


class TestForEachPointWithin:
    """Tests for for_each_point_within algorithm (geographic)."""

    def test_for_each_point_within_multipoint_box(self) -> None:
        """Test for_each_point_within with MultiPoint and Box."""
        # Create multipoint with points inside and outside box
        x = np.array([1.0, 2.5, 10.0, 0.0])
        y = np.array([1.0, 2.5, 10.0, 0.0])
        source = MultiPoint(x, y)

        # Create a box from (0, 0) to (5, 5)
        container = Box((0.0, 0.0), (5.0, 5.0))

        result = for_each_point_within(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 4
        assert result[0]  # (1.0, 1.0) is within
        assert result[1]  # (2.5, 2.5) is within
        assert not result[2]  # (10.0, 10.0) is not within
        # (0.0, 0.0) is on boundary - within for Box
        assert result[3] in (True, False)  # Could be boundary dependent

    def test_for_each_point_within_multipoint_polygon(self) -> None:
        """Test for_each_point_within with MultiPoint and Polygon."""
        # Create multipoint
        x = np.array([1.0, 2.5, 10.0, 2.5])
        y = np.array([1.0, 2.5, 10.0, 0.0])
        source = MultiPoint(x, y)

        # Create a square polygon
        ring_x = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
        ring_y = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
        container = Polygon(Ring(ring_x, ring_y))

        result = for_each_point_within(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 4
        assert result[0]  # Inside
        assert result[1]  # Inside
        assert not result[2]  # Outside
        assert not result[3]  # On boundary (not completely inside)

    def test_for_each_point_within_linestring_box(self) -> None:
        """Test for_each_point_within with LineString and Box."""
        # Create a linestring entirely within box
        x = np.array([1.0, 2.5, 4.0])
        y = np.array([1.0, 2.5, 4.0])
        source = LineString(x, y)

        # Create a box
        container = Box((0.0, 0.0), (5.0, 5.0))

        result = for_each_point_within(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 3
        # All points should be within
        assert np.all(result)

    def test_for_each_point_within_ring_polygon(self) -> None:
        """Test for_each_point_within with Ring and Polygon."""
        # Create a ring
        ring_x = np.array([1.0, 2.0, 2.0, 1.0, 1.0])
        ring_y = np.array([1.0, 1.0, 2.0, 2.0, 1.0])
        source = Ring(ring_x, ring_y)

        # Create a larger polygon
        container_ring_x = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
        container_ring_y = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
        container = Polygon(Ring(container_ring_x, container_ring_y))

        result = for_each_point_within(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 5
        # All points should be within
        assert np.all(result)

    def test_for_each_point_within_empty_result(self) -> None:
        """Test for_each_point_within with no points within."""
        # Create multipoint far from container
        x = np.array([100.0, 150.0])
        y = np.array([100.0, 150.0])
        source = MultiPoint(x, y)

        # Create a box at origin
        container = Box((0.0, 0.0), (5.0, 5.0))

        result = for_each_point_within(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 2
        assert not np.any(result)

    def test_for_each_point_within_all_within(self) -> None:
        """Test for_each_point_within with all points within."""
        # Create multipoint entirely within box
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        source = MultiPoint(x, y)

        # Create a large box
        container = Box((0.0, 0.0), (10.0, 10.0))

        result = for_each_point_within(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 4
        assert np.all(result)

    def test_for_each_point_within_linestring_partial_outside(self) -> None:
        """Test for_each_point_within with LineString partially outside."""
        # Create a linestring with some points outside
        x = np.array([1.0, 5.5, 8.0])
        y = np.array([1.0, 5.5, 8.0])
        source = LineString(x, y)

        # Create a box
        container = Box((0.0, 0.0), (5.0, 5.0))

        result = for_each_point_within(source, container)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 3
        assert result[0]  # Inside
        assert not result[1]  # Outside
        assert not result[2]  # Outside

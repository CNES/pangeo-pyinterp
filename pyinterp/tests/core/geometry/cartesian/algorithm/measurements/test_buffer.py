# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for buffer algorithm and strategy classes."""

from __future__ import annotations

import numpy as np

from .......core.geometry.cartesian import LineString, Point, Polygon, Ring
from .......core.geometry.cartesian.algorithms import (
    DistanceAsymmetric,
    DistanceSymmetric,
    EndFlat,
    EndRound,
    JoinMiter,
    JoinRound,
    PointCircle,
    PointSquare,
    buffer,
)


def test_distance_symmetric_construction() -> None:
    """Test DistanceSymmetric strategy construction."""
    strategy = DistanceSymmetric(5.0)
    assert strategy is not None


def test_distance_asymmetric_construction() -> None:
    """Test DistanceAsymmetric strategy construction."""
    strategy = DistanceAsymmetric(3.0, 7.0)
    assert strategy is not None


def test_end_flat_construction() -> None:
    """Test EndFlat strategy construction."""
    strategy = EndFlat()
    assert strategy is not None


def test_end_round_construction() -> None:
    """Test EndRound strategy construction."""
    # Default construction
    strategy1 = EndRound()
    assert strategy1 is not None

    # Construction with points_per_circle parameter
    strategy2 = EndRound(points_per_circle=16)
    assert strategy2 is not None


def test_join_miter_construction() -> None:
    """Test JoinMiter strategy construction."""
    # Default construction
    strategy1 = JoinMiter()
    assert strategy1 is not None

    # Construction with miter_limit parameter
    strategy2 = JoinMiter(miter_limit=5.0)
    assert strategy2 is not None


def test_join_round_construction() -> None:
    """Test JoinRound strategy construction."""
    # Default construction
    strategy1 = JoinRound()
    assert strategy1 is not None

    # Construction with points_per_circle parameter
    strategy2 = JoinRound(points_per_circle=16)
    assert strategy2 is not None


def test_point_circle_construction() -> None:
    """Test PointCircle strategy construction."""
    # Default construction
    strategy1 = PointCircle()
    assert strategy1 is not None

    # Construction with points_per_circle parameter
    strategy2 = PointCircle(points_per_circle=16)
    assert strategy2 is not None


def test_point_square_construction() -> None:
    """Test PointSquare strategy construction."""
    strategy = PointSquare()
    assert strategy is not None


def test_buffer_point() -> None:
    """Test buffer operation on a point."""
    point = Point(5.0, 5.0)

    distance_strategy = DistanceSymmetric(2.0)
    join_strategy = JoinRound()
    end_strategy = EndRound()
    point_strategy = PointCircle()

    result = buffer(
        point, distance_strategy, join_strategy, end_strategy, point_strategy
    )

    # Buffer of a point should create a circular polygon
    # (returned as MultiPolygon)
    assert result is not None
    assert len(result) > 0


def test_buffer_point_square() -> None:
    """Test buffer operation on a point with square strategy."""
    point = Point(5.0, 5.0)

    distance_strategy = DistanceSymmetric(2.0)
    join_strategy = JoinMiter()
    end_strategy = EndFlat()
    point_strategy = PointSquare()

    result = buffer(
        point, distance_strategy, join_strategy, end_strategy, point_strategy
    )

    # Buffer of a point with square strategy should create a square polygon
    assert result is not None
    assert len(result) > 0


def test_buffer_linestring() -> None:
    """Test buffer operation on a linestring."""
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 0.0])
    linestring = LineString(x, y)

    distance_strategy = DistanceSymmetric(1.0)
    join_strategy = JoinRound()
    end_strategy = EndRound()
    point_strategy = PointCircle()

    result = buffer(
        linestring,
        distance_strategy,
        join_strategy,
        end_strategy,
        point_strategy,
    )

    # Buffer of a linestring should create a polygon
    assert result is not None
    assert len(result) > 0


def test_buffer_linestring_flat_ends() -> None:
    """Test buffer operation on a linestring with flat ends."""
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 0.0])
    linestring = LineString(x, y)

    distance_strategy = DistanceSymmetric(1.0)
    join_strategy = JoinRound()
    end_strategy = EndFlat()
    point_strategy = PointCircle()

    result = buffer(
        linestring,
        distance_strategy,
        join_strategy,
        end_strategy,
        point_strategy,
    )

    # Buffer with flat ends should create a rectangular-ish polygon
    assert result is not None
    assert len(result) > 0


def test_buffer_polygon(polygon_1x1: Polygon) -> None:
    """Test buffer operation on a polygon."""
    distance_strategy = DistanceSymmetric(0.5)
    join_strategy = JoinRound()
    end_strategy = EndRound()
    point_strategy = PointCircle()

    result = buffer(
        polygon_1x1,
        distance_strategy,
        join_strategy,
        end_strategy,
        point_strategy,
    )

    # Buffer of a polygon should create a larger polygon
    assert result is not None
    assert len(result) > 0


def test_buffer_polygon_miter_join(polygon_1x1: Polygon) -> None:
    """Test buffer operation on a polygon with miter join."""
    distance_strategy = DistanceSymmetric(0.5)
    join_strategy = JoinMiter(miter_limit=5.0)
    end_strategy = EndRound()
    point_strategy = PointCircle()

    result = buffer(
        polygon_1x1,
        distance_strategy,
        join_strategy,
        end_strategy,
        point_strategy,
    )

    # Buffer with miter join should create sharp corners
    assert result is not None
    assert len(result) > 0


def test_buffer_asymmetric() -> None:
    """Test buffer operation with asymmetric distance."""
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 0.0])
    linestring = LineString(x, y)

    # Different distances on left and right sides
    distance_strategy = DistanceAsymmetric(1.0, 2.0)
    join_strategy = JoinRound()
    end_strategy = EndRound()
    point_strategy = PointCircle()

    result = buffer(
        linestring,
        distance_strategy,
        join_strategy,
        end_strategy,
        point_strategy,
    )

    # Asymmetric buffer should create an asymmetric polygon
    assert result is not None
    assert len(result) > 0


def test_buffer_negative_distance() -> None:
    """Test buffer operation with negative distance (erosion)."""
    x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    polygon = Polygon(Ring(x, y))

    # Negative buffer (erosion)
    distance_strategy = DistanceSymmetric(-1.0)
    join_strategy = JoinRound()
    end_strategy = EndRound()
    point_strategy = PointCircle()

    result = buffer(
        polygon, distance_strategy, join_strategy, end_strategy, point_strategy
    )

    # Negative buffer should create a smaller polygon (or empty if too large)
    assert result is not None

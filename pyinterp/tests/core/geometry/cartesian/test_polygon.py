# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test Polygon class."""

from __future__ import annotations

import pickle

import numpy as np

from .....core.geometry.cartesian import Point, Polygon, Ring


def test_polygon_construction() -> None:
    """Test Polygon construction."""
    # Default construction (empty)
    p1 = Polygon()
    assert len(p1.outer) == 0
    assert len(p1.inners) == 0

    # Construction with exterior ring
    x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    exterior = Ring(x, y)
    p2 = Polygon(exterior)
    assert len(p2.outer) == 5
    assert len(p2.inners) == 0

    # Construction with exterior and interior rings
    x_inner = np.array([2.0, 8.0, 8.0, 2.0, 2.0])
    y_inner = np.array([2.0, 2.0, 8.0, 8.0, 2.0])
    interior = Ring(x_inner, y_inner)
    p3 = Polygon(exterior, [interior])
    assert len(p3.outer) == 5
    assert len(p3.inners) == 1


def test_polygon_outer_ring() -> None:
    """Test Polygon outer ring access."""
    x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    exterior = Ring(x, y)
    polygon = Polygon(exterior)

    outer = polygon.outer
    assert len(outer) == 5
    assert outer[0] == Point(0.0, 0.0)


def test_polygon_inner_rings() -> None:
    """Test Polygon inner rings access."""
    x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    exterior = Ring(x, y)

    x_inner = np.array([2.0, 8.0, 8.0, 2.0, 2.0])
    y_inner = np.array([2.0, 2.0, 8.0, 8.0, 2.0])
    interior = Ring(x_inner, y_inner)

    polygon = Polygon(exterior, [interior])

    assert len(polygon.inners) == 1
    assert polygon.inners[0] == interior


def test_polygon_append() -> None:
    """Test Polygon append operation (add interior ring)."""
    x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    exterior = Ring(x, y)
    polygon = Polygon(exterior)

    assert len(polygon.inners) == 0

    x_inner = np.array([2.0, 8.0, 8.0, 2.0, 2.0])
    y_inner = np.array([2.0, 2.0, 8.0, 8.0, 2.0])
    interior = Ring(x_inner, y_inner)

    polygon.append(interior)
    assert len(polygon.inners) == 1
    assert polygon.inners[0] == interior


def test_polygon_clear() -> None:
    """Test Polygon clear operation."""
    x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    exterior = Ring(x, y)

    x_inner = np.array([2.0, 8.0, 8.0, 2.0, 2.0])
    y_inner = np.array([2.0, 2.0, 8.0, 8.0, 2.0])
    interior = Ring(x_inner, y_inner)

    polygon = Polygon(exterior, [interior])
    assert len(polygon.inners) == 1

    polygon.clear()
    assert len(polygon.inners) == 0


def test_polygon_equality() -> None:
    """Test Polygon equality comparison."""
    x1 = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y1 = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    p1 = Polygon(Ring(x1, y1))

    x2 = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y2 = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    p2 = Polygon(Ring(x2, y2))

    x3 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    y3 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    p3 = Polygon(Ring(x3, y3))

    assert p1 == p2
    assert p1 != p3


def test_polygon_repr() -> None:
    """Test Polygon string representation."""
    x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    polygon = Polygon(Ring(x, y))
    repr_str = repr(polygon)

    # Should contain class name
    assert "Polygon" in repr_str


def test_polygon_pickle() -> None:
    """Test Polygon serialization with pickle."""
    x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    exterior = Ring(x, y)

    x_inner = np.array([2.0, 8.0, 8.0, 2.0, 2.0])
    y_inner = np.array([2.0, 2.0, 8.0, 8.0, 2.0])
    interior = Ring(x_inner, y_inner)

    original = Polygon(exterior, [interior])

    # Pickle and unpickle
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)

    # Should be equal after round-trip
    assert restored == original
    assert restored.outer == original.outer
    assert len(restored.inners) == len(original.inners)

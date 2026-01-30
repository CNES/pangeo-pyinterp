# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test Ring class."""

from __future__ import annotations

import pickle

import numpy as np

from .....core.geometry.cartesian import Point, Ring


def test_ring_construction() -> None:
    """Test Ring construction."""
    # Default construction (empty ring)
    r1 = Ring()
    assert len(r1) == 0

    # Construction with coordinates
    x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    r2 = Ring(x, y)
    assert len(r2) == 5


def test_ring_indexing() -> None:
    """Test Ring indexing."""
    x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    ring = Ring(x, y)

    # Get items
    assert ring[0] == Point(0.0, 0.0)
    assert ring[1] == Point(10.0, 0.0)
    assert ring[4] == Point(0.0, 0.0)

    # Set items
    ring[0] = Point(1.0, 1.0)
    assert ring[0] == Point(1.0, 1.0)


def test_ring_iteration() -> None:
    """Test Ring iteration."""
    x = np.array([0.0, 10.0, 10.0])
    y = np.array([0.0, 0.0, 10.0])
    ring = Ring(x, y)

    points = list(ring)
    assert len(points) == 3
    assert points[0] == Point(0.0, 0.0)
    assert points[1] == Point(10.0, 0.0)
    assert points[2] == Point(10.0, 10.0)


def test_ring_length() -> None:
    """Test Ring length."""
    x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    ring = Ring(x, y)

    assert len(ring) == 5


def test_ring_append() -> None:
    """Test Ring append operation."""
    ring = Ring()
    assert len(ring) == 0

    ring.append(Point(0.0, 0.0))
    assert len(ring) == 1
    assert ring[0] == Point(0.0, 0.0)

    ring.append(Point(10.0, 0.0))
    assert len(ring) == 2
    assert ring[1] == Point(10.0, 0.0)


def test_ring_clear() -> None:
    """Test Ring clear operation."""
    x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    ring = Ring(x, y)

    assert len(ring) == 5
    ring.clear()
    assert len(ring) == 0


def test_ring_bool() -> None:
    """Test Ring boolean conversion."""
    # Empty ring
    r1 = Ring()
    assert bool(r1) is False

    # Non-empty ring
    x = np.array([0.0, 10.0, 10.0])
    y = np.array([0.0, 0.0, 10.0])
    r2 = Ring(x, y)
    assert bool(r2) is True


def test_ring_equality() -> None:
    """Test Ring equality comparison."""
    x1 = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y1 = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    r1 = Ring(x1, y1)

    x2 = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y2 = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    r2 = Ring(x2, y2)

    x3 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    y3 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    r3 = Ring(x3, y3)

    assert r1 == r2
    assert r1 != r3


def test_ring_repr() -> None:
    """Test Ring string representation."""
    x = np.array([0.0, 10.0, 10.0])
    y = np.array([0.0, 0.0, 10.0])
    ring = Ring(x, y)
    repr_str = repr(ring)

    # Should contain class name
    assert "Ring" in repr_str


def test_ring_pickle() -> None:
    """Test Ring serialization with pickle."""
    x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    original = Ring(x, y)

    # Pickle and unpickle
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)

    # Should be equal after round-trip
    assert restored == original
    assert len(restored) == len(original)
    for i in range(len(original)):
        assert restored[i] == original[i]


def test_ring_to_arrays() -> None:
    """Test Ring to_arrays method."""
    # Test with populated ring
    x = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    ring = Ring(x, y)

    x_result, y_result = ring.to_arrays()

    # Verify returned arrays match input
    np.testing.assert_array_equal(x_result, x)
    np.testing.assert_array_equal(y_result, y)

    # Verify returned arrays are numpy arrays
    assert isinstance(x_result, np.ndarray)
    assert isinstance(y_result, np.ndarray)

    # Verify array shapes
    assert x_result.shape == (5,)
    assert y_result.shape == (5,)

    # Test with empty ring
    empty_ring = Ring()
    x_empty, y_empty = empty_ring.to_arrays()
    assert len(x_empty) == 0
    assert len(y_empty) == 0

    # Test with triangular ring
    x_tri = np.array([0.0, 5.0, 2.5])
    y_tri = np.array([0.0, 0.0, 5.0])
    ring_tri = Ring(x_tri, y_tri)
    x_tri_result, y_tri_result = ring_tri.to_arrays()
    np.testing.assert_array_equal(x_tri_result, x_tri)
    np.testing.assert_array_equal(y_tri_result, y_tri)

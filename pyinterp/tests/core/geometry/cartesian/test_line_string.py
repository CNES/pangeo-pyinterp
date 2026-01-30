# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test LineString class."""

from __future__ import annotations

import pickle

import numpy as np

from .....core.geometry.cartesian import LineString, Point


def test_linestring_construction() -> None:
    """Test LineString construction."""
    # Default construction (empty)
    ls1 = LineString()
    assert len(ls1) == 0

    # Construction with coordinates
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([0.0, 5.0, 0.0])
    ls2 = LineString(x, y)
    assert len(ls2) == 3


def test_linestring_indexing() -> None:
    """Test LineString indexing."""
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([0.0, 5.0, 0.0])
    ls = LineString(x, y)

    # Get items
    assert ls[0] == Point(0.0, 0.0)
    assert ls[1] == Point(5.0, 5.0)
    assert ls[2] == Point(10.0, 0.0)

    # Set items
    ls[1] = Point(6.0, 6.0)
    assert ls[1] == Point(6.0, 6.0)


def test_linestring_iteration() -> None:
    """Test LineString iteration."""
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([0.0, 5.0, 0.0])
    ls = LineString(x, y)

    points = list(ls)
    assert len(points) == 3
    assert points[0] == Point(0.0, 0.0)
    assert points[1] == Point(5.0, 5.0)
    assert points[2] == Point(10.0, 0.0)


def test_linestring_length() -> None:
    """Test LineString length."""
    x = np.array([0.0, 5.0, 10.0, 15.0])
    y = np.array([0.0, 5.0, 0.0, 5.0])
    ls = LineString(x, y)

    assert len(ls) == 4


def test_linestring_append() -> None:
    """Test LineString append operation."""
    ls = LineString()
    assert len(ls) == 0

    ls.append(Point(0.0, 0.0))
    assert len(ls) == 1
    assert ls[0] == Point(0.0, 0.0)

    ls.append(Point(5.0, 5.0))
    assert len(ls) == 2
    assert ls[1] == Point(5.0, 5.0)


def test_linestring_clear() -> None:
    """Test LineString clear operation."""
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([0.0, 5.0, 0.0])
    ls = LineString(x, y)

    assert len(ls) == 3
    ls.clear()
    assert len(ls) == 0


def test_linestring_bool() -> None:
    """Test LineString boolean conversion."""
    # Empty linestring
    ls1 = LineString()
    assert bool(ls1) is False

    # Non-empty linestring
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([0.0, 5.0, 0.0])
    ls2 = LineString(x, y)
    assert bool(ls2) is True


def test_linestring_equality() -> None:
    """Test LineString equality comparison."""
    x1 = np.array([0.0, 5.0, 10.0])
    y1 = np.array([0.0, 5.0, 0.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([0.0, 5.0, 10.0])
    y2 = np.array([0.0, 5.0, 0.0])
    ls2 = LineString(x2, y2)

    x3 = np.array([0.0, 5.0])
    y3 = np.array([0.0, 5.0])
    ls3 = LineString(x3, y3)

    assert ls1 == ls2
    assert ls1 != ls3


def test_linestring_repr() -> None:
    """Test LineString string representation."""
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([0.0, 5.0, 0.0])
    ls = LineString(x, y)
    repr_str = repr(ls)

    # Should contain class name
    assert "LineString" in repr_str


def test_linestring_pickle() -> None:
    """Test LineString serialization with pickle."""
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([0.0, 5.0, 0.0])
    original = LineString(x, y)

    # Pickle and unpickle
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)

    # Should be equal after round-trip
    assert restored == original
    assert len(restored) == len(original)
    for i in range(len(original)):
        assert restored[i] == original[i]


def test_linestring_to_arrays() -> None:
    """Test LineString to_arrays method."""
    # Test with populated linestring
    x = np.array([0.0, 5.0, 10.0, 15.0])
    y = np.array([0.0, 5.0, 10.0, 15.0])
    ls = LineString(x, y)

    x_result, y_result = ls.to_arrays()

    # Verify returned arrays match input
    np.testing.assert_array_equal(x_result, x)
    np.testing.assert_array_equal(y_result, y)

    # Verify returned arrays are numpy arrays
    assert isinstance(x_result, np.ndarray)
    assert isinstance(y_result, np.ndarray)

    # Verify array shapes
    assert x_result.shape == (4,)
    assert y_result.shape == (4,)

    # Test with empty linestring
    empty_ls = LineString()
    x_empty, y_empty = empty_ls.to_arrays()
    assert len(x_empty) == 0
    assert len(y_empty) == 0

    # Test with single point
    ls_single = LineString(np.array([1.0]), np.array([2.0]))
    x_single, y_single = ls_single.to_arrays()
    np.testing.assert_array_equal(x_single, np.array([1.0]))
    np.testing.assert_array_equal(y_single, np.array([2.0]))

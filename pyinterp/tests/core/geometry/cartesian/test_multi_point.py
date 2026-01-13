# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test MultiPoint class."""

import pickle

import numpy as np

from .....core.geometry.cartesian import MultiPoint, Point


def test_multipoint_construction() -> None:
    """Test MultiPoint construction."""
    # Default construction (empty)
    mp1 = MultiPoint()
    assert len(mp1) == 0

    # Construction with points
    points = [Point(0.0, 0.0), Point(10.0, 10.0), Point(20.0, 20.0)]
    mp2 = MultiPoint(points)
    assert len(mp2) == 3

    # Construction with separate coordinate arrays
    xs = np.arange(0.0, 30.0, 10.0, dtype=np.float64)
    ys = np.arange(0.0, 30.0, 10.0, dtype=np.float64)
    mp3 = MultiPoint(xs, ys)
    assert len(mp3) == 3
    for i in range(3):
        assert mp3[i] == Point(xs[i], ys[i])


def test_multipoint_indexing() -> None:
    """Test MultiPoint indexing."""
    points = [Point(0.0, 0.0), Point(10.0, 10.0), Point(20.0, 20.0)]
    mp = MultiPoint(points)

    # Get items
    assert mp[0] == Point(0.0, 0.0)
    assert mp[1] == Point(10.0, 10.0)
    assert mp[2] == Point(20.0, 20.0)

    # Set items
    mp[1] = Point(15.0, 15.0)
    assert mp[1] == Point(15.0, 15.0)


def test_multipoint_iteration() -> None:
    """Test MultiPoint iteration."""
    points = [Point(0.0, 0.0), Point(10.0, 10.0), Point(20.0, 20.0)]
    mp = MultiPoint(points)

    result = list(mp)
    assert len(result) == 3
    assert result[0] == Point(0.0, 0.0)
    assert result[1] == Point(10.0, 10.0)
    assert result[2] == Point(20.0, 20.0)


def test_multipoint_length() -> None:
    """Test MultiPoint length."""
    points = [Point(0.0, 0.0), Point(10.0, 10.0)]
    mp = MultiPoint(points)

    assert len(mp) == 2


def test_multipoint_append() -> None:
    """Test MultiPoint append operation."""
    mp = MultiPoint()
    assert len(mp) == 0

    mp.append(Point(0.0, 0.0))
    assert len(mp) == 1
    assert mp[0] == Point(0.0, 0.0)

    mp.append(Point(10.0, 10.0))
    assert len(mp) == 2
    assert mp[1] == Point(10.0, 10.0)


def test_multipoint_clear() -> None:
    """Test MultiPoint clear operation."""
    points = [Point(0.0, 0.0), Point(10.0, 10.0), Point(20.0, 20.0)]
    mp = MultiPoint(points)

    assert len(mp) == 3
    mp.clear()
    assert len(mp) == 0


def test_multipoint_bool() -> None:
    """Test MultiPoint boolean conversion."""
    # Empty multipoint
    mp1 = MultiPoint()
    assert bool(mp1) is False

    # Non-empty multipoint
    mp2 = MultiPoint([Point(0.0, 0.0)])
    assert bool(mp2) is True


def test_multipoint_equality() -> None:
    """Test MultiPoint equality comparison."""
    points1 = [Point(0.0, 0.0), Point(10.0, 10.0)]
    mp1 = MultiPoint(points1)

    points2 = [Point(0.0, 0.0), Point(10.0, 10.0)]
    mp2 = MultiPoint(points2)

    points3 = [Point(0.0, 0.0), Point(20.0, 20.0)]
    mp3 = MultiPoint(points3)

    assert mp1 == mp2
    assert mp1 != mp3


def test_multipoint_repr() -> None:
    """Test MultiPoint string representation."""
    points = [Point(0.0, 0.0), Point(10.0, 10.0)]
    mp = MultiPoint(points)
    repr_str = repr(mp)

    # Should contain class name
    assert "MultiPoint" in repr_str


def test_multipoint_pickle() -> None:
    """Test MultiPoint serialization with pickle."""
    points = [Point(0.0, 0.0), Point(10.0, 10.0), Point(20.0, 20.0)]
    original = MultiPoint(points)

    # Pickle and unpickle
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)

    # Should be equal after round-trip
    assert restored == original
    assert len(restored) == len(original)
    for i in range(len(original)):
        assert restored[i] == original[i]


def test_multipoint_iteration_and_len() -> None:
    """Test MultiPoint iteration and length."""
    points = [Point(0.0, 0.0), Point(10.0, 10.0)]
    mp = MultiPoint(points)

    assert len(mp) == 2
    items = list(mp)
    assert items[0] == Point(0.0, 0.0)
    assert items[1] == Point(10.0, 10.0)

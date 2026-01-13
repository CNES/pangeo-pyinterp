# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test Point class."""

import pickle

from .....core.geometry.cartesian import Point


def test_point_construction() -> None:
    """Test Point construction."""
    # Default construction
    p1 = Point()
    assert p1.x == 0.0
    assert p1.y == 0.0

    # Construction with coordinates (x, y semantics but lon/lat attribute names)
    p2 = Point(2.3, 48.9)
    assert p2.x == 2.3
    assert p2.y == 48.9

    # Construction with keyword arguments
    p3 = Point(x=10.5, y=20.5)
    assert p3.x == 10.5
    assert p3.y == 20.5


def test_point_properties() -> None:
    """Test Point property access and mutation."""
    point = Point(10.0, 20.0)

    # Read properties
    assert point.x == 10.0
    assert point.y == 20.0

    # Write properties
    point.x = 15.0
    point.y = 25.0
    assert point.x == 15.0
    assert point.y == 25.0


def test_point_equality() -> None:
    """Test Point equality comparison."""
    p1 = Point(2.3, 48.9)
    p2 = Point(2.3, 48.9)
    p3 = Point(2.4, 48.9)

    assert p1 == p2
    assert p1 != p3
    assert p2 != p3


def test_point_hash() -> None:
    """Test Point hashing."""
    p1 = Point(10.0, 20.0)
    p2 = Point(10.0, 20.0)
    p3 = Point(15.0, 25.0)

    # Equal points should have equal hashes
    assert hash(p1) == hash(p2)

    # Different points should likely have different hashes
    assert hash(p1) != hash(p3)

    # Points should be usable in sets and dicts
    point_set = {p1, p2, p3}
    assert len(point_set) == 2  # p1 and p2 are equal


def test_point_repr() -> None:
    """Test Point string representation."""
    point = Point(2.3, 48.9)
    repr_str = repr(point)

    # Should contain class name and coordinates
    assert "Point" in repr_str
    assert "2.3" in repr_str
    assert "48.9" in repr_str


def test_point_pickle() -> None:
    """Test Point serialization with pickle."""
    original = Point(100.5, 200.5)

    # Pickle and unpickle
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)

    # Should be equal after round-trip
    assert restored == original
    assert restored.x == original.x
    assert restored.y == original.y


def test_point_with_negative_coordinates() -> None:
    """Test Point with negative coordinates (valid in Cartesian)."""
    point = Point(-10.0, -20.0)
    assert point.x == -10.0
    assert point.y == -20.0


def test_point_with_large_coordinates() -> None:
    """Test Point with large coordinates."""
    point = Point(1e6, 1e6)
    assert point.x == 1e6
    assert point.y == 1e6

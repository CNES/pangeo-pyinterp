# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test Box class."""

from __future__ import annotations

import pickle

from .....core.geometry.cartesian import Box, Point


def test_box_construction() -> None:
    """Test Box construction."""
    # Default construction
    b1 = Box()
    assert b1.min_corner == Point(0.0, 0.0)
    assert b1.max_corner == Point(0.0, 0.0)

    # Construction with corners
    b2 = Box((0.0, 0.0), (10.0, 20.0))
    assert b2.min_corner == Point(0.0, 0.0)
    assert b2.max_corner == Point(10.0, 20.0)


def test_box_corners() -> None:
    """Test Box corner access."""
    box = Box((1.0, 2.0), (11.0, 22.0))

    min_corner = box.min_corner
    assert min_corner.x == 1.0
    assert min_corner.y == 2.0

    max_corner = box.max_corner
    assert max_corner.x == 11.0
    assert max_corner.y == 22.0


def test_box_equality() -> None:
    """Test Box equality comparison."""
    b1 = Box((0.0, 0.0), (10.0, 10.0))
    b2 = Box((0.0, 0.0), (10.0, 10.0))
    b3 = Box((0.0, 0.0), (10.0, 11.0))

    assert b1 == b2
    assert b1 != b3
    assert b2 != b3


def test_box_hash() -> None:
    """Test Box hashing."""
    b1 = Box((0.0, 0.0), (10.0, 10.0))
    b2 = Box((0.0, 0.0), (10.0, 10.0))
    b3 = Box((5.0, 5.0), (15.0, 15.0))

    # Equal boxes should have equal hashes
    assert hash(b1) == hash(b2)

    # Different boxes should likely have different hashes
    assert hash(b1) != hash(b3)

    # Boxes should be usable in sets
    box_set = {b1, b2, b3}
    assert len(box_set) == 2


def test_box_repr() -> None:
    """Test Box string representation."""
    box = Box((0.0, 0.0), (10.0, 20.0))
    repr_str = repr(box)

    # Should contain class name
    assert "Box" in repr_str


def test_box_pickle() -> None:
    """Test Box serialization with pickle."""
    original = Box((1.0, 2.0), (11.0, 22.0))

    # Pickle and unpickle
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)

    # Should be equal after round-trip
    assert restored == original
    assert restored.min_corner == original.min_corner
    assert restored.max_corner == original.max_corner


def test_box_zero_area() -> None:
    """Test Box with zero area (point box)."""
    box = Box((5.0, 5.0), (5.0, 5.0))
    assert box.min_corner == box.max_corner


def test_box_with_negative_coordinates() -> None:
    """Test Box with negative coordinates."""
    box = Box((-10.0, -20.0), (10.0, 20.0))
    assert box.min_corner == Point(-10.0, -20.0)
    assert box.max_corner == Point(10.0, 20.0)

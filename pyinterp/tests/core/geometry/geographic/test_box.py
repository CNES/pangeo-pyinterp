# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test Box class."""

import pickle

import pytest

from .....core.geometry.geographic import Box, Point


def test_box_construction() -> None:
    """Test Box construction."""
    # Default construction
    b1 = Box()
    assert b1.min_corner == Point(0.0, 0.0)
    assert b1.max_corner == Point(0.0, 0.0)

    # Construction with numpy arrays
    b2 = Box((-10.0, 40.0), (10.0, 50.0))
    assert b2.min_corner.lon == -10.0
    assert b2.min_corner.lat == 40.0
    assert b2.max_corner.lon == 10.0
    assert b2.max_corner.lat == 50.0


def test_box_construction_validation() -> None:
    """Test Box construction validates array sizes."""
    # Arrays must have exactly 2 elements
    with pytest.raises(TypeError, match="incompatible function arguments"):
        Box((1.0, 2.0, 3.0), (4.0, 5.0))  # type: ignore[arg-type]


def test_box_corners() -> None:
    """Test Box corner accessors."""
    box = Box((-122.5, 37.7), (-122.3, 37.8))

    min_corner = box.min_corner
    assert min_corner.lon == -122.5
    assert min_corner.lat == 37.7

    max_corner = box.max_corner
    assert max_corner.lon == -122.3
    assert max_corner.lat == 37.8


def test_box_centroid() -> None:
    """Test Box centroid calculation."""
    box = Box((-10.0, 40.0), (10.0, 50.0))

    centroid = box.centroid()
    assert centroid.lon == 0.0
    assert centroid.lat == 45.0


def test_box_equality() -> None:
    """Test Box equality comparison."""
    b1 = Box((-10.0, 40.0), (10.0, 50.0))
    b2 = Box((-10.0, 40.0), (10.0, 50.0))
    b3 = Box((-10.0, 40.0), (10.0, 51.0))

    assert b1 == b2
    assert b1 != b3
    assert b2 != b3


def test_box_hash() -> None:
    """Test Box hashing for use in sets and dicts."""
    b1 = Box((-10.0, 40.0), (10.0, 50.0))
    b2 = Box((-10.0, 40.0), (10.0, 50.0))
    b3 = Box((-10.0, 40.0), (10.0, 51.0))

    # Equal boxes should have the same hash
    assert hash(b1) == hash(b2)

    # Can be used in sets
    box_set = {b1, b2, b3}
    assert len(box_set) == 2  # b1 and b2 are equal

    # Can be used as dict keys
    box_dict = {b1: "Europe", b3: "Other"}
    assert box_dict[b2] == "Europe"  # b2 equals b1


def test_box_repr() -> None:
    """Test Box string representation."""
    box = Box((-10.0, 40.0), (10.0, 50.0))

    # __repr__ should show detailed info
    assert repr(box) == "Box(min=(-10, 40), max=(10, 50))"

    # __str__ should show a simple range
    assert str(box) == "[(-10, 40) to (10, 50)]"


def test_box_pickle() -> None:
    """Test Box pickle support."""
    original = Box((-122.5, 37.7), (-122.3, 37.8))

    # Pickle and unpickle
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)

    assert restored.min_corner == original.min_corner
    assert restored.max_corner == original.max_corner
    assert restored == original


def test_box_getstate_setstate() -> None:
    """Test Box __getstate__ and __setstate__ directly."""
    original = Box((-180.0, -90.0), (180.0, 90.0))

    new_box = pickle.loads(pickle.dumps(original))
    assert new_box.min_corner.lon == -180.0
    assert new_box.min_corner.lat == -90.0
    assert new_box.max_corner.lon == 180.0
    assert new_box.max_corner.lat == 90.0
    assert new_box == original


def test_box_edge_cases() -> None:
    """Test Box with edge case values."""
    # Test with global extent
    global_box = Box((-180.0, -90.0), (180.0, 90.0))
    assert global_box.min_corner.lon == -180.0
    assert global_box.max_corner.lat == 90.0

    # Test with very small box
    tiny_box = Box((0.0, 0.0), (1e-10, 1e-10))
    assert abs(tiny_box.max_corner.lon - 1e-10) < 1e-15
    assert abs(tiny_box.max_corner.lat - 1e-10) < 1e-15

    # Test with same min/max (degenerate box)
    point_box = Box((5.0, 10.0), (5.0, 10.0))
    assert point_box.min_corner == point_box.max_corner

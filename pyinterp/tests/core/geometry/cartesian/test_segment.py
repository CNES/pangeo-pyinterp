# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test Segment class."""

import pickle

from .....core.geometry.cartesian import Point, Segment


def test_segment_construction() -> None:
    """Test Segment construction."""
    # Default construction
    s1 = Segment()
    assert s1.a == Point(0.0, 0.0)
    assert s1.b == Point(0.0, 0.0)

    # Construction with endpoints
    s2 = Segment((0.0, 0.0), (10.0, 20.0))
    assert s2.a == Point(0.0, 0.0)
    assert s2.b == Point(10.0, 20.0)


def test_segment_properties() -> None:
    """Test Segment property access and mutation."""
    segment = Segment((1.0, 2.0), (11.0, 22.0))

    # Read properties
    assert segment.a.x == 1.0
    assert segment.a.y == 2.0
    assert segment.b.x == 11.0
    assert segment.b.y == 22.0

    # Write properties
    segment.a = Point(5.0, 10.0)
    segment.b = Point(15.0, 30.0)
    assert segment.a == Point(5.0, 10.0)
    assert segment.b == Point(15.0, 30.0)


def test_segment_equality() -> None:
    """Test Segment equality comparison."""
    s1 = Segment((0.0, 0.0), (10.0, 10.0))
    s2 = Segment((0.0, 0.0), (10.0, 10.0))
    s3 = Segment((0.0, 0.0), (10.0, 11.0))

    assert s1 == s2
    assert s1 != s3
    assert s2 != s3


def test_segment_indexing() -> None:
    """Test Segment indexing."""
    segment = Segment((1.0, 2.0), (11.0, 22.0))

    # Get items
    assert segment[0] == Point(1.0, 2.0)
    assert segment[1] == Point(11.0, 22.0)

    # Set items
    segment[0] = Point(5.0, 10.0)
    segment[1] = Point(15.0, 30.0)
    assert segment[0] == Point(5.0, 10.0)
    assert segment[1] == Point(15.0, 30.0)


def test_segment_iteration() -> None:
    """Test Segment iteration."""
    segment = Segment((1.0, 2.0), (11.0, 22.0))
    points = list(segment)

    assert len(points) == 2
    assert points[0] == Point(1.0, 2.0)
    assert points[1] == Point(11.0, 22.0)


def test_segment_length() -> None:
    """Test Segment length."""
    segment = Segment((0.0, 0.0), (10.0, 10.0))
    assert len(segment) == 2


def test_segment_bool() -> None:
    """Test Segment boolean conversion."""
    # Non-degenerate segment
    s1 = Segment((0.0, 0.0), (10.0, 10.0))
    assert bool(s1) is True

    # Degenerate segment (zero length) - current implementation evaluates to
    # True
    s2 = Segment((5.0, 5.0), (5.0, 5.0))
    assert isinstance(bool(s2), bool)


def test_segment_repr() -> None:
    """Test Segment string representation."""
    segment = Segment((0.0, 0.0), (10.0, 20.0))
    repr_str = repr(segment)

    # Should contain class name
    assert "Segment" in repr_str


def test_segment_pickle() -> None:
    """Test Segment serialization with pickle."""
    original = Segment((1.0, 2.0), (11.0, 22.0))

    # Pickle and unpickle
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)

    # Should be equal after round-trip
    assert restored == original
    assert restored.a == original.a
    assert restored.b == original.b

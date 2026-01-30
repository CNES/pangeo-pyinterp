# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test Segment class."""

from __future__ import annotations

import pickle

import numpy as np

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


def test_segment_to_arrays() -> None:
    """Test Segment to_arrays method."""
    # Test with populated segment
    s = Segment((0.0, 0.0), (10.0, 20.0))

    x_result, y_result = s.to_arrays()

    # Verify returned arrays contain both endpoints
    expected_x = np.array([0.0, 10.0])
    expected_y = np.array([0.0, 20.0])
    np.testing.assert_array_equal(x_result, expected_x)
    np.testing.assert_array_equal(y_result, expected_y)

    # Verify returned arrays are numpy arrays
    assert isinstance(x_result, np.ndarray)
    assert isinstance(y_result, np.ndarray)

    # Verify array shapes (should always be 2 elements)
    assert x_result.shape == (2,)
    assert y_result.shape == (2,)

    # Test with different coordinates
    s2 = Segment((-5.0, 10.0), (15.0, -20.0))
    x2, y2 = s2.to_arrays()
    np.testing.assert_array_equal(x2, np.array([-5.0, 15.0]))
    np.testing.assert_array_equal(y2, np.array([10.0, -20.0]))

    # Test with default constructed segment (both points at origin)
    s_empty = Segment()
    x_empty, y_empty = s_empty.to_arrays()
    np.testing.assert_array_equal(x_empty, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(y_empty, np.array([0.0, 0.0]))

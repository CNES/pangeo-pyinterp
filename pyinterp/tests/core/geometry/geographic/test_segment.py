# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test Segment class."""

import pickle

import numpy as np
import pytest

from .....core.geometry.geographic import Point, Segment


def test_segment_construction() -> None:
    """Test Segment construction and length."""
    s_empty = Segment()
    assert len(s_empty) == 2

    s = Segment((0.0, 0.0), (10.0, 5.0))
    assert len(s) == 2
    assert s[0].lon == 0.0 and s[0].lat == 0.0
    assert s[1].lon == 10.0 and s[1].lat == 5.0


def test_segment_indexing_and_setting() -> None:
    """Test Segment indexing get and set operations."""
    s = Segment((0.0, 0.0), (10.0, 10.0))

    a = s[0]
    b = s[1]
    assert a == Point(0.0, 0.0)
    assert b == Point(10.0, 10.0)

    s[0] = Point(1.0, 2.0)
    s[1] = Point(3.0, 4.0)
    assert s[0] == Point(1.0, 2.0)
    assert s[1] == Point(3.0, 4.0)


def test_segment_properties() -> None:
    """Test `a` and `b` property accessors and assignment."""
    s = Segment()
    assert s.a == Point(0.0, 0.0)
    assert s.b == Point(0.0, 0.0)

    s.a = Point(-5.0, 1.5)
    s.b = Point(2.5, -1.0)
    assert s.a == Point(-5.0, 1.5)
    assert s.b == Point(2.5, -1.0)


def test_segment_bool() -> None:
    """Test boolean conversion of Segment."""
    empty = Segment()
    assert not bool(empty)

    nonempty = Segment((0.0, 0.0), (1.0, 1.0))
    assert bool(nonempty)


def test_segment_iteration() -> None:
    """Test iteration over segment endpoints."""
    s = Segment((0.0, 1.0), (2.0, 3.0))
    pts = list(s)
    assert len(pts) == 2
    assert pts[0] == Point(0.0, 1.0)
    assert pts[1] == Point(2.0, 3.0)


def test_segment_equality() -> None:
    """Test Segment equality and inequality."""
    s1 = Segment((0.0, 0.0), (5.0, 5.0))
    s2 = Segment((0.0, 0.0), (5.0, 5.0))
    s3 = Segment((0.0, 0.0), (1.0, 1.0))

    assert s1 == s2
    assert s1 != s3


def test_segment_repr_and_str() -> None:
    """Test string representations."""
    s = Segment((0.0, 0.0), (10.0, 5.0))
    r = repr(s)
    st = str(s)
    assert "Segment" in r and "a=" in r and "b=" in r
    assert "Segment[" in st and "->" in st


def test_segment_pickle() -> None:
    """Test pickle support restores endpoints."""
    original = Segment((0.0, 0.0), (10.0, 5.0))
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)

    assert restored == original
    assert restored[0] == original[0] and restored[1] == original[1]


def test_segment_index_out_of_range() -> None:
    """Indexing out of range should raise."""
    s = Segment()
    with pytest.raises(IndexError):
        _ = s[2]
    with pytest.raises(IndexError):
        s[2] = Point(0.0, 0.0)


def test_segment_to_arrays() -> None:
    """Test Segment to_arrays method."""
    # Test with populated segment
    s = Segment((0.0, 0.0), (10.0, 5.0))

    lon_result, lat_result = s.to_arrays()

    # Verify returned arrays contain both endpoints
    expected_lon = np.array([0.0, 10.0])
    expected_lat = np.array([0.0, 5.0])
    np.testing.assert_array_equal(lon_result, expected_lon)
    np.testing.assert_array_equal(lat_result, expected_lat)

    # Verify returned arrays are numpy arrays
    assert isinstance(lon_result, np.ndarray)
    assert isinstance(lat_result, np.ndarray)

    # Verify array shapes (should always be 2 elements)
    assert lon_result.shape == (2,)
    assert lat_result.shape == (2,)

    # Test with different coordinates
    s2 = Segment((-5.0, 10.0), (15.0, -20.0))
    lon2, lat2 = s2.to_arrays()
    np.testing.assert_array_equal(lon2, np.array([-5.0, 15.0]))
    np.testing.assert_array_equal(lat2, np.array([10.0, -20.0]))

    # Test with default constructed segment (both points at origin)
    s_empty = Segment()
    lon_empty, lat_empty = s_empty.to_arrays()
    np.testing.assert_array_equal(lon_empty, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(lat_empty, np.array([0.0, 0.0]))

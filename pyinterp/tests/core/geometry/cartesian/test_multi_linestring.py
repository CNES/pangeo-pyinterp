# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test MultiLineString class."""

import pickle

import numpy as np

from .....core.geometry.cartesian import LineString, MultiLineString


def test_multilinestring_construction() -> None:
    """Test MultiLineString construction."""
    # Default construction (empty)
    mls1 = MultiLineString()
    assert len(mls1) == 0

    # Construction with linestrings
    x1 = np.array([0.0, 5.0])
    y1 = np.array([0.0, 5.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([10.0, 15.0])
    y2 = np.array([10.0, 15.0])
    ls2 = LineString(x2, y2)

    mls2 = MultiLineString([ls1, ls2])
    assert len(mls2) == 2


def test_multilinestring_indexing() -> None:
    """Test MultiLineString indexing."""
    x1 = np.array([0.0, 5.0])
    y1 = np.array([0.0, 5.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([10.0, 15.0])
    y2 = np.array([10.0, 15.0])
    ls2 = LineString(x2, y2)

    mls = MultiLineString([ls1, ls2])

    # Get items
    assert mls[0] == ls1
    assert mls[1] == ls2

    # Set items
    x3 = np.array([20.0, 25.0])
    y3 = np.array([20.0, 25.0])
    ls3 = LineString(x3, y3)
    mls[0] = ls3
    assert mls[0] == ls3


def test_multilinestring_iteration() -> None:
    """Test MultiLineString iteration."""
    x1 = np.array([0.0, 5.0])
    y1 = np.array([0.0, 5.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([10.0, 15.0])
    y2 = np.array([10.0, 15.0])
    ls2 = LineString(x2, y2)

    mls = MultiLineString([ls1, ls2])

    result = list(mls)
    assert len(result) == 2
    assert result[0] == ls1
    assert result[1] == ls2


def test_multilinestring_length() -> None:
    """Test MultiLineString length."""
    x1 = np.array([0.0, 5.0])
    y1 = np.array([0.0, 5.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([10.0, 15.0])
    y2 = np.array([10.0, 15.0])
    ls2 = LineString(x2, y2)

    mls = MultiLineString([ls1, ls2])
    assert len(mls) == 2


def test_multilinestring_append() -> None:
    """Test MultiLineString append operation."""
    mls = MultiLineString()
    assert len(mls) == 0

    x1 = np.array([0.0, 5.0])
    y1 = np.array([0.0, 5.0])
    ls1 = LineString(x1, y1)

    mls.append(ls1)
    assert len(mls) == 1
    assert mls[0] == ls1

    x2 = np.array([10.0, 15.0])
    y2 = np.array([10.0, 15.0])
    ls2 = LineString(x2, y2)

    mls.append(ls2)
    assert len(mls) == 2
    assert mls[1] == ls2


def test_multilinestring_clear() -> None:
    """Test MultiLineString clear operation."""
    x1 = np.array([0.0, 5.0])
    y1 = np.array([0.0, 5.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([10.0, 15.0])
    y2 = np.array([10.0, 15.0])
    ls2 = LineString(x2, y2)

    mls = MultiLineString([ls1, ls2])
    assert len(mls) == 2

    mls.clear()
    assert len(mls) == 0


def test_multilinestring_bool() -> None:
    """Test MultiLineString boolean conversion."""
    # Empty multilinestring
    mls1 = MultiLineString()
    assert bool(mls1) is False

    # Non-empty multilinestring
    x = np.array([0.0, 5.0])
    y = np.array([0.0, 5.0])
    ls = LineString(x, y)
    mls2 = MultiLineString([ls])
    assert bool(mls2) is True


def test_multilinestring_equality() -> None:
    """Test MultiLineString equality comparison."""
    x1 = np.array([0.0, 5.0])
    y1 = np.array([0.0, 5.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([10.0, 15.0])
    y2 = np.array([10.0, 15.0])
    ls2 = LineString(x2, y2)

    mls1 = MultiLineString([ls1, ls2])
    mls2 = MultiLineString([ls1, ls2])
    mls3 = MultiLineString([ls1])

    assert mls1 == mls2
    assert mls1 != mls3


def test_multilinestring_repr() -> None:
    """Test MultiLineString string representation."""
    x = np.array([0.0, 5.0])
    y = np.array([0.0, 5.0])
    ls = LineString(x, y)
    mls = MultiLineString([ls])
    repr_str = repr(mls)

    # Should contain class name
    assert "MultiLineString" in repr_str


def test_multilinestring_pickle() -> None:
    """Test MultiLineString serialization with pickle."""
    x1 = np.array([0.0, 5.0])
    y1 = np.array([0.0, 5.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([10.0, 15.0])
    y2 = np.array([10.0, 15.0])
    ls2 = LineString(x2, y2)

    original = MultiLineString([ls1, ls2])

    # Pickle and unpickle
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)

    # Should be equal after round-trip
    assert restored == original
    assert len(restored) == len(original)


def test_multilinestring_lines_property() -> None:
    """Test MultiLineString lines property."""
    x1 = np.array([0.0, 5.0])
    y1 = np.array([0.0, 5.0])
    ls1 = LineString(x1, y1)

    x2 = np.array([10.0, 15.0])
    y2 = np.array([10.0, 15.0])
    ls2 = LineString(x2, y2)

    mls = MultiLineString([ls1, ls2])

    # Access lines via property
    assert len(mls.lines) == 2
    assert mls.lines[0] == ls1
    assert mls.lines[1] == ls2

# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test MultiLineString class."""

import pickle

import numpy as np

from .....core.geometry.geographic import LineString, MultiLineString


def make_line(a: float = 0.0) -> LineString:
    """Create a simple V-shaped linestring offset by a."""
    lon = np.array([0.0 + a, 5.0 + a, 10.0 + a], dtype=float)
    lat = np.array([0.0, 5.0, 0.0], dtype=float)
    return LineString(lon, lat)


def test_multilinestring_creation_and_len() -> None:
    """Test MultiLineString creation and length."""
    mls = MultiLineString()
    assert bool(mls) is False
    assert len(mls) == 0

    l1 = make_line(0.0)
    l2 = make_line(1.0)

    mls.append(l1)
    mls.append(l2)

    assert bool(mls) is True
    assert len(mls) == 2


def test_multilinestring_indexing_and_view() -> None:
    """Test MultiLineString indexing and lines view."""
    l1 = make_line(0.0)
    l2 = make_line(1.0)
    mls = MultiLineString([l1])

    # __getitem__/__setitem__
    assert isinstance(mls[0], LineString)
    mls[0] = l2
    assert isinstance(mls[0], LineString)

    # View over lines property (RW)
    view = mls.lines
    assert len(view) == len(mls)
    view.append(l1)
    assert len(view) == 2
    view[0] = l1
    assert isinstance(view[0], LineString)

    # Iterate
    items = list(view)
    assert len(items) == 2
    assert all(isinstance(x, LineString) for x in items)

    # Replace via property setter
    mls.lines = [l2]
    assert len(mls) == 1


def test_multilinestring_pickle_roundtrip() -> None:
    """Test MultiLineString pickling and unpickling."""
    l1 = make_line(0.0)
    l2 = make_line(1.0)
    mls = MultiLineString([l1, l2])

    data = pickle.dumps(mls)
    mq = pickle.loads(data)

    assert len(mq) == len(mls)
    assert bool(mq) is True


def test_multilinestring_repr_str() -> None:
    """Test MultiLineString __repr__ and __str__ methods."""
    mls = MultiLineString([make_line(0.0)])
    mls_repr = repr(mls)
    mls_str = str(mls)
    assert "MultiLineString(" in mls_repr
    assert "MultiLineString[n=" in mls_str

# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test MultiPolygon class."""

import pickle

import numpy as np

from .....core.geometry.cartesian import MultiPolygon, Polygon, Ring


def test_multipolygon_construction() -> None:
    """Test MultiPolygon construction."""
    # Default construction (empty)
    mp1 = MultiPolygon()
    assert len(mp1) == 0

    # Construction with polygons
    x1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    y1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    x2 = np.array([10.0, 15.0, 15.0, 10.0, 10.0])
    y2 = np.array([10.0, 10.0, 15.0, 15.0, 10.0])
    poly2 = Polygon(Ring(x2, y2))

    mp2 = MultiPolygon([poly1, poly2])
    assert len(mp2) == 2


def test_multipolygon_indexing() -> None:
    """Test MultiPolygon indexing."""
    x1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    y1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    x2 = np.array([10.0, 15.0, 15.0, 10.0, 10.0])
    y2 = np.array([10.0, 10.0, 15.0, 15.0, 10.0])
    poly2 = Polygon(Ring(x2, y2))

    mp = MultiPolygon([poly1, poly2])

    # Get items
    assert mp[0] == poly1
    assert mp[1] == poly2

    # Set items
    x3 = np.array([20.0, 25.0, 25.0, 20.0, 20.0])
    y3 = np.array([20.0, 20.0, 25.0, 25.0, 20.0])
    poly3 = Polygon(Ring(x3, y3))
    mp[0] = poly3
    assert mp[0] == poly3


def test_multipolygon_iteration() -> None:
    """Test MultiPolygon iteration."""
    x1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    y1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    x2 = np.array([10.0, 15.0, 15.0, 10.0, 10.0])
    y2 = np.array([10.0, 10.0, 15.0, 15.0, 10.0])
    poly2 = Polygon(Ring(x2, y2))

    mp = MultiPolygon([poly1, poly2])

    result = list(mp)
    assert len(result) == 2
    assert result[0] == poly1
    assert result[1] == poly2


def test_multipolygon_length() -> None:
    """Test MultiPolygon length."""
    x1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    y1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    x2 = np.array([10.0, 15.0, 15.0, 10.0, 10.0])
    y2 = np.array([10.0, 10.0, 15.0, 15.0, 10.0])
    poly2 = Polygon(Ring(x2, y2))

    mp = MultiPolygon([poly1, poly2])
    assert len(mp) == 2


def test_multipolygon_append() -> None:
    """Test MultiPolygon append operation."""
    mp = MultiPolygon()
    assert len(mp) == 0

    x1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    y1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    mp.append(poly1)
    assert len(mp) == 1
    assert mp[0] == poly1

    x2 = np.array([10.0, 15.0, 15.0, 10.0, 10.0])
    y2 = np.array([10.0, 10.0, 15.0, 15.0, 10.0])
    poly2 = Polygon(Ring(x2, y2))

    mp.append(poly2)
    assert len(mp) == 2
    assert mp[1] == poly2


def test_multipolygon_clear() -> None:
    """Test MultiPolygon clear operation."""
    x1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    y1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    x2 = np.array([10.0, 15.0, 15.0, 10.0, 10.0])
    y2 = np.array([10.0, 10.0, 15.0, 15.0, 10.0])
    poly2 = Polygon(Ring(x2, y2))

    mp = MultiPolygon([poly1, poly2])
    assert len(mp) == 2

    mp.clear()
    assert len(mp) == 0


def test_multipolygon_bool() -> None:
    """Test MultiPolygon boolean conversion."""
    # Empty multipolygon
    mp1 = MultiPolygon()
    assert bool(mp1) is False

    # Non-empty multipolygon
    x = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly = Polygon(Ring(x, y))
    mp2 = MultiPolygon([poly])
    assert bool(mp2) is True


def test_multipolygon_equality() -> None:
    """Test MultiPolygon equality comparison."""
    x1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    y1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    x2 = np.array([10.0, 15.0, 15.0, 10.0, 10.0])
    y2 = np.array([10.0, 10.0, 15.0, 15.0, 10.0])
    poly2 = Polygon(Ring(x2, y2))

    mp1 = MultiPolygon([poly1, poly2])
    mp2 = MultiPolygon([poly1, poly2])
    mp3 = MultiPolygon([poly1])

    assert mp1 == mp2
    assert mp1 != mp3


def test_multipolygon_repr() -> None:
    """Test MultiPolygon string representation."""
    x = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly = Polygon(Ring(x, y))
    mp = MultiPolygon([poly])
    repr_str = repr(mp)

    # Should contain class name
    assert "MultiPolygon" in repr_str


def test_multipolygon_pickle() -> None:
    """Test MultiPolygon serialization with pickle."""
    x1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    y1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    x2 = np.array([10.0, 15.0, 15.0, 10.0, 10.0])
    y2 = np.array([10.0, 10.0, 15.0, 15.0, 10.0])
    poly2 = Polygon(Ring(x2, y2))

    original = MultiPolygon([poly1, poly2])

    # Pickle and unpickle
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)

    # Should be equal after round-trip
    assert restored == original
    assert len(restored) == len(original)


def test_multipolygon_polygons_property() -> None:
    """Test MultiPolygon polygons property."""
    x1 = np.array([0.0, 5.0, 5.0, 0.0, 0.0])
    y1 = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    poly1 = Polygon(Ring(x1, y1))

    x2 = np.array([10.0, 15.0, 15.0, 10.0, 10.0])
    y2 = np.array([10.0, 10.0, 15.0, 15.0, 10.0])
    poly2 = Polygon(Ring(x2, y2))

    mp = MultiPolygon([poly1, poly2])

    # Access polygons via property
    assert len(mp.polygons) == 2
    assert mp.polygons[0] == poly1
    assert mp.polygons[1] == poly2

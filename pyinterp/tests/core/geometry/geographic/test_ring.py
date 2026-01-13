# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test Ring class."""

import pickle

import numpy as np

from .....core.geometry.geographic import Point, Ring


def test_ring_construction() -> None:
    """Test Ring construction."""
    # Default construction
    r1 = Ring()
    assert len(r1) == 0

    # Construction with numpy arrays
    lon = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    lat = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    r2 = Ring(lon, lat)
    assert len(r2) == 5


def test_ring_construction_validation() -> None:
    """Test Ring construction validates array sizes."""
    # Arrays must have the same size
    with np.testing.assert_raises(ValueError):
        Ring(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0]))


def test_ring_indexing() -> None:
    """Test Ring indexing operations."""
    lon = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    lat = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    ring = Ring(lon, lat)

    # Get item
    p0 = ring[0]
    assert p0.lon == 0.0
    assert p0.lat == 0.0

    p2 = ring[2]
    assert p2.lon == 10.0
    assert p2.lat == 10.0

    # Set item
    ring[1] = Point(5.0, 5.0)
    assert ring[1].lon == 5.0
    assert ring[1].lat == 5.0


def test_ring_append() -> None:
    """Test Ring append operation."""
    ring = Ring()
    assert len(ring) == 0

    ring.append(Point(0.0, 0.0))
    assert len(ring) == 1

    ring.append(Point(10.0, 10.0))
    assert len(ring) == 2

    assert ring[0].lon == 0.0
    assert ring[1].lon == 10.0


def test_ring_clear() -> None:
    """Test Ring clear operation."""
    lon = np.array([0.0, 10.0, 10.0, 0.0])
    lat = np.array([0.0, 0.0, 10.0, 10.0])
    ring = Ring(lon, lat)

    assert len(ring) == 4
    ring.clear()
    assert len(ring) == 0


def test_ring_bool() -> None:
    """Test Ring boolean conversion."""
    empty_ring = Ring()
    assert not bool(empty_ring)

    lon = np.array([0.0, 10.0])
    lat = np.array([0.0, 10.0])
    nonempty_ring = Ring(lon, lat)
    assert bool(nonempty_ring)


def test_ring_iteration() -> None:
    """Test Ring iteration."""
    lon = np.array([0.0, 10.0, 20.0])
    lat = np.array([5.0, 15.0, 25.0])
    ring = Ring(lon, lat)

    points = list(ring)
    assert len(points) == 3
    assert points[0] == Point(0.0, 5.0)
    assert points[1] == Point(10.0, 15.0)
    assert points[2] == Point(20.0, 25.0)


def test_ring_equality() -> None:
    """Test Ring equality comparison."""
    lon1 = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    lat1 = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    r1 = Ring(lon1, lat1)

    lon2 = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    lat2 = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    r2 = Ring(lon2, lat2)

    lon3 = np.array([0.0, 10.0, 10.0])
    lat3 = np.array([0.0, 0.0, 10.0])
    r3 = Ring(lon3, lat3)

    assert r1 == r2
    assert r1 != r3


def test_ring_repr() -> None:
    """Test Ring string representation."""
    lon = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    lat = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    ring = Ring(lon, lat)

    assert repr(ring) == "Ring(5 points)"
    assert "Ring[" in str(ring)


def test_ring_pickle() -> None:
    """Test Ring pickle support."""
    lon = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
    lat = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    original = Ring(lon, lat)

    # Pickle and unpickle
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)

    assert len(restored) == len(original)
    for i in range(len(original)):
        assert restored[i] == original[i]
    assert restored == original


def test_ring_getstate_setstate() -> None:
    """Test Ring __getstate__ and __setstate__ directly."""
    lon = np.array([0.0, 10.0, 10.0])
    lat = np.array([0.0, 0.0, 10.0])
    original = Ring(lon, lat)

    new_ring = pickle.loads(pickle.dumps(original))
    assert new_ring == original

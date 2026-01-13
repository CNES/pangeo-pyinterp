# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test Point class."""

import pickle

from .....core.geometry.geographic import Point


def test_point_construction() -> None:
    """Test Point construction."""
    # Default construction
    p1 = Point()
    assert p1.lon == 0.0
    assert p1.lat == 0.0

    # Construction with coordinates
    p2 = Point(2.3, 48.9)
    assert p2.lon == 2.3
    assert p2.lat == 48.9

    # Construction with keyword arguments
    p3 = Point(lon=-122.4, lat=37.8)
    assert p3.lon == -122.4
    assert p3.lat == 37.8


def test_point_properties() -> None:
    """Test Point property access and mutation."""
    point = Point(10.0, 20.0)

    # Read properties
    assert point.lon == 10.0
    assert point.lat == 20.0

    # Write properties
    point.lon = 15.0
    point.lat = 25.0
    assert point.lon == 15.0
    assert point.lat == 25.0


def test_point_equality() -> None:
    """Test Point equality comparison."""
    p1 = Point(2.3, 48.9)
    p2 = Point(2.3, 48.9)
    p3 = Point(2.4, 48.9)

    assert p1 == p2
    assert p1 != p3
    assert p2 != p3


def test_point_hash() -> None:
    """Test Point hashing for use in sets and dicts."""
    p1 = Point(2.3, 48.9)
    p2 = Point(2.3, 48.9)
    p3 = Point(2.4, 48.9)

    # Equal points should have the same hash
    assert hash(p1) == hash(p2)

    # Can be used in sets
    point_set = {p1, p2, p3}
    assert len(point_set) == 2  # p1 and p2 are equal

    # Can be used as dict keys
    point_dict = {p1: "Paris", p3: "Other"}
    assert point_dict[p2] == "Paris"  # p2 equals p1


def test_point_repr() -> None:
    """Test Point string representation."""
    point = Point(2.3, 48.9)

    # __repr__ should show the constructor call
    assert repr(point) == "Point(lon=2.3, lat=48.9)"

    # __str__ should show a simple tuple
    assert str(point) == "(2.3, 48.9)"


def test_point_pickle() -> None:
    """Test Point pickle support."""
    original = Point(2.3, 48.9)

    # Pickle and unpickle
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)

    assert restored.lon == original.lon
    assert restored.lat == original.lat
    assert restored == original


def test_point_getstate_setstate() -> None:
    """Test Point __getstate__ and __setstate__ directly."""
    original = Point(-122.4, 37.8)

    new_point = pickle.loads(pickle.dumps(original))

    assert new_point.lon == -122.4
    assert new_point.lat == 37.8
    assert new_point == original


def test_point_edge_cases() -> None:
    """Test Point with edge case values."""
    # Test with extreme longitude/latitude values
    p1 = Point(-180.0, -90.0)
    assert p1.lon == -180.0
    assert p1.lat == -90.0

    p2 = Point(180.0, 90.0)
    assert p2.lon == 180.0
    assert p2.lat == 90.0

    # Test with zero
    p3 = Point(0.0, 0.0)
    assert p3.lon == 0.0
    assert p3.lat == 0.0

    # Test with very small numbers
    p4 = Point(1e-10, -1e-10)
    assert abs(p4.lon - 1e-10) < 1e-15
    assert abs(p4.lat - (-1e-10)) < 1e-15

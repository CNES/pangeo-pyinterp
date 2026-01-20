# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Test LineString class."""

import pickle

import numpy as np

from .....core.geometry.geographic import LineString, Point


def test_linestring_construction() -> None:
    """Test LineString construction."""
    line1 = LineString()
    assert len(line1) == 0

    lon = np.array([0.0, 5.0, 10.0])
    lat = np.array([0.0, 5.0, 0.0])
    line2 = LineString(lon, lat)
    assert len(line2) == 3


def test_linestring_construction_validation() -> None:
    """Test LineString construction validates array sizes."""
    with np.testing.assert_raises(ValueError):
        LineString(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0]))


def test_linestring_indexing() -> None:
    """Test LineString indexing operations."""
    lon = np.array([0.0, 5.0, 10.0])
    lat = np.array([0.0, 5.0, 0.0])
    line = LineString(lon, lat)

    p0 = line[0]
    assert p0.lon == 0.0
    assert p0.lat == 0.0

    p2 = line[2]
    assert p2.lon == 10.0
    assert p2.lat == 0.0

    line[1] = Point(2.5, 7.5)
    assert line[1].lon == 2.5
    assert line[1].lat == 7.5


def test_linestring_append() -> None:
    """Test LineString append operation."""
    line = LineString()
    assert len(line) == 0

    line.append(Point(0.0, 0.0))
    line.append(Point(10.0, 10.0))

    assert len(line) == 2
    assert line[0] == Point(0.0, 0.0)
    assert line[1] == Point(10.0, 10.0)


def test_linestring_clear() -> None:
    """Test LineString clear operation."""
    lon = np.array([0.0, 10.0, 20.0])
    lat = np.array([0.0, 5.0, 0.0])
    line = LineString(lon, lat)

    assert len(line) == 3
    line.clear()
    assert len(line) == 0


def test_linestring_bool() -> None:
    """Test LineString boolean conversion."""
    empty_line = LineString()
    assert not bool(empty_line)

    lon = np.array([0.0, 10.0])
    lat = np.array([0.0, 10.0])
    nonempty_line = LineString(lon, lat)
    assert bool(nonempty_line)


def test_linestring_iteration() -> None:
    """Test LineString iteration."""
    lon = np.array([0.0, 5.0, 10.0])
    lat = np.array([1.0, 2.0, 3.0])
    line = LineString(lon, lat)

    points = list(line)
    assert len(points) == 3
    assert points[0] == Point(0.0, 1.0)
    assert points[1] == Point(5.0, 2.0)
    assert points[2] == Point(10.0, 3.0)


def test_linestring_equality() -> None:
    """Test LineString equality comparison."""
    lon1 = np.array([0.0, 5.0, 10.0])
    lat1 = np.array([0.0, 5.0, 0.0])
    l1 = LineString(lon1, lat1)

    lon2 = np.array([0.0, 5.0, 10.0])
    lat2 = np.array([0.0, 5.0, 0.0])
    l2 = LineString(lon2, lat2)

    lon3 = np.array([0.0, 10.0])
    lat3 = np.array([0.0, 0.0])
    l3 = LineString(lon3, lat3)

    assert l1 == l2
    assert l1 != l3


def test_linestring_repr() -> None:
    """Test LineString string representation."""
    lon = np.array([0.0, 5.0, 10.0])
    lat = np.array([0.0, 5.0, 0.0])
    line = LineString(lon, lat)

    assert repr(line) == "LineString(3 points)"
    assert "LineString[" in str(line)


def test_linestring_pickle() -> None:
    """Test LineString pickle support."""
    lon = np.array([0.0, 5.0, 10.0])
    lat = np.array([0.0, 5.0, 0.0])
    original = LineString(lon, lat)

    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)

    assert len(restored) == len(original)
    for i in range(len(original)):
        assert restored[i] == original[i]
    assert restored == original


def test_linestring_getstate_setstate() -> None:
    """Test LineString __getstate__ and __setstate__ directly."""
    lon = np.array([0.0, 5.0, 10.0])
    lat = np.array([0.0, 5.0, 0.0])
    original = LineString(lon, lat)

    new_line = pickle.loads(pickle.dumps(original))
    assert new_line == original


def test_linestring_to_arrays() -> None:
    """Test LineString to_arrays method."""
    # Test with populated linestring
    lon = np.array([0.0, 5.0, 10.0, 15.0])
    lat = np.array([0.0, 5.0, 10.0, 15.0])
    line = LineString(lon, lat)

    lon_result, lat_result = line.to_arrays()

    # Verify returned arrays match input
    np.testing.assert_array_equal(lon_result, lon)
    np.testing.assert_array_equal(lat_result, lat)

    # Verify returned arrays are numpy arrays
    assert isinstance(lon_result, np.ndarray)
    assert isinstance(lat_result, np.ndarray)

    # Verify array shapes
    assert lon_result.shape == (4,)
    assert lat_result.shape == (4,)

    # Test with empty linestring
    empty_line = LineString()
    lon_empty, lat_empty = empty_line.to_arrays()
    assert len(lon_empty) == 0
    assert len(lat_empty) == 0

    # Test with single point
    line_single = LineString(np.array([1.0]), np.array([2.0]))
    lon_single, lat_single = line_single.to_arrays()
    np.testing.assert_array_equal(lon_single, np.array([1.0]))
    np.testing.assert_array_equal(lat_single, np.array([2.0]))

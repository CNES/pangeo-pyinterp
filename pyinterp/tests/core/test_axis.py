# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for Axis."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from ... import core


MERCATOR_LATITUDES = np.array(
    [
        -89.000000,
        -88.908818,
        -88.809323,
        -88.700757,
        -88.582294,
        -88.453032,
        -88.311987,
        -88.158087,
        -87.990161,
        -87.806932,
        -87.607008,
        -87.388869,
        -87.150861,
        -86.891178,
        -86.607851,
        -86.298736,
        -85.961495,
        -85.593582,
        -85.192224,
        -84.754402,
        -84.276831,
        -83.755939,
        -83.187844,
        -82.568330,
        -81.892820,
        -81.156357,
        -80.353575,
        -79.478674,
        -78.525397,
        -77.487013,
        -76.356296,
        -75.125518,
        -73.786444,
        -72.330344,
        -70.748017,
        -69.029837,
        -67.165823,
        -65.145744,
        -62.959262,
        -60.596124,
        -58.046413,
        -55.300856,
        -52.351206,
        -49.190700,
        -45.814573,
        -42.220632,
        -38.409866,
        -34.387043,
        -30.161252,
        -25.746331,
        -21.161107,
        -16.429384,
        -11.579629,
        -6.644331,
        -1.659041,
        3.338836,
        8.311423,
        13.221792,
        18.035297,
        22.720709,
        27.251074,
        31.604243,
        35.763079,
        39.715378,
        43.453560,
        46.974192,
        50.277423,
        53.366377,
        56.246554,
        58.925270,
        61.411164,
        63.713764,
        65.843134,
        67.809578,
        69.623418,
        71.294813,
        72.833637,
        74.249378,
        75.551083,
        76.747318,
        77.846146,
        78.855128,
        79.781321,
        80.631294,
        81.411149,
        82.126535,
        82.782681,
        83.384411,
        83.936179,
        84.442084,
        84.905904,
        85.331111,
        85.720897,
        86.078198,
        86.405707,
        86.705898,
        86.981044,
        87.233227,
        87.464359,
        87.676195,
        87.870342,
        88.048275,
        88.211348,
        88.360799,
        88.497766,
        88.623291,
        88.738328,
        88.843755,
        88.940374,
    ],
    dtype=np.float64,
)


def test_axis_regular_ascending() -> None:
    """Test regular ascending Axis accessors and methods."""
    lon = np.linspace(0, 359, 360, dtype=np.float64)
    a = core.Axis(lon, epsilon=1e-6, period=360.0)

    # Test equality
    b = core.Axis(lon, epsilon=1e-5, period=360.0)
    assert a == b
    assert not (a != b)

    # Test string representation
    assert isinstance(str(a), str)
    a_repr = str(a)
    assert "Axis(regular, period=360)" in a_repr
    assert "step: 1" in a_repr
    assert "size: 360" in a_repr

    # Test basic properties
    assert a.is_regular()
    assert a.is_ascending()
    assert a.min_value() == 0.0
    assert a.max_value() == 359.0
    assert a.front() == 0.0
    assert a.back() == 359.0
    assert len(a) == 360
    assert abs(a.increment() - 1.0) < 1e-10

    # Test indexing
    assert a[0] == 0.0
    assert a[180] == 180.0
    assert a[359] == 359.0

    # Test slicing
    values = a[0:10]
    assert len(values) == 10
    assert np.allclose(values, np.arange(0, 10, dtype=np.float64))

    # Test find_index
    test_coords = np.arange(0, 359, 1, dtype=np.float64) + 0.01
    index = a.find_index(test_coords, bounded=True)
    assert np.all(index == np.arange(0, 359, 1))

    # Test find_indexes (surrounding indexes)
    test_coords = np.arange(0, 359, 1, dtype=np.float64) + 0.5
    indexes = a.find_indexes(test_coords)
    assert indexes.shape == (359, 2)
    assert np.all(indexes[:, 0] == np.arange(0, 359, 1))
    assert np.all(indexes[:, 1] == np.arange(1, 360, 1))


def test_axis_regular_descending() -> None:
    """Test regular descending Axis after flipping."""
    lon = np.linspace(0, 359, 360, dtype=np.float64)
    a = core.Axis(lon, epsilon=1e-6, period=360.0)

    # Flip the axis
    a.flip()

    assert a.is_regular()
    assert a.is_ascending()
    assert a.increment() == 1.0
    assert a.front() == 0
    assert a.back() == 359.0
    assert a.min_value() == 0.0
    assert a.max_value() == 359.0
    assert a[0] == a.front()
    assert a[359] == a.back()

    # Test find_index on flipped axis
    test_coords = np.arange(359, -1, -1, dtype=np.float64) + 0.01
    indexes = a.find_index(test_coords[:360], bounded=True)
    expected_indexes = np.arange(359, -1, -1)
    assert np.all(indexes[1:] == expected_indexes[1:])
    assert indexes[0] == 359  # Special case for first element


def test_axis_irregular() -> None:
    """Test irregular Axis."""
    values = np.array([0.0, 1.0, 4.0, 8.0, 20.0, 50.0], dtype=np.float64)
    a = core.Axis(values, epsilon=1e-6)
    a_repr = str(a)
    assert "Axis(irregular)" in a_repr
    assert "values: [0, 1, 4, 8, 20, 50]" in a_repr
    assert "size: 6" in a_repr

    assert not a.is_regular()
    assert a.is_ascending()
    assert len(a) == 6
    assert a.front() == 0.0
    assert a.back() == 50.0
    assert a.min_value() == 0.0
    assert a.max_value() == 50.0

    # Test indexing
    assert a[0] == 0.0
    assert a[3] == 8.0
    assert a[5] == 50.0

    # Test that increment raises for irregular axis
    with pytest.raises(RuntimeError):
        a.increment()

    # Test find_index
    assert a.find_index(np.array([0.4]), bounded=True)[0] == 0
    assert a.find_index(np.array([2.4]), bounded=True)[0] == 1
    assert a.find_index(np.array([10.0]), bounded=True)[0] == 3

    # Test find_indexes
    indexes = a.find_indexes(np.array([2.5, 10.0]))
    assert indexes[0, 0] == 1
    assert indexes[0, 1] == 2
    assert indexes[1, 0] == 3
    assert indexes[1, 1] == 4


def test_axis_periodic() -> None:
    """Test periodic Axis behavior."""
    # Longitude axis 0-360
    lon = np.arange(0, 360, 5, dtype=np.float64)
    a = core.Axis(lon, epsilon=1e-6, period=360.0)

    assert a.is_periodic
    assert a.period == 360.0

    # Test wrap-around behavior
    test_coords = np.array(
        [356.0, 358.0, -2.0, -4.0, 362.0, 365.0], dtype=np.float64
    )
    indexes = a.find_index(test_coords, bounded=True)

    # Values near 360 should wrap to 0, values near 0 should wrap to 360
    assert indexes[0] == 71  # 356 is near last element
    assert indexes[1] in [0, 71]  # 358 could be near 0 or last
    assert indexes[2] in [0, 71]  # -2 wraps to 358
    assert indexes[3] in [0, 71]  # -4 wraps to 356

    # Test -180 to 180 longitude
    lon = np.arange(-180, 180, 5, dtype=np.float64)
    a = core.Axis(lon, epsilon=1e-6, period=360.0)

    assert a.is_periodic

    test_coords = np.array(
        [176.0, 178.0, -182.0, -184.0, 182.0], dtype=np.float64
    )
    indexes = a.find_index(test_coords, bounded=True)

    assert indexes[0] == 71  # 176 is near last element
    assert indexes[1] in [0, 71]  # 178 wraps
    assert indexes[2] in [0, 71]  # -182 wraps to 178
    assert indexes[3] in [0, 71]  # -184 wraps to 176


def test_axis_mercator_latitudes() -> None:
    """Test Axis with Mercator projection latitudes.

    Ensure latitudes are handled correctly.
    """
    a = core.Axis(MERCATOR_LATITUDES, epsilon=1e-6)

    assert not a.is_regular()
    assert a.is_ascending()
    assert not a.is_periodic
    assert a.period is None
    assert len(a) == len(MERCATOR_LATITUDES)
    assert a.front() == MERCATOR_LATITUDES[0]
    assert a.back() == MERCATOR_LATITUDES[-1]

    a_repr = str(a)
    assert "Axis(irregular)" in a_repr
    assert (
        "values: [-89, -88.908818, -88.809323, ..., "
        "88.738328, 88.843755, 88.940374]" in a_repr
    )
    assert "size: 109" in a_repr

    # Test find_index
    mid_idx = len(MERCATOR_LATITUDES) // 2
    mid_val = MERCATOR_LATITUDES[mid_idx]
    idx = a.find_index(np.array([mid_val]), bounded=True)[0]
    assert idx == mid_idx


def test_axis_pickle() -> None:
    """Test pickling and unpickling of Axis."""
    # Regular axis
    lon = np.linspace(0, 359, 360, dtype=np.float64)
    a = core.Axis(lon, epsilon=1e-6)
    b = pickle.loads(pickle.dumps(a))
    assert a == b
    assert a.is_regular() == b.is_regular()
    assert a.is_periodic == b.is_periodic

    # Irregular axis
    a = core.Axis(MERCATOR_LATITUDES, epsilon=1e-6)
    b = pickle.loads(pickle.dumps(a))
    assert a == b
    assert not b.is_regular()

    # Periodic axis
    lon = np.arange(0, 360, 5, dtype=np.float64)
    a = core.Axis(lon, epsilon=1e-6, period=360.0)
    b = pickle.loads(pickle.dumps(a))
    assert a == b
    assert b.is_periodic


def test_axis_equality() -> None:
    """Test Axis equality comparisons."""
    lon1 = np.linspace(0, 359, 360, dtype=np.float64)
    lon2 = np.linspace(0, 359, 360, dtype=np.float64)
    lon3 = np.linspace(0, 180, 360, dtype=np.float64)

    a = core.Axis(lon1, epsilon=1e-6)
    b = core.Axis(lon2, epsilon=1e-6)
    c = core.Axis(lon3, epsilon=1e-6)
    d = core.Axis(lon1, epsilon=1e-6, period=360.0)

    assert a == b
    assert not (a != b)
    assert a != c
    assert not (a == c)
    assert a != d  # Different periodicity
    assert not (a == d)


def test_axis_validation() -> None:
    """Test Axis validation and error handling."""
    # Empty array
    with pytest.raises(ValueError):
        core.Axis(np.array([]), epsilon=1e-6)

    # Non-monotonic values
    with pytest.raises(ValueError):
        core.Axis(np.array([5.0, 2.0, 7.0]), epsilon=1e-6)

    # Single value (should work but might have limited functionality)
    a = core.Axis(np.array([42.0]), epsilon=1e-6)
    assert len(a) == 1
    assert a[0] == 42.0


def test_axis_out_of_bounds() -> None:
    """Test Axis behavior with out-of-bounds queries."""
    lon = np.linspace(0, 359, 360, dtype=np.float64)
    a = core.Axis(lon, epsilon=1e-6)

    # Test find_index with bounded=False (should return -1)
    test_coords = np.array([-10.0, 370.0], dtype=np.float64)
    index = a.find_index(test_coords, bounded=False)
    assert index[0] == -1
    assert index[1] == -1

    # Test find_index with bounded=True (should clamp)
    index = a.find_index(test_coords, bounded=True)
    assert index[0] == 0  # Clamped to first
    assert index[1] == 359  # Clamped to last

    # Test find_indexes (should return -1 for out of bounds)
    indexes = a.find_indexes(test_coords)
    assert indexes[0, 0] == -1
    assert indexes[0, 1] == -1
    assert indexes[1, 0] == -1
    assert indexes[1, 1] == -1


def test_axis_flip_inplace() -> None:
    """Test in-place flipping of Axis."""
    lon = np.linspace(0, 359, 360, dtype=np.float64)
    a = core.Axis(lon, epsilon=1e-6)
    original_front = a.front()
    original_back = a.back()

    # Flip once
    a.flip()
    assert a.front() == original_front
    assert a.back() == original_back
    assert a.is_ascending()

    # Flip back
    a.flip()
    assert a.front() == original_front
    assert a.back() == original_back
    assert a.is_ascending()


def test_axis_slice_operations() -> None:
    """Test various slice operations on Axis."""
    lon = np.linspace(0, 100, 101, dtype=np.float64)
    a = core.Axis(lon, epsilon=1e-6)

    # Full slice
    values = a[:]
    assert len(values) == 101
    assert np.allclose(values, lon)

    # Partial slice
    values = a[10:20]
    assert len(values) == 10
    assert np.allclose(values, lon[10:20])

    # Slice with step
    values = a[::10]
    assert len(values) == 11
    assert np.allclose(values, lon[::10])

    # Negative indices
    with pytest.raises(TypeError):
        _ = a[-1]


def test_axis_edge_cases() -> None:
    """Test edge cases and boundary conditions."""
    # Two-element axis
    a = core.Axis(np.array([0.0, 1.0]), epsilon=1e-6)
    assert len(a) == 2
    assert a.is_regular()
    assert abs(a.increment() - 1.0) < 1e-10

    # Very small values
    a = core.Axis(np.array([1e-10, 2e-10, 3e-10]), epsilon=1e-12)
    assert len(a) == 3
    assert a.is_ascending()

    # Large values
    a = core.Axis(np.array([1e10, 2e10, 3e10]), epsilon=1e6)
    assert len(a) == 3
    assert a.is_ascending()


def test_axis_periodic_edge_cases() -> None:
    """Test periodic Axis with edge cases."""
    lon = np.arange(0, 355, 5, dtype=np.float64)
    a = core.Axis(lon, epsilon=1e-6, period=360.0)

    lon = np.arange(0, 360, 1, dtype=np.float64)
    a = core.Axis(lon, epsilon=1e-6, period=360.0)
    assert a.is_periodic

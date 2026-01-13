# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for TemporalAxis."""

from __future__ import annotations

import datetime
import pickle

import numpy as np
import pytest

from ... import core


def test_temporal_axis_datetime64_regular() -> None:
    """Test TemporalAxis with regular datetime64 values."""
    # Create hourly axis over one day
    start = datetime.datetime(2000, 1, 1)
    values = np.array(
        [start + datetime.timedelta(hours=index) for index in range(24)],
        dtype="datetime64[h]",
    )

    axis = core.TemporalAxis(values)

    # Test string representation
    assert isinstance(str(axis), str)

    # Test dtype
    assert axis.dtype == np.dtype("datetime64[h]")

    # Test basic properties
    assert axis.is_regular()
    assert axis.is_ascending()
    assert len(axis) == 24

    # Test front, back, min, max
    assert axis.front() == np.datetime64("2000-01-01T00:00")
    assert axis.back() == np.datetime64("2000-01-01T23:00")
    assert axis.min_value() == np.datetime64("2000-01-01T00:00")
    assert axis.max_value() == np.datetime64("2000-01-01T23:00")

    # Test increment
    increment = axis.increment()
    assert increment == np.timedelta64(1, "h")

    # Test indexing
    assert axis[0] == np.datetime64("2000-01-01T00:00")
    assert axis[12] == np.datetime64("2000-01-01T12:00")
    assert axis[23] == np.datetime64("2000-01-01T23:00")

    # Test slicing
    slice_values = axis[5:10]
    assert len(slice_values) == 5
    assert np.array_equal(slice_values, values[5:10])

    # Test find_index
    test_coords = np.array(
        ["2000-01-01T06:00", "2000-01-01T18:00"], dtype="datetime64[h]"
    )
    indexes = axis.find_index(test_coords, bounded=True)
    assert np.array_equal(indexes, [6, 18])

    # Test find_index with out of bounds
    test_coords_oob = np.array(
        ["1999-12-31", "2000-01-02"], dtype="datetime64[h]"
    )
    indexes_oob = axis.find_index(test_coords_oob, bounded=False)
    assert indexes_oob[0] == -1
    assert indexes_oob[1] == -1

    # Test find_indexes
    test_coords_mid = np.array(["2000-01-01T06:30"], dtype="datetime64[h]")
    indexes_surround = axis.find_indexes(test_coords_mid)
    assert indexes_surround.shape == (1, 2)
    # Since we're using hourly resolution, 6:30 rounds to 6:00
    assert indexes_surround[0, 0] == 6
    assert indexes_surround[0, 1] == 7


def test_temporal_axis_datetime64_microseconds() -> None:
    """Test TemporalAxis with microsecond resolution datetime64."""
    # Create axis with 1-second intervals at microsecond resolution
    start = datetime.datetime(2000, 1, 1)
    values = np.array(
        [start + datetime.timedelta(seconds=index) for index in range(86400)],
        dtype="datetime64[us]",
    )

    axis = core.TemporalAxis(values)

    assert axis.dtype == np.dtype("datetime64[us]")
    assert axis.is_regular()
    assert len(axis) == 86400

    # Test increment (1 second = 1,000,000 microseconds)
    increment = axis.increment()
    assert increment == np.timedelta64(1000000, "us")

    # Test boundaries
    assert axis.front() == np.datetime64("2000-01-01T00:00:00.000000")
    assert axis.back() == np.datetime64("2000-01-01T23:59:59.000000")
    assert axis.min_value() == np.datetime64("2000-01-01T00:00:00.000000")
    assert axis.max_value() == np.datetime64("2000-01-01T23:59:59.000000")

    # Test find_index with different dates
    test_coords = np.array(["2000-01-01", "2000-02-01"], dtype="datetime64")
    indexes = axis.find_index(test_coords, bounded=False)
    assert indexes[0] == 0
    assert indexes[1] == -1

    # Test bounded search
    indexes_bounded = axis.find_index(test_coords, bounded=True)
    assert indexes_bounded[0] == 0
    assert indexes_bounded[1] == 86399

    # Test slicing
    assert np.array_equal(axis[10:20], values[10:20])


def test_temporal_axis_timedelta64() -> None:
    """Test TemporalAxis with timedelta64 values."""
    # Create axis with timedelta values
    values = np.array(
        [datetime.timedelta(seconds=index) for index in range(86400)],
        dtype="timedelta64[us]",
    )

    axis = core.TemporalAxis(values)

    assert axis.dtype == np.dtype("timedelta64[us]")
    assert axis.is_regular()
    assert len(axis) == 86400

    # Test increment
    assert axis.increment() == np.timedelta64(1000000, "us")

    # Test boundaries
    assert axis.front() == np.timedelta64(0, "us")
    assert axis.back() == np.timedelta64(86399, "s")
    assert axis.min_value() == np.timedelta64(0, "us")
    assert axis.max_value() == np.timedelta64(86399, "s")

    # Test indexing
    assert axis[0] == np.timedelta64(0, "us")

    # Test find_index
    test_coords = np.array([0, 86400], dtype="timedelta64[s]")
    indexes = axis.find_index(test_coords, bounded=False)
    assert indexes[0] == 0
    assert indexes[1] == -1

    # Test bounded
    indexes_bounded = axis.find_index(test_coords, bounded=True)
    assert indexes_bounded[0] == 0
    assert indexes_bounded[1] == 86399

    # Test slicing
    assert np.array_equal(axis[10:20], values[10:20])

    # Irregular timedelta axis
    irregular_values = np.array(
        [datetime.timedelta(seconds=index) for index in [0, 1, 4, 8, 20, 50]],
        dtype="timedelta64[s]",
    )
    irregular_axis = core.TemporalAxis(irregular_values)
    assert not irregular_axis.is_regular()

    # Test find_indexes
    test_coords_irregular = np.array([2, 10], dtype="timedelta64[s]")
    indexes_surround = irregular_axis.find_indexes(test_coords_irregular)
    assert indexes_surround.shape == (2, 2)
    assert indexes_surround[0, 0] == 1  # Surrounding index for 2s
    assert indexes_surround[0, 1] == 2
    assert indexes_surround[1, 0] == 3  # Surrounding index for 10s
    assert indexes_surround[1, 1] == 4


def test_temporal_axis_flip() -> None:
    """Test flipping TemporalAxis."""
    start = datetime.datetime(2000, 1, 1)
    values = np.array(
        [start + datetime.timedelta(seconds=index) for index in range(86400)],
        dtype="datetime64[us]",
    )

    axis = core.TemporalAxis(values)
    original_front = axis.front()
    original_back = axis.back()

    # Flip the axis (inplace)
    axis.flip(inplace=True)

    # After flip, front and back should be swapped
    assert axis.front() == original_back
    assert axis.back() == original_front

    # Min and max should remain the same
    assert axis.min_value() == np.datetime64("2000-01-01T00:00:00.000000")
    assert axis.max_value() == np.datetime64("2000-01-01T23:59:59.000000")

    # Increment should be negative
    assert axis.increment() == np.timedelta64(-1000000, "us")

    # Test find_index after flip
    test_coords = np.array(["2000-01-01", "2000-02-01"], dtype="datetime64")
    indexes = axis.find_index(test_coords, bounded=False)
    assert indexes[0] == 86399  # Now at the end
    assert indexes[1] == -1

    # Test find_indexes
    indexes_surround = axis.find_indexes(test_coords)
    assert indexes_surround[0, 0] == 86398
    assert indexes_surround[0, 1] == 86399
    assert indexes_surround[1, 0] == -1
    assert indexes_surround[1, 1] == -1


def test_temporal_axis_flip_copy() -> None:
    """Test flipping TemporalAxis with copy."""
    values = np.array(
        [datetime.timedelta(seconds=index) for index in range(100)],
        dtype="timedelta64[s]",
    )

    axis = core.TemporalAxis(values)
    original_front = axis.front()

    # Flip creating a copy
    flipped = axis.flip(inplace=False)

    # Original should be unchanged
    assert axis.front() == original_front
    assert axis.is_ascending()

    # Flipped should be different
    assert flipped.front() == axis.back()
    assert flipped.back() == axis.front()


def test_temporal_axis_epsilon() -> None:
    """Test TemporalAxis with epsilon parameter."""
    values = np.array(
        ["2000-01-01T00:00:00", "2000-01-01T00:00:01", "2000-01-01T00:00:02"],
        dtype="datetime64[s]",
    )

    # Epsilon finer than the time resolution cannot determine axis regularity
    epsilon = np.timedelta64(100, "ms")
    with pytest.raises(ValueError, match="is finer than points resolution"):
        core.TemporalAxis(values, epsilon=epsilon)

    # Epsilon of 1 seconds
    epsilon = np.timedelta64(1, "s")
    axis = core.TemporalAxis(values, epsilon=epsilon)

    assert axis.is_regular()
    assert len(axis) == 3


def test_temporal_axis_period() -> None:
    """Test TemporalAxis with period parameter."""
    # Daily cycle (24 hours)
    values = np.array(
        [
            "2000-01-01T00:00",
            "2000-01-01T06:00",
            "2000-01-01T12:00",
            "2000-01-01T18:00",
        ],
        dtype="datetime64[h]",
    )

    period = np.timedelta64(24, "h")
    axis = core.TemporalAxis(values, period=period)

    assert axis.is_periodic
    assert axis.period == period

    # Test without period
    axis_no_period = core.TemporalAxis(values)
    assert not axis_no_period.is_periodic
    assert axis_no_period.period is None


def test_temporal_axis_cast_to_temporal_axis() -> None:
    """Test cast_to_temporal_axis for resolution conversion."""
    start = datetime.datetime(2000, 1, 1)
    values = np.array(
        [start + datetime.timedelta(seconds=index) for index in range(86400)],
        dtype="datetime64[us]",
    )

    axis = core.TemporalAxis(values)

    # Convert from hours to microseconds - should produce different values
    values_hours = values.astype("datetime64[h]")
    converted = axis.cast_to_temporal_axis(values_hours)
    assert converted.dtype == np.dtype("datetime64[us]")
    assert not np.array_equal(values, converted)

    # Test with same resolution
    converted_same = axis.cast_to_temporal_axis(values)
    assert np.array_equal(values, converted_same)


def test_temporal_axis_pickle() -> None:
    """Test pickling and unpickling TemporalAxis."""
    start = datetime.datetime(2000, 1, 1)
    values = np.array(
        [start + datetime.timedelta(hours=index) for index in range(24)],
        dtype="datetime64[h]",
    )

    axis = core.TemporalAxis(values)

    # Pickle and unpickle
    pickled = pickle.dumps(axis)
    unpickled = pickle.loads(pickled)

    # Test equality
    assert axis == unpickled
    assert id(axis) != id(unpickled)

    # Test that values are preserved
    assert np.array_equal(axis[:], unpickled[:])
    assert axis.dtype == unpickled.dtype
    assert axis.is_regular() == unpickled.is_regular()


def test_temporal_axis_equality() -> None:
    """Test equality comparison for TemporalAxis."""
    values1 = np.array(
        ["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64[D]"
    )

    values2 = np.array(
        ["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64[D]"
    )

    values3 = np.array(
        ["2000-01-01", "2000-01-02", "2000-01-04"], dtype="datetime64[D]"
    )

    axis1 = core.TemporalAxis(values1)
    axis2 = core.TemporalAxis(values2)
    axis3 = core.TemporalAxis(values3)

    assert axis1 == axis2
    assert not (axis1 != axis2)
    assert axis1 != axis3
    assert not (axis1 == axis3)


def test_temporal_axis_validation() -> None:
    """Test TemporalAxis validation and error handling."""
    # Non-datetime64 array should raise ValueError
    with pytest.raises(ValueError):
        core.TemporalAxis(np.arange(10))  # type: ignore[arg-type]

    # Empty array should raise ValueError
    with pytest.raises(ValueError):
        core.TemporalAxis(np.array([], dtype="datetime64[s]"))

    # Non-monotonic values should raise ValueError
    with pytest.raises(ValueError):
        core.TemporalAxis(
            np.array(
                ["2000-01-03", "2000-01-01", "2000-01-02"],
                dtype="datetime64[D]",
            )
        )


def test_temporal_axis_cast_to_temporal_axis_validation() -> None:
    """Test cast_to_temporal_axis validation."""
    values = np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[s]")

    axis = core.TemporalAxis(values)

    # Non-datetime64 array should raise ValueError
    with pytest.raises(ValueError):
        axis.cast_to_temporal_axis(np.arange(2))  # type: ignore[arg-type]

    # Valid conversion from days to seconds
    values_days = np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[D]")
    converted = axis.cast_to_temporal_axis(values_days)
    assert converted.dtype == np.dtype("datetime64[s]")


def test_temporal_axis_irregular() -> None:
    """Test TemporalAxis with irregular spacing."""
    values = np.array(
        [
            "2000-01-01T00:00",
            "2000-01-01T06:09",
            "2000-01-01T12:30",
            "2000-01-01T18:45",
        ],
        dtype="datetime64[m]",
    )

    axis = core.TemporalAxis(values, epsilon=np.timedelta64(1, "m"))

    assert not axis.is_regular()
    assert axis.is_ascending()
    assert len(axis) == 4

    # Test that increment raises for irregular axis
    with pytest.raises(RuntimeError):
        axis.increment()

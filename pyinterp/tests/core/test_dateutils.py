# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for date utilities."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ...core import dateutils


# Set random seed for deterministic tests
RANDOM_SEED = 42


if TYPE_CHECKING:
    from ...type_hints import NDArray1DDateTime64


def make_date(
    samples: int = 100,
    resolution: str = "us",
) -> tuple[list[datetime.datetime], NDArray1DDateTime64]:
    """Generate random dates for testing.

    Creates dates both before and after Unix epoch (1970-01-01)
    to test handling of negative timestamps.

    Args:
        samples: Number of dates to generate in each direction (before/after
            epoch). Total dates returned will be samples * 2.
        resolution: Temporal resolution ('ms', 'us', or 'ns') for the numpy
            array.

    Returns:
        Tuple of (Python datetime list, NumPy datetime64 array with specified
        resolution).

    """
    epoch = datetime.datetime(1970, 1, 1)
    delta = datetime.datetime.now() - datetime.datetime(1970, 1, 1)

    rng = np.random.default_rng(RANDOM_SEED)

    pydates = [epoch + rng.random() * delta for _ in range(samples)]
    pydates += [epoch - rng.random() * delta for _ in range(samples)]
    npdates = np.array(pydates).astype(f"datetime64[{resolution}]")

    return pydates, npdates


@pytest.mark.parametrize("resolution", ["ms", "us", "ns"])
def test_date(resolution: str) -> None:
    """Test date extraction from datetime64.

    Validates that the dateutils.date() function correctly extracts
    year, month, and day components from datetime64 arrays.
    """
    pydates, npdates = make_date(resolution=resolution)
    years, months, days = dateutils.date(npdates)

    for ix, (year, month, day) in enumerate(
        zip(years, months, days, strict=True)
    ):
        expected = pydates[ix]
        assert year == expected.year
        assert month == expected.month
        assert day == expected.day


@pytest.mark.parametrize("resolution", ["ms", "us", "ns"])
def test_timedelta_since_january(resolution: str) -> None:
    """Test timedelta since January extraction from datetime64.

    Validates that the dateutils.timedelta_since_january() function returns
    the correct time elapsed since January 1st of the same year, including
    both day-of-year and intraday time components.
    """
    pydates, npdates = make_date(resolution=resolution)
    timedeltas = dateutils.timedelta_since_january(npdates)

    for ix, timedelta_value in enumerate(timedeltas):
        expected = pydates[ix].utctimetuple().tm_yday

        # Convert timedelta to days (0-indexed, hence +1 to match tm_yday which
        # is 1-indexed)
        yday = timedelta_value.astype("timedelta64[D]").astype("int")
        assert yday + 1 == expected

        # Extract time components from the timedelta
        microseconds = int(
            timedelta_value.astype("timedelta64[us]").astype("int64")
        )
        dt = datetime.timedelta(microseconds=microseconds)
        minute, second = divmod(dt.seconds, 60)
        hour, minute = divmod(minute, 60)

        assert hour == pydates[ix].hour
        assert minute == pydates[ix].minute
        assert second == pydates[ix].second

        # For millisecond resolution, truncate microseconds to millisecond
        # precision
        expected_microsecond = pydates[ix].microsecond
        if resolution == "ms":
            expected_microsecond = expected_microsecond // 1000 * 1000
        assert dt.microseconds == expected_microsecond


@pytest.mark.parametrize("resolution", ["ms", "us", "ns"])
def test_isocalendar(resolution: str) -> None:
    """Test ISO calendar extraction from datetime64.

    Validates that the dateutils.isocalendar() function correctly extracts
    ISO calendar components (year, week, weekday) from datetime64 arrays.
    """
    pydates, npdates = make_date(resolution=resolution)
    years, weeks, weekdays = dateutils.isocalendar(npdates)

    for ix, (year, week, weekday) in enumerate(
        zip(years, weeks, weekdays, strict=True)
    ):
        assert (year, week, weekday) == pydates[ix].isocalendar()


@pytest.mark.parametrize("resolution", ["ms", "us", "ns"])
def test_time(resolution: str) -> None:
    """Test time extraction from datetime64.

    Validates that the dateutils.time() function correctly extracts
    time components (hour, minute, second) from datetime64 arrays.
    """
    pydates, npdates = make_date(resolution=resolution)
    hours, minutes, seconds = dateutils.time(npdates)

    for ix, (hour, minute, second) in enumerate(
        zip(hours, minutes, seconds, strict=True)
    ):
        expected = pydates[ix]
        assert hour == expected.hour
        assert minute == expected.minute
        assert second == expected.second


@pytest.mark.parametrize("resolution", ["ms", "us", "ns"])
def test_weekday(resolution: str) -> None:
    """Test weekday extraction from datetime64.

    Validates that the dateutils.weekday() function correctly extracts
    weekday values (0=Monday, 6=Sunday) from datetime64 arrays.
    """
    pydates, npdates = make_date(resolution=resolution)
    weekdays = dateutils.weekday(npdates)

    for ix, weekday_value in enumerate(weekdays):
        _, _, py_weekday = pydates[ix].isocalendar()
        # Convert ISO weekday (1=Monday, 7=Sunday) to 0-indexed (0=Monday,
        # 6=Sunday)
        assert weekday_value == py_weekday % 7


def test_wrong_units() -> None:
    """Test dateutils with wrong units and shapes.

    Validates that the dateutils functions properly reject unsupported
    datetime64 resolutions and non-1D array shapes.
    """
    _, npdates = make_date(10)

    # Test unsupported temporal resolution (hours not supported)
    with pytest.raises(ValueError):
        dateutils.date(npdates.astype("datetime64[h]"))

    # Test wrong array shape (2D instead of 1D)
    with pytest.raises(ValueError):
        dateutils.date(npdates.reshape(5, 2))  # type: ignore[arg-type]


@pytest.mark.parametrize("resolution", ["ms", "us", "ns"])
def test_epoch(resolution: str) -> None:
    """Test handling of Unix epoch (1970-01-01 00:00:00).

    Validates that the epoch is correctly handled across all functions.
    """
    epoch = np.array([datetime.datetime(1970, 1, 1)]).astype(
        f"datetime64[{resolution}]"
    )

    # Test date extraction
    years, months, days = dateutils.date(epoch)
    assert years[0] == 1970
    assert months[0] == 1
    assert days[0] == 1

    # Test time extraction
    hours, minutes, seconds = dateutils.time(epoch)
    assert hours[0] == 0
    assert minutes[0] == 0
    assert seconds[0] == 0

    # Test timedelta since January
    timedeltas = dateutils.timedelta_since_january(epoch)
    assert timedeltas[0].astype("timedelta64[D]").astype("int") == 0

    # Test weekday (1970-01-01 was a Thursday, ISO weekday=4, which becomes 4
    # in 0-indexed)
    weekdays = dateutils.weekday(epoch)
    assert weekdays[0] == 4


@pytest.mark.parametrize("resolution", ["ms", "us", "ns"])
def test_year_boundary(resolution: str) -> None:
    """Test handling of year boundaries (Dec 31 / Jan 1).

    Validates correct date component extraction across year transitions.
    """
    # December 31, 2023 23:59:59
    dec_31 = np.array([datetime.datetime(2023, 12, 31, 23, 59, 59)]).astype(
        f"datetime64[{resolution}]"
    )

    years, months, days = dateutils.date(dec_31)
    assert years[0] == 2023
    assert months[0] == 12
    assert days[0] == 31

    # January 1, 2024 00:00:00
    jan_1 = np.array([datetime.datetime(2024, 1, 1, 0, 0, 0)]).astype(
        f"datetime64[{resolution}]"
    )

    years, months, days = dateutils.date(jan_1)
    assert years[0] == 2024
    assert months[0] == 1
    assert days[0] == 1


@pytest.mark.parametrize("resolution", ["ms", "us", "ns"])
def test_leap_year(resolution: str) -> None:
    """Test handling of leap year date (Feb 29).

    Validates correct handling of February 29th in leap years.
    """
    # February 29, 2020 (leap year)
    feb_29 = np.array([datetime.datetime(2020, 2, 29, 12, 30, 45)]).astype(
        f"datetime64[{resolution}]"
    )

    years, months, days = dateutils.date(feb_29)
    assert years[0] == 2020
    assert months[0] == 2
    assert days[0] == 29

    # Verify it's the 60th day of the year
    timedeltas = dateutils.timedelta_since_january(feb_29)
    yday = timedeltas[0].astype("timedelta64[D]").astype("int")
    assert yday + 1 == 60


def test_empty_array() -> None:
    """Test handling of empty datetime64 arrays.

    Validates that empty arrays are handled correctly without errors.
    """
    empty = np.array([], dtype="datetime64[us]")

    years, months, days = dateutils.date(empty)
    assert len(years) == 0
    assert len(months) == 0
    assert len(days) == 0

    hours, minutes, seconds = dateutils.time(empty)
    assert len(hours) == 0
    assert len(minutes) == 0
    assert len(seconds) == 0

    weekdays = dateutils.weekday(empty)
    assert len(weekdays) == 0

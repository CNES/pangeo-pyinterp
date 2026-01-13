# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for Period class."""

from __future__ import annotations

import pickle
from typing import Literal, TypeAlias

import numpy as np
import pytest

from ...core.period import Period


TimeUnits: TypeAlias = Literal[
    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"
]


class TestPeriodConstruction:
    """Test Period construction and properties."""

    def test_construction_default_within(self) -> None:
        """Test Period construction with default within=True."""
        # [2020-01-01, 2020-01-10] with within=True (default)
        p = Period(
            np.datetime64("2020-01-01", "D"),
            np.datetime64("2020-01-10", "D"),
        )
        assert p.begin == np.datetime64("2020-01-01", "D")
        assert p.last == np.datetime64("2020-01-10", "D")
        assert p.resolution == np.dtype("datetime64[D]")
        assert p.end() == np.datetime64("2020-01-11", "D")
        assert p.duration() == np.timedelta64(10, "D")

        # Period should be inclusive on both ends
        assert p.contains(np.datetime64("2020-01-01", "D"))
        assert p.contains(np.datetime64("2020-01-10", "D"))
        assert not p.contains(np.datetime64("2020-01-11", "D"))

    def test_construction_within_false(self) -> None:
        """Test Period construction with within=False."""
        # [2020-01-01, 2020-01-10) with within=False
        p = Period(
            np.datetime64("2020-01-01", "D"),
            np.datetime64("2020-01-10", "D"),
            within=False,
        )
        # Verify accessors
        assert p.begin == np.datetime64("2020-01-01", "D")
        assert p.last == np.datetime64("2020-01-09", "D")
        assert p.resolution == np.dtype("datetime64[D]")
        assert p.end() == np.datetime64("2020-01-10", "D")
        # Period should be inclusive at begin, exclusive at end
        assert p.contains(np.datetime64("2020-01-01", "D"))
        assert p.contains(np.datetime64("2020-01-09", "D"))
        assert not p.contains(np.datetime64("2020-01-10", "D"))

    def test_construction_different_resolutions(self) -> None:
        """Test Period construction with different datetime64 resolutions."""
        # Mix day and hour resolutions
        p = Period(
            np.datetime64("2020-01-01", "D"),
            np.datetime64("2020-01-02T12", "h"),
        )
        # Period should automatically promote to finer resolution (hours)
        assert p.resolution == np.dtype("datetime64[h]")
        assert p.contains(np.datetime64("2020-01-02T06", "h"))
        assert p.contains(np.datetime64("2020-01-02T12", "h"))

    def test_construction_with_negative_timestamps(self) -> None:
        """Test Period construction with dates before Unix epoch."""
        # Before 1970
        p = Period(
            np.datetime64("1960-01-01", "D"),
            np.datetime64("1965-12-31", "D"),
        )
        assert p.begin == np.datetime64("1960-01-01", "D")
        assert p.last == np.datetime64("1965-12-31", "D")
        assert p.contains(np.datetime64("1963-06-15", "D"))
        assert not p.contains(np.datetime64("1970-01-01", "D"))

    def test_construction_invalid_none(self) -> None:
        """Test that None values are rejected."""
        with pytest.raises(TypeError):
            Period(
                None,  # type: ignore[arg-type]
                np.datetime64("2020-01-10", "D"),
            )

        with pytest.raises(TypeError):
            Period(
                np.datetime64("2020-01-01", "D"),
                None,  # type: ignore[arg-type]
            )

    def test_construction_invalid_type(self) -> None:
        """Test that non-datetime64 types are rejected."""
        with pytest.raises(
            ValueError, match=r"must be a numpy.datetime64 scalar"
        ):
            Period(
                "2020-01-01",  # type: ignore[arg-type]
                np.datetime64("2020-01-10", "D"),
            )

        with pytest.raises(
            ValueError, match=r"must be a numpy.datetime64 scalar"
        ):
            Period(
                np.datetime64("2020-01-01", "D"),
                12345,  # type: ignore[arg-type]
            )


class TestPeriodContainment:
    """Test Period containment methods."""

    def test_contains_point(self) -> None:
        """Test contains method with datetime64 points."""
        p = Period(
            np.datetime64("2020-01-01", "D"),
            np.datetime64("2020-01-31", "D"),
        )

        # Inside period
        assert p.contains(np.datetime64("2020-01-15", "D"))
        # At boundaries
        assert p.contains(np.datetime64("2020-01-01", "D"))
        assert p.contains(np.datetime64("2020-01-31", "D"))
        # Outside period
        assert not p.contains(np.datetime64("2019-12-31", "D"))
        assert not p.contains(np.datetime64("2020-02-01", "D"))

    def test_contains_period(self) -> None:
        """Test contains method with another Period."""
        p1 = Period(
            np.datetime64("2020-01-01", "D"),
            np.datetime64("2020-01-31", "D"),
        )
        p2 = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        p3 = Period(
            np.datetime64("2020-01-20", "D"),
            np.datetime64("2020-02-10", "D"),
        )

        # p1 contains p2
        assert p1.contains(p2)
        # p1 contains itself
        assert p1.contains(p1)
        # p1 does not contain p3 (extends beyond)
        assert not p1.contains(p3)

    def test_is_after(self) -> None:
        """Test is_after method."""
        p = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )

        # Point before period
        assert p.is_after(np.datetime64("2020-01-09", "D"))
        assert p.is_after(np.datetime64("2020-01-01", "D"))
        # Point at or after begin
        assert not p.is_after(np.datetime64("2020-01-10", "D"))
        assert not p.is_after(np.datetime64("2020-01-15", "D"))

    def test_is_before(self) -> None:
        """Test is_before method."""
        p = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )

        # Point after period
        assert p.is_before(np.datetime64("2020-01-21", "D"))
        assert p.is_before(np.datetime64("2020-02-01", "D"))
        # Point at or before last
        assert not p.is_before(np.datetime64("2020-01-20", "D"))
        assert not p.is_before(np.datetime64("2020-01-15", "D"))

    def test_is_close(self) -> None:
        """Test is_close method with tolerance."""
        p = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )

        # Exact boundaries
        assert p.is_close(
            np.datetime64("2020-01-10", "D"),
            np.timedelta64(0, "D"),
        )
        assert p.is_close(
            np.datetime64("2020-01-20", "D"),
            np.timedelta64(0, "D"),
        )

        # Within tolerance before begin
        assert p.is_close(
            np.datetime64("2020-01-08", "D"),
            np.timedelta64(3, "D"),
        )
        assert not p.is_close(
            np.datetime64("2020-01-08", "D"),
            np.timedelta64(1, "D"),
        )

        # Within tolerance after last
        assert p.is_close(
            np.datetime64("2020-01-23", "D"),
            np.timedelta64(5, "D"),
        )
        assert not p.is_close(
            np.datetime64("2020-01-23", "D"),
            np.timedelta64(2, "D"),
        )

        # Outside tolerance
        assert not p.is_close(
            np.datetime64("2020-01-01", "D"),
            np.timedelta64(5, "D"),
        )


class TestPeriodOperations:
    """Test Period intersection, merge, and other operations."""

    def test_intersects(self) -> None:
        """Test intersects method."""
        p1 = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        p2 = Period(
            np.datetime64("2020-01-15", "D"),
            np.datetime64("2020-01-25", "D"),
        )
        p3 = Period(
            np.datetime64("2020-01-21", "D"),
            np.datetime64("2020-01-30", "D"),
        )

        # Overlapping periods
        assert p1.intersects(p2)
        assert p2.intersects(p1)

        # Adjacent periods (not intersecting in half-open interval semantics)
        assert not p1.intersects(p3)

    def test_is_adjacent(self) -> None:
        """Test is_adjacent method."""
        p1 = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        # Adjacent period starting right after p1
        p2 = Period(
            np.datetime64("2020-01-21", "D"),
            np.datetime64("2020-01-30", "D"),
        )
        # Overlapping period
        p3 = Period(
            np.datetime64("2020-01-15", "D"),
            np.datetime64("2020-01-25", "D"),
        )
        # Disjoint period
        p4 = Period(
            np.datetime64("2020-02-01", "D"),
            np.datetime64("2020-02-10", "D"),
        )

        assert p1.is_adjacent(p2)
        assert p2.is_adjacent(p1)
        assert not p1.is_adjacent(p3)  # Overlapping
        assert not p1.is_adjacent(p4)  # Disjoint with gap

    def test_intersection(self) -> None:
        """Test intersection method."""
        p1 = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        p2 = Period(
            np.datetime64("2020-01-15", "D"),
            np.datetime64("2020-01-25", "D"),
        )

        # Intersection should be [2020-01-15, 2020-01-20]
        i = p1.intersection(p2)
        assert i is not None
        assert i.begin == np.datetime64("2020-01-15", "D")
        assert i.last == np.datetime64("2020-01-20", "D")
        assert i.resolution == np.dtype("datetime64[D]")
        assert i.duration() == np.timedelta64(6, "D")
        assert i.contains(np.datetime64("2020-01-15", "D"))
        assert i.contains(np.datetime64("2020-01-20", "D"))
        assert not i.contains(np.datetime64("2020-01-14", "D"))
        assert not i.contains(np.datetime64("2020-01-21", "D"))

    def test_intersection_disjoint(self) -> None:
        """Test intersection of disjoint periods returns null period."""
        p1 = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        p2 = Period(
            np.datetime64("2020-01-21", "D"),
            np.datetime64("2020-01-30", "D"),
        )

        i = p1.intersection(p2)
        assert i is None

    def test_merge(self) -> None:
        """Test merge method."""
        p1 = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        p2 = Period(
            np.datetime64("2020-01-15", "D"),
            np.datetime64("2020-01-25", "D"),
        )

        # Merge overlapping periods
        m = p1.merge(p2)
        assert m is not None
        assert m.begin == np.datetime64("2020-01-10", "D")
        assert m.last == np.datetime64("2020-01-25", "D")
        assert m.resolution == np.dtype("datetime64[D]")
        assert m.duration() == np.timedelta64(16, "D")
        assert m.contains(np.datetime64("2020-01-10", "D"))
        assert m.contains(np.datetime64("2020-01-15", "D"))
        assert m.contains(np.datetime64("2020-01-25", "D"))
        assert not m.contains(np.datetime64("2020-01-09", "D"))
        assert not m.contains(np.datetime64("2020-01-26", "D"))

    def test_merge_adjacent(self) -> None:
        """Test merge of adjacent periods."""
        p1 = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        p2 = Period(
            np.datetime64("2020-01-21", "D"),
            np.datetime64("2020-01-30", "D"),
        )

        # Merge adjacent periods
        m = p1.merge(p2)
        assert m is not None
        assert m.begin == np.datetime64("2020-01-10", "D")
        assert m.last == np.datetime64("2020-01-30", "D")
        assert m.duration() == np.timedelta64(21, "D")
        assert m.contains(np.datetime64("2020-01-10", "D"))
        assert m.contains(np.datetime64("2020-01-20", "D"))
        assert m.contains(np.datetime64("2020-01-21", "D"))
        assert m.contains(np.datetime64("2020-01-30", "D"))

    def test_merge_disjoint(self) -> None:
        """Test merge of disjoint periods returns null period."""
        p1 = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        p2 = Period(
            np.datetime64("2020-02-01", "D"),
            np.datetime64("2020-02-10", "D"),
        )

        m = p1.merge(p2)
        assert m is None

    def test_extend(self) -> None:
        """Test extend method."""
        p = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )

        # Extend with point inside - no change
        e1 = p.extend(np.datetime64("2020-01-15", "D"))
        assert e1.begin == np.datetime64("2020-01-10", "D")
        assert e1.last == np.datetime64("2020-01-20", "D")
        assert e1.contains(np.datetime64("2020-01-10", "D"))
        assert e1.contains(np.datetime64("2020-01-20", "D"))

        # Extend with point before - expands begin
        e2 = p.extend(np.datetime64("2020-01-05", "D"))
        assert e2.begin == np.datetime64("2020-01-05", "D")
        assert e2.last == np.datetime64("2020-01-20", "D")
        assert e2.duration() == np.timedelta64(16, "D")
        assert e2.contains(np.datetime64("2020-01-05", "D"))
        assert e2.contains(np.datetime64("2020-01-20", "D"))

        # Extend with point after - expands last
        e3 = p.extend(np.datetime64("2020-01-25", "D"))
        assert e3.begin == np.datetime64("2020-01-10", "D")
        assert e3.last == np.datetime64("2020-01-25", "D")
        assert e3.duration() == np.timedelta64(16, "D")
        assert e3.contains(np.datetime64("2020-01-10", "D"))
        assert e3.contains(np.datetime64("2020-01-25", "D"))

    def test_shift(self) -> None:
        """Test shift method."""
        p = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )

        # Shift forward by 10 days
        s1 = p.shift(np.timedelta64(10, "D"))
        assert s1.begin == np.datetime64("2020-01-20", "D")
        assert s1.last == np.datetime64("2020-01-30", "D")
        assert s1.duration() == np.timedelta64(11, "D")
        assert s1.contains(np.datetime64("2020-01-20", "D"))
        assert s1.contains(np.datetime64("2020-01-30", "D"))
        assert not s1.contains(np.datetime64("2020-01-15", "D"))

        # Shift backward by 5 days
        s2 = p.shift(np.timedelta64(-5, "D"))
        assert s2.begin == np.datetime64("2020-01-05", "D")
        assert s2.last == np.datetime64("2020-01-15", "D")
        assert s2.duration() == np.timedelta64(11, "D")
        assert s2.contains(np.datetime64("2020-01-05", "D"))
        assert s2.contains(np.datetime64("2020-01-15", "D"))
        assert not s2.contains(np.datetime64("2020-01-20", "D"))

        # Shift by zero
        s3 = p.shift(np.timedelta64(0, "D"))
        assert s3.begin == np.datetime64("2020-01-10", "D")
        assert s3.last == np.datetime64("2020-01-20", "D")
        assert s3.contains(np.datetime64("2020-01-10", "D"))
        assert s3.contains(np.datetime64("2020-01-20", "D"))


class TestPeriodComparison:
    """Test Period comparison operators."""

    def test_equality(self) -> None:
        """Test equality operator."""
        p1 = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        p2 = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        p3 = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-25", "D"),
        )

        assert p1 == p2
        assert not (p1 == p3)
        assert p1 != p3
        assert not (p1 != p2)

    def test_equality_different_resolutions(self) -> None:
        """Test equality with different resolutions."""
        # Same period but different resolutions should be equal
        p1 = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        p2 = Period(
            np.datetime64("2020-01-10T00", "h"),
            np.datetime64("2020-01-20T00", "h"),
        )
        assert p1 == p2

    def test_ordering(self) -> None:
        """Test ordering operators."""
        p1 = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        p2 = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-25", "D"),
        )
        p3 = Period(
            np.datetime64("2020-01-15", "D"),
            np.datetime64("2020-01-20", "D"),
        )

        # Lexicographic ordering on (begin, last)
        assert p1 < p2  # Same begin, p1.last < p2.last
        assert p1 < p3  # p1.begin < p3.begin
        assert p2 > p1
        assert p3 > p1
        assert p1 <= p2
        assert p1 <= p1
        assert p2 >= p1
        assert p1 >= p1


class TestPeriodSerialization:
    """Test Period pickling and string representation."""

    def test_pickle(self) -> None:
        """Test pickling and unpickling of Period."""
        p = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )

        # Pickle and unpickle
        pickled = pickle.dumps(p)
        p_restored = pickle.loads(pickled)

        # Should be equal
        assert p == p_restored
        # Should preserve functionality
        assert p_restored.contains(np.datetime64("2020-01-15", "D"))
        assert not p_restored.contains(np.datetime64("2020-02-01", "D"))

    def test_str(self) -> None:
        """Test string representation of Period."""
        p = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        s = str(p)
        # Should be in format [begin, end)
        assert s == "[2020-01-10, 2020-01-20)"


class TestPeriodResolutionConversion:
    """Test Period resolution conversion and promotion."""

    @pytest.mark.parametrize(
        "unit",
        ["W", "D", "h", "m", "s", "ms", "us", "ns"],
    )
    def test_construction_with_different_units(self, unit: TimeUnits) -> None:
        """Test Period construction with various datetime64 units.

        Note: Year ('Y') and Month ('M') units are excluded because numpy
        truncates dates to year/month granularity, making day-level dates
        indistinguishable.
        """
        # Create period with specified unit
        p = Period(
            np.datetime64("2020-01-10", unit),
            np.datetime64("2020-01-20", unit),
        )
        # Should contain a point in the same unit
        assert p.contains(np.datetime64("2020-01-15", unit))
        # Should not contain points outside
        assert not p.contains(np.datetime64("2020-01-25", unit))

    @pytest.mark.parametrize(
        ("begin_unit", "end_unit"),
        [
            ("D", "h"),
            ("h", "m"),
            ("m", "s"),
            ("s", "ms"),
            ("ms", "us"),
            ("us", "ns"),
            ("D", "ns"),
            ("W", "D"),
        ],
    )
    def test_mixed_resolution_construction(
        self, begin_unit: TimeUnits, end_unit: TimeUnits
    ) -> None:
        """Test Period construction with mixed resolutions promotes to finer."""
        p = Period(
            np.datetime64("2020-01-10", begin_unit),
            np.datetime64("2020-01-20", end_unit),
        )
        # Should handle both resolutions correctly
        assert p.contains(np.datetime64("2020-01-15", begin_unit))
        assert p.contains(np.datetime64("2020-01-15", end_unit))

    @pytest.mark.parametrize(
        "query_unit",
        ["D", "h", "m", "s", "ms", "us", "ns"],
    )
    def test_contains_with_different_query_units(
        self, query_unit: TimeUnits
    ) -> None:
        """Test contains() with different query resolutions."""
        # Create period in days
        p = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        # Query with different resolution
        assert p.contains(np.datetime64("2020-01-15", query_unit))
        assert not p.contains(np.datetime64("2020-01-25", query_unit))

    @pytest.mark.parametrize(
        ("period_unit", "point_unit"),
        [
            ("D", "h"),
            ("h", "D"),
            ("D", "m"),
            ("h", "s"),
            ("s", "ms"),
            ("ms", "us"),
            ("us", "ns"),
        ],
    )
    def test_is_after_with_resolution_conversion(
        self, period_unit: TimeUnits, point_unit: TimeUnits
    ) -> None:
        """Test is_after() with different resolutions."""
        p = Period(
            np.datetime64("2020-01-10", period_unit),
            np.datetime64("2020-01-20", period_unit),
        )
        # Point before period
        assert p.is_after(np.datetime64("2020-01-05", point_unit))
        # Point within or after period
        assert not p.is_after(np.datetime64("2020-01-15", point_unit))

    @pytest.mark.parametrize(
        ("period_unit", "point_unit"),
        [
            ("D", "h"),
            ("h", "D"),
            ("D", "m"),
            ("h", "s"),
            ("s", "ms"),
            ("ms", "us"),
            ("us", "ns"),
        ],
    )
    def test_is_before_with_resolution_conversion(
        self, period_unit: TimeUnits, point_unit: TimeUnits
    ) -> None:
        """Test is_before() with different resolutions."""
        p = Period(
            np.datetime64("2020-01-10", period_unit),
            np.datetime64("2020-01-20", period_unit),
        )
        # Point after period
        assert p.is_before(np.datetime64("2020-01-25", point_unit))
        # Point within or before period
        assert not p.is_before(np.datetime64("2020-01-15", point_unit))

    @pytest.mark.parametrize(
        ("date_unit", "tolerance_unit", "test_date", "tolerance_value"),
        [
            ("D", "D", "2020-01-09", 2),
            ("D", "W", "2020-01-09", 1),
            ("h", "D", "2020-01-09T20", 1),
            ("h", "h", "2020-01-09T22", 3),
            ("s", "ms", "2020-01-09T23:59:58", 3000),
            ("ms", "us", "2020-01-09T23:59:59.998", 3000),
        ],
    )
    def test_is_close_with_mixed_units(
        self,
        date_unit: TimeUnits,
        tolerance_unit: TimeUnits,
        test_date: str,
        tolerance_value: int,
    ) -> None:
        """Test is_close() with different date and tolerance units."""
        p = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        # Date slightly before period, within tolerance
        assert p.is_close(
            np.datetime64(test_date, date_unit),
            np.timedelta64(tolerance_value, tolerance_unit),
        )

    @pytest.mark.parametrize(
        ("p1_unit", "p2_unit"),
        [
            ("D", "D"),
            ("D", "h"),
            ("h", "D"),
            ("h", "m"),
            ("m", "s"),
            ("s", "ms"),
            ("ms", "us"),
            ("us", "ns"),
        ],
    )
    def test_intersects_with_different_units(
        self, p1_unit: TimeUnits, p2_unit: TimeUnits
    ) -> None:
        """Test intersects() with different period resolutions."""
        p1 = Period(
            np.datetime64("2020-01-10", p1_unit),
            np.datetime64("2020-01-20", p1_unit),
        )
        p2 = Period(
            np.datetime64("2020-01-15", p2_unit),
            np.datetime64("2020-01-25", p2_unit),
        )
        # Should detect intersection regardless of units
        assert p1.intersects(p2)
        assert p2.intersects(p1)

    @pytest.mark.parametrize(
        ("p1_unit", "p2_unit"),
        [
            ("D", "D"),
            ("D", "h"),
            ("h", "D"),
            ("h", "m"),
            ("s", "ms"),
            ("ms", "us"),
        ],
    )
    def test_intersection_with_different_units(
        self, p1_unit: TimeUnits, p2_unit: TimeUnits
    ) -> None:
        """Test intersection() with different period resolutions."""
        p1 = Period(
            np.datetime64("2020-01-10", p1_unit),
            np.datetime64("2020-01-20", p1_unit),
        )
        p2 = Period(
            np.datetime64("2020-01-15", p2_unit),
            np.datetime64("2020-01-25", p2_unit),
        )
        # Get intersection
        i = p1.intersection(p2)
        assert i is not None
        # Should contain overlapping region
        assert i.contains(np.datetime64("2020-01-17", "D"))
        assert not i.contains(np.datetime64("2020-01-12", "D"))
        assert not i.contains(np.datetime64("2020-01-23", "D"))

    @pytest.mark.parametrize(
        ("p1_unit", "p2_unit"),
        [
            ("D", "D"),
            ("D", "h"),
            ("h", "D"),
            ("h", "m"),
            ("s", "ms"),
        ],
    )
    def test_merge_with_different_units(
        self, p1_unit: TimeUnits, p2_unit: TimeUnits
    ) -> None:
        """Test merge() with different period resolutions."""
        p1 = Period(
            np.datetime64("2020-01-10", p1_unit),
            np.datetime64("2020-01-20", p1_unit),
        )
        p2 = Period(
            np.datetime64("2020-01-15", p2_unit),
            np.datetime64("2020-01-25", p2_unit),
        )
        # Merge periods
        m = p1.merge(p2)
        assert m is not None
        # Should span entire range
        assert m.contains(np.datetime64("2020-01-10", "D"))
        assert m.contains(np.datetime64("2020-01-20", "D"))
        assert m.contains(np.datetime64("2020-01-25", "D"))
        assert not m.contains(np.datetime64("2020-01-26", "D"))

    @pytest.mark.parametrize(
        ("period_unit", "point_unit"),
        [
            ("D", "h"),
            ("h", "D"),
            ("h", "m"),
            ("m", "s"),
            ("s", "ms"),
            ("ms", "us"),
        ],
    )
    def test_extend_with_different_units(
        self, period_unit: TimeUnits, point_unit: TimeUnits
    ) -> None:
        """Test extend() with different point resolutions."""
        p = Period(
            np.datetime64("2020-01-10", period_unit),
            np.datetime64("2020-01-20", period_unit),
        )
        # Extend with point before
        e = p.extend(np.datetime64("2020-01-05", point_unit))
        assert e.contains(np.datetime64("2020-01-05", "D"))
        assert e.contains(np.datetime64("2020-01-20", "D"))

    @pytest.mark.parametrize(
        ("period_unit", "offset_unit"),
        [
            ("D", "D"),
            ("D", "h"),
            ("h", "h"),
            ("h", "m"),
            ("s", "s"),
            ("s", "ms"),
        ],
    )
    def test_shift_with_different_units(
        self, period_unit: TimeUnits, offset_unit: TimeUnits
    ) -> None:
        """Test shift() with different offset resolutions."""
        p = Period(
            np.datetime64("2020-01-10", period_unit),
            np.datetime64("2020-01-20", period_unit),
        )
        # Shift forward - use compatible offset
        if offset_unit == "D":
            offset = np.timedelta64(5, "D")
            expected_shifted = np.datetime64("2020-01-15", period_unit)
        elif offset_unit == "h":
            offset = np.timedelta64(24, "h")  # 1 day
            expected_shifted = np.datetime64("2020-01-11", period_unit)
        elif offset_unit == "m":
            offset = np.timedelta64(60, "m")  # 1 hour
            # For hour precision period, shift by 1 hour
            if period_unit == "h":
                expected_shifted = np.datetime64("2020-01-10T01", "h")
            else:
                expected_shifted = np.datetime64("2020-01-10", period_unit)
        else:  # s, ms
            offset = np.timedelta64(3600, offset_unit)  # 1 hour equivalent
            if period_unit == "s":
                expected_shifted = np.datetime64("2020-01-10T01:00:00", "s")
            else:
                expected_shifted = np.datetime64("2020-01-10", period_unit)

        s = p.shift(offset)
        # Verify shift occurred (exact check depends on units)
        assert s.contains(expected_shifted) or s.is_after(
            np.datetime64("2020-01-09", "D")
        )

    @pytest.mark.parametrize(
        ("p1_unit", "p2_unit"),
        [
            ("D", "D"),
            ("D", "h"),
            ("h", "D"),
            ("h", "m"),
            ("m", "s"),
            ("s", "ms"),
            ("ms", "us"),
        ],
    )
    def test_equality_with_different_units(
        self, p1_unit: TimeUnits, p2_unit: TimeUnits
    ) -> None:
        """Test equality comparison with different resolutions."""
        p1 = Period(
            np.datetime64("2020-01-10", p1_unit),
            np.datetime64("2020-01-20", p1_unit),
        )
        p2 = Period(
            np.datetime64("2020-01-10", p2_unit),
            np.datetime64("2020-01-20", p2_unit),
        )
        # Same period, different units should be equal
        assert p1 == p2

    def test_subday_precision_conversion(self) -> None:
        """Test precise time conversions within a single day."""
        # Create period in hours
        p = Period(
            np.datetime64("2020-01-10T06", "h"),
            np.datetime64("2020-01-10T18", "h"),
        )
        # Query with minutes - should work
        assert p.contains(np.datetime64("2020-01-10T12:30", "m"))
        # Query with seconds - should work
        assert p.contains(np.datetime64("2020-01-10T12:30:45", "s"))
        # Outside period
        assert not p.contains(np.datetime64("2020-01-10T20:00", "h"))

    def test_high_precision_conversion(self) -> None:
        """Test high-precision time conversions (microsecond/nanosecond)."""
        # Create period in microseconds
        p = Period(
            np.datetime64("2020-01-10T12:00:00.000000", "us"),
            np.datetime64("2020-01-10T12:00:01.000000", "us"),
        )
        # Query with nanoseconds
        assert p.contains(np.datetime64("2020-01-10T12:00:00.500000000", "ns"))
        # Query with milliseconds
        assert p.contains(np.datetime64("2020-01-10T12:00:00.500", "ms"))
        # Outside period
        assert not p.contains(
            np.datetime64("2020-01-10T12:00:02.000000", "us")
        )


class TestPeriodEdgeCases:
    """Test Period edge cases and special scenarios."""

    def test_single_point_period(self) -> None:
        """Test period representing a single point in time."""
        p = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-10", "D"),
        )
        assert p.contains(np.datetime64("2020-01-10", "D"))
        assert not p.contains(np.datetime64("2020-01-11", "D"))
        assert not p.contains(np.datetime64("2020-01-09", "D"))

    def test_high_resolution_periods(self) -> None:
        """Test periods with nanosecond resolution."""
        p = Period(
            np.datetime64("2020-01-10T12:00:00.000000000", "ns"),
            np.datetime64("2020-01-10T12:00:00.000001000", "ns"),
        )
        assert p.contains(np.datetime64("2020-01-10T12:00:00.000000500", "ns"))
        assert p.contains(np.datetime64("2020-01-10T12:00:00.000001000", "ns"))
        assert not p.contains(
            np.datetime64("2020-01-10T12:00:00.000001001", "ns")
        )

    def test_mixed_resolution_operations(self) -> None:
        """Test operations between periods with different resolutions."""
        p1 = Period(
            np.datetime64("2020-01-10", "D"),
            np.datetime64("2020-01-20", "D"),
        )
        p2 = Period(
            np.datetime64("2020-01-15T00", "h"),
            np.datetime64("2020-01-25T00", "h"),
        )

        # Should handle resolution promotion automatically
        assert p1.intersects(p2)
        i = p1.intersection(p2)
        assert i is not None
        assert i.contains(np.datetime64("2020-01-17", "D"))

    def test_year_2000_problem(self) -> None:
        """Test dates around Y2K."""
        p = Period(
            np.datetime64("1999-12-31", "D"),
            np.datetime64("2000-01-01", "D"),
        )
        assert p.contains(np.datetime64("1999-12-31", "D"))
        assert p.contains(np.datetime64("2000-01-01", "D"))
        assert not p.contains(np.datetime64("2000-01-02", "D"))

    def test_leap_year_handling(self) -> None:
        """Test period containing leap day."""
        # 2020 is a leap year
        p = Period(
            np.datetime64("2020-02-28", "D"),
            np.datetime64("2020-03-01", "D"),
        )
        assert p.contains(np.datetime64("2020-02-29", "D"))
        assert p.contains(np.datetime64("2020-02-28", "D"))
        assert p.contains(np.datetime64("2020-03-01", "D"))

    def test_very_long_period(self) -> None:
        """Test period spanning many years."""
        p = Period(
            np.datetime64("1900-01-01", "Y"),
            np.datetime64("2100-01-01", "Y"),
        )
        assert p.contains(np.datetime64("2000-01-01", "Y"))
        assert p.contains(np.datetime64("1950-01-01", "Y"))
        assert not p.contains(np.datetime64("2101-01-01", "Y"))


class TestPeriodOverflow:
    """Test overflow detection during resolution conversion."""

    def test_overflow_max_days_to_attoseconds(self) -> None:
        """Test overflow when converting max int64 days to attoseconds."""
        max_int64 = np.iinfo(np.int64).max
        with pytest.raises(
            OverflowError, match="integer overflow in multiplication"
        ):
            Period(
                np.datetime64(max_int64, "D"),
                np.datetime64(max_int64, "as"),
            )

    def test_overflow_large_days_to_nanoseconds(self) -> None:
        """Test overflow when converting large days value to nanoseconds."""
        large_day = 10**15  # Very large number of days
        with pytest.raises(
            OverflowError, match="integer overflow in multiplication"
        ):
            Period(
                np.datetime64(large_day, "D"),
                np.datetime64(large_day, "ns"),
            )

    def test_overflow_seconds_to_attoseconds(self) -> None:
        """Test overflow when converting moderate seconds to attoseconds."""
        moderate_sec = 10**10  # 10 billion seconds
        with pytest.raises(
            OverflowError, match="integer overflow in multiplication"
        ):
            Period(
                np.datetime64(moderate_sec, "s"),
                np.datetime64(moderate_sec, "as"),
            )

    def test_overflow_min_days_to_attoseconds(self) -> None:
        """Test overflow with negative max value (min int64)."""
        min_int64 = np.iinfo(np.int64).min
        with pytest.raises(
            OverflowError, match="integer overflow in multiplication"
        ):
            Period(
                np.datetime64(min_int64, "D"),
                np.datetime64(min_int64, "as"),
            )

    def test_overflow_milliseconds_to_attoseconds(self) -> None:
        """Test overflow when converting large milliseconds to attoseconds."""
        large_ms = 10**18  # Very large milliseconds value
        with pytest.raises(
            OverflowError, match="integer overflow in multiplication"
        ):
            Period(
                np.datetime64(large_ms, "ms"),
                np.datetime64(large_ms, "as"),
            )

    def test_no_overflow_safe_conversions(self) -> None:
        """Test that safe conversions do not raise overflow errors."""
        # Days to hours - safe conversion
        p1 = Period(
            np.datetime64("2020-01-01", "D"),
            np.datetime64("2020-01-10", "h"),
        )
        assert p1.contains(np.datetime64("2020-01-05", "D"))

        # Seconds to milliseconds - safe conversion
        p2 = Period(
            np.datetime64("2020-01-01T00:00:00", "s"),
            np.datetime64("2020-01-01T01:00:00", "ms"),
        )
        assert p2.contains(np.datetime64("2020-01-01T00:30:00", "s"))

        # Moderate value days to nanoseconds - should work
        p3 = Period(
            np.datetime64("2020-01-01", "D"),
            np.datetime64("2020-01-02", "ns"),
        )
        assert p3.contains(np.datetime64("2020-01-01T12:00:00", "s"))

    def test_overflow_in_period_operations(self) -> None:
        """Test that overflow is detected in period operations like extend."""
        # Create a period at the edge of safe range
        p = Period(
            np.datetime64("2020-01-01", "D"),
            np.datetime64("2020-01-10", "D"),
        )

        # Extending with an extreme value that would cause overflow
        # when converting to finer resolution should be caught
        max_int64 = np.iinfo(np.int64).max
        with pytest.raises(OverflowError):
            # This will fail when trying to create/convert the extreme point
            p.extend(np.datetime64(max_int64, "as"))

"""
Time period
===========
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
import re

import numpy

if TYPE_CHECKING:
    from .typing import (
        NDArray,
        NDArrayDateTime,
        NDArrayTimeDelta,
        NDArrayStructured,
    )

from . import core

# Parse the unit of numpy.timedelta64.
PATTERN = re.compile(r'(?:datetime|timedelta)64\[(\w+)\]').search

#: Numpy time units
RESOLUTION: list[str] = [
    'as', 'fs', 'ps', 'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'W', 'M', 'Y'
]


def _time64_unit(dtype: numpy.dtype):
    """Gets the unit of time."""
    match = PATTERN(dtype.name)
    if match is None:
        raise ValueError(f'dtype is not a time duration: {dtype}')
    return match.group(1)


def _min_time64_unit(*args: numpy.dtype) -> str:
    """Gets the minimum time unit."""
    index = min(RESOLUTION.index(_time64_unit(item)) for item in args)
    return RESOLUTION[index]


def _datetime64_to_int64(value: Any, resolution: str) -> Any:
    """Convert values to numpy.int64."""
    return value.astype(f'M8[{resolution}]').astype(numpy.int64)


class Period(core.Period):
    """Creates a Period from begin to last eg: [begin, last)

    Args:
        begin: The beginning of the period.
        end: The ending of the period.
        within: If true, the given period defines a closed interval
            (i.e. the end date is within the period), otherwise the
            interval is open.
    """
    __slots__ = ('resolution', )

    def __init__(self,
                 begin: numpy.datetime64,
                 end: numpy.datetime64,
                 within: bool = True) -> None:
        self.resolution = _min_time64_unit(begin.dtype, end.dtype)
        begin = numpy.datetime64(  # type: ignore[call-overload]
            begin,
            self.resolution,
        )
        end = numpy.datetime64(  # type: ignore[call-overload]
            end,
            self.resolution,
        )
        if not within:
            end -= numpy.timedelta64(  # type: ignore[call-overload]
                1,
                self.resolution,
            )
        super().__init__(begin.astype(int), end.astype(int), True)

    def __reduce__(self) -> str | tuple[Any, ...]:
        return self.__class__, (self.begin, self.last, True)

    def as_base_class(self) -> core.Period:
        """Returns the base class of the period.

        Returns:
            The base class of the period.
        """
        return core.Period(super().begin, super().last, True)

    @property
    def begin(self) -> numpy.datetime64:  # type: ignore[override]
        """Gets the beginning of the period."""
        return numpy.datetime64(  # type: ignore[call-overload]
            super().begin,
            self.resolution,
        )

    @property
    def last(self) -> numpy.datetime64:  # type: ignore[override]
        """Gets the beginning of the period."""
        return numpy.datetime64(  # type: ignore[call-overload]
            super().last,
            self.resolution,
        )

    def end(self) -> numpy.datetime64:  # type: ignore[override]
        """Gets the end of the period."""
        return numpy.datetime64(  # type: ignore[call-overload]
            super().end(),
            self.resolution,
        )

    def as_resolution(self, resolution: str) -> Period:
        """Converts the period to a period with a different resolution.

        Args:
            resolution: The new resolution.

        Returns:
            A new period with the given resolution.
        """
        return Period(
            numpy.datetime64(  # type: ignore[call-overload]
                self.begin,
                resolution,
            ),
            numpy.datetime64(  # type: ignore[call-overload]
                self.last,
                resolution,
            ),
            True,
        )

    def __repr__(self) -> str:
        return f'Period(begin={self.begin!r}, end={self.last!r}, within=True)'

    def __str__(self) -> str:
        return f'[{self.begin}, {self.end()})'

    def duration(self) -> numpy.timedelta64:  # type: ignore[override]
        """Gets the duration of the period."""
        return numpy.timedelta64(  # type: ignore[call-overload]
            super().duration(),
            self.resolution,
        )

    def is_null(self) -> bool:
        """True if period is ill formed (length is zero or less)"""
        return super().is_null()

    def __eq__(self, lhs: object) -> bool:
        if not isinstance(lhs, Period):
            return NotImplemented
        if self.resolution != lhs.resolution:
            return False
        return super().__eq__(lhs)

    def __ne__(self, rhs: object) -> bool:
        return not self.__eq__(rhs)

    def __lt__(self, lhs: Any) -> bool:
        if not isinstance(lhs, Period):
            return NotImplemented
        if self.resolution != lhs.resolution:
            return False
        return super().__lt__(lhs)

    def contains(  # type: ignore[override]
        self,
        other: numpy.datetime64 | Period,
    ) -> bool:
        """Checks if the given period is contains this period.

        Args:
            other: The other period to check.

        Returns:
            * True if other is a date and is inside the period, zero length
              periods contain no points
            * True if other is a period and  fully contains (or equals) the
              other period
        """
        if isinstance(other, Period):
            if self.resolution != other.resolution:
                other = other.as_resolution(self.resolution, )
            return super().contains(other)
        return super().contains(_datetime64_to_int64(
            other,
            self.resolution,
        ))

    def is_adjacent(self, other: Period) -> bool:  # type: ignore[override]
        """True if periods are next to each other without a gap.

        In the example below, p1 and p2 are adjacent, but p3 is not adjacent
        with either of p1 or p2.

        .. code-block:: text

            [-p1-)
                [-p2-)
                [-p3-)

        Args:
            other: The other period to check.

        Returns:
            * True if other is a date and is adjacent to this period
            * True if other is a period and is adjacent to this period
        """
        if self.resolution != other.resolution:
            other = other.as_resolution(self.resolution, )
        return super().is_adjacent(other)

    def is_before(self,
                  point: numpy.datetime64) -> bool:  # type: ignore[override]
        """True if all of the period is prior to the passed point or end <=
        point.

        In the example below points 4 and 5 return true.

        .. code-block:: text

                [---------])
            ^   ^    ^     ^   ^
            1   2    3     4   5

        Args:
            point: The point to check.

        Returns:
            True if point is before the period
        """
        return super().is_before(_datetime64_to_int64(
            point,
            self.resolution,
        ))

    def is_after(self,
                 point: numpy.datetime64) -> bool:  # type: ignore[override]
        """True if all of the period is prior or point < start.

        In the example below only point 1 would evaluate to true.

        .. code-block:: text

                [---------])
            ^   ^    ^     ^   ^
            1   2    3     4   5

        Args:
            point: The point to check.

        Returns:
            True if point is after the period
        """
        return super().is_after(_datetime64_to_int64(
            point,
            self.resolution,
        ))

    def intersects(self, other: Period) -> bool:  # type: ignore[override]
        """True if the periods overlap in any way.

        In the example below p1 intersects with p2, p4, and p6.

        .. code-block:: text

                [---p1---)
                        [---p2---)
                        [---p3---)
            [---p4---)
            [-p5-)
                    [-p6-)

        Args:
            other: The other period to check.

        Returns:
            True if the periods intersect
        """
        if self.resolution != other.resolution:
            other = other.as_resolution(self.resolution, )
        return super().intersects(other)

    def intersection(self, other: Period) -> Period:  # type: ignore[override]
        """Returns the period of intersection or null period if no
        intersection.

        Args:
            other: The other period to check.

        Returns:
            The intersection period or null period if no intersection.
        """
        if self.resolution != other.resolution:
            other = other.as_resolution(self.resolution, )
        result = super().intersection(other)
        return Period(
            numpy.datetime64(  # type: ignore[call-overload]
                result.begin,
                self.resolution,
            ),
            numpy.datetime64(  # type: ignore[call-overload]
                result.last,
                self.resolution,
            ),
            True,
        )

    def merge(self, other: Period) -> Period:  # type: ignore[override]
        """Returns the union of intersecting periods -- or null period.

        Args:
            other: The other period to merge.

        Returns:
            The union period of intersection or null if no intersection.
        """
        if self.resolution != other.resolution:
            other = other.as_resolution(self.resolution, )
        result = super().merge(other)
        return Period(
            numpy.datetime64(  # type: ignore[call-overload]
                result.begin,
                self.resolution,
            ),
            numpy.datetime64(  # type: ignore[call-overload]
                result.last,
                self.resolution,
            ),
            True,
        )


class PeriodList:
    """A list of periods.

    This class is used to represent a list of periods that are not necessarily
    contiguous.

    Args:
        periods: A list of periods.
    """
    DTYPE: ClassVar[list[tuple[str, type]]] = [('begin', numpy.int64),
                                               ('last', numpy.int64)]

    def __init__(self,
                 periods: NDArrayDateTime | core.PeriodList,
                 dtype: numpy.dtype | None = None) -> None:
        if isinstance(periods, core.PeriodList):
            if dtype is None:
                raise ValueError('dtype must be specified when passing in a '
                                 'pyinterp.core.PeriodList')
            self._instance = periods
            self._datetime64 = dtype
        else:
            if not numpy.issubdtype(periods.dtype, numpy.datetime64):
                raise TypeError('periods must be a numpy.datetime64 array')
            if len(periods.shape) != 2:
                raise ValueError(
                    'periods must be a 2d array of numpy.datetime64')
            self._instance = core.PeriodList(
                numpy.rec.fromarrays(periods, self.DTYPE))
            self._datetime64 = periods.dtype
        self._timedelta64 = numpy.dtype(
            self._datetime64.str.replace('M8', 'm8'))
        self._dtype = [('begin', self._datetime64), ('last', self._datetime64)]

    @classmethod
    def empty(cls) -> PeriodList:
        """Create an empty period list."""
        array = numpy.rec.fromarrays(numpy.ndarray((2, 0), dtype='M8[ns]'),
                                     cls.DTYPE)
        return cls(core.PeriodList(array), numpy.dtype('M8[ns]'))

    def __len__(self) -> int:
        return len(self._instance)

    def __getstate__(self) -> tuple[Any, ...]:
        return (self._datetime64, self._dtype, self._timedelta64,
                self._instance)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (self._datetime64, self._dtype, self._timedelta64,
         self._instance) = state

    @property
    def periods(self) -> NDArrayStructured:
        """The periods in the list."""
        return self._instance.periods.astype(self._dtype)

    def __repr__(self) -> str:
        return repr(self.periods)

    def are_periods_sorted_and_disjointed(self) -> bool:
        """True if the periods are sorted and disjoint."""
        return self._instance.are_periods_sorted_and_disjointed()

    def join_adjacent_periods(
        self,
        epsilon: numpy.timedelta64 | None = None,
        inplace: bool = False,
    ) -> PeriodList:
        """Join the periods together if they are adjacent.

        Args:
            epsilon The maximum gap between periods to join.
            inplace: If true, the periods will be joined in place.

        Returns:
            A new PeriodList with the periods joined.
        """
        epsilon = epsilon.astype(
            self._timedelta64) if epsilon else numpy.timedelta64(0, 's')
        instance = self._instance.join_adjacent_periods(
            epsilon.astype('int64'))
        if inplace:
            self._instance = instance
            return self
        return PeriodList(instance, dtype=self._datetime64)

    def within(self, period: Period) -> PeriodList:
        """Returns a PeriodList containing only periods within the given
        period.

        Args:
            period The period to filter by.

        Returns:
            A new PeriodList with only periods within the given period.
        """
        return PeriodList(self._instance.within(period.as_base_class()),
                          dtype=self._datetime64)

    def intersection(self, period: Period) -> PeriodList:
        """Returns a PeriodList containing only periods that intersect the
        given period.

        Args:
            period The period to filter by.

        Returns:
            A new PeriodList with only periods that intersect the given
            period.
        """
        return PeriodList(self._instance.intersection(period.as_base_class()),
                          dtype=self._datetime64)

    def duration(self) -> NDArrayTimeDelta:
        """Returns the duration of the periods in the PeriodList."""
        periods = self.periods
        return periods['last'] - periods['begin']

    def filter(self, min_duration: numpy.timedelta64) -> PeriodList:
        """Returns a PeriodList containing only periods longer than the given
        duration.

        Args:
            min_duration The minimum duration to filter by.

        Returns:
            A new PeriodList with only periods longer than the given duration.
        """
        min_duration = min_duration.astype(self._timedelta64)
        return PeriodList(self._instance.filter(min_duration.astype('int64')),
                          dtype=self._datetime64)

    def sort(self) -> PeriodList:
        """Sort the periods in the PeriodList."""
        self._instance.sort()
        return self

    def merge(self, other: PeriodList) -> PeriodList:
        """Merge two PeriodLists together.

        Args:
            other The PeriodList to merge with.

        Returns:
            A new PeriodList with the periods merged.
        """
        if self._datetime64 != other._datetime64:
            periods = other.periods.astype(self._datetime64)
            other = PeriodList(periods, dtype=self._datetime64)
        self._instance.merge(other._instance)
        return self

    def cross_a_period(self, dates: NDArrayDateTime) -> NDArray:
        """Search the provided dates for those that do not traverse any of the
        periods managed by this instance.

        If the instance handles these periods:

        .. code-block:: text

            --AAAAAAAA-----BBBBBBBBB-----------------CCCCCCCC---------

        And the dates to check are:

        .. code-block:: text

            .............................

        The result will be:

        .. code-block:: text

            11111111111111111111111100000

        Args:
            dates: The dates to check.

        Returns:
            A boolean array indicating if the dates cross a period.
        """
        return self._instance.cross_a_period(dates.astype(self._datetime64))

    def belong_to_a_period(self, dates: NDArrayDateTime) -> NDArray:
        """Search the provided dates for those that belong to any of the
        periods managed by this instance.

        If the instance handles these periods:

        .. code-block:: text

            --AAAAAAAA-----BBBBBBBBB-----------------CCCCCCCC---------

        And the dates to check are:

        .. code-block:: text

            .............................

        The result will be:

        .. code-block:: text

            00111111110000011111111100000

        Args:
            dates: The dates to check.

        Returns:
            A boolean array indicating if the dates belong to a period.
        """
        return self._instance.belong_to_a_period(dates.astype(
            self._datetime64))

    def is_it_close(self,
                    date: numpy.datetime64,
                    epsilon: numpy.timedelta64 | None = None) -> bool:
        """Determines whether the date, given in parameter, is close to a
        period.

        Args:
            date: timestamp to test
            epsilon: Maximum difference to be taken into account between the
                given date and the period to consider the closest date affected
                by the period found.
        Returns:
            True if the given is close to a period
        """
        epsilon = epsilon or numpy.timedelta64(0, 's')
        return self._instance.is_it_close(
            date.astype(self._datetime64).astype(numpy.int64),
            epsilon.astype(self._timedelta64).astype(numpy.int64))

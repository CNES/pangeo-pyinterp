"""Time period."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
import re
from types import GenericAlias

import numpy
import numpy.typing

if TYPE_CHECKING:
    from .typing import (
        NDArray1DBool,
        NDArray1DDateTime,
        NDArray1DTimeDelta,
        NDArray2DDateTime,
        NDArrayStructured,
    )

from . import core

# Parse the unit of numpy.timedelta64.
PATTERN = re.compile(r'(?:datetime|timedelta)64\[(\w+)\]').search

#: Numpy time units
RESOLUTION: list[str] = [
    'as', 'fs', 'ps', 'ns', 'us', 'ms', 's', 'm', 'h', 'D', 'W', 'M', 'Y'
]

#: Type of the structured array representing a period.
DTypePeriod = numpy.dtype([('begin', numpy.datetime64),
                           ('last', numpy.datetime64)])

if TYPE_CHECKING:
    #: Type alias for a structured array representing periods.
    NDArrayPeriodType = numpy.ndarray[tuple[int], numpy.dtype[numpy.void]]
else:
    NDArrayPeriodType = GenericAlias(
        numpy.ndarray,
        (tuple[int], DTypePeriod),
    )


def _time64_unit(dtype: numpy.dtype) -> str:
    """Get the unit of time.

    Extract the time unit from a numpy datetime64 or timedelta64 dtype.

    Args:
        dtype: The numpy dtype to extract the unit from.

    Returns:
        The time unit string.

    Raises:
        ValueError: If the dtype is not a time duration.

    """
    match = PATTERN(dtype.name)
    if match is None:
        raise ValueError(f'dtype is not a time duration: {dtype}')
    return match.group(1)


def _min_time64_unit(*args: numpy.dtype) -> str:
    """Get the minimum time unit.

    Find the finest time resolution among the provided dtypes.

    Args:
        *args: Variable number of numpy dtypes to compare.

    Returns:
        The minimum time unit string.

    """
    index = min(RESOLUTION.index(_time64_unit(item)) for item in args)
    return RESOLUTION[index]


def _datetime64_to_int64(
    value: numpy.datetime64,
    resolution: str,
) -> int:
    """Convert values to numpy.int64."""
    return value.astype(f'M8[{resolution}]').astype(numpy.int64).item()


class Period(core.Period):
    """Represent a time period with a begin and end time.

    Create a time period defined by a begin and end point. The end point is
    the last point in the time period and must be the same or greater than
    the begin point.

    Args:
        begin: The starting point of the period.
        last: The last point in the period.

    Raises:
        ValueError: If last < begin.

    """

    __slots__ = ('resolution', )

    def __init__(self,
                 begin: numpy.datetime64,
                 end: numpy.datetime64,
                 within: bool = True) -> None:
        """Initialize a Period instance."""
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
        """Get the state for serialization."""
        return self.__class__, (self.begin, self.last, True)

    def as_base_class(self) -> core.Period:
        """Convert this Period to its base class representation.

        Returns:
            The base class representation of the period.

        """
        return core.Period(super().begin, super().last, True)

    @property
    def begin(self) -> numpy.datetime64:  # type: ignore[override]
        """Get the beginning of the period.

        Returns:
            The starting point of the period as a numpy datetime64.

        """
        return numpy.datetime64(  # type: ignore[call-overload]
            super().begin,
            self.resolution,
        )

    @property
    def last(self) -> numpy.datetime64:  # type: ignore[override]
        """Get the last time point in the period.

        Returns:
            The last point in the period as a numpy datetime64.

        """
        return numpy.datetime64(  # type: ignore[call-overload]
            super().last,
            self.resolution,
        )

    def end(self) -> numpy.datetime64:  # type: ignore[override]
        """Get the ending point of the time period.

        The end point is one unit past the last point in the period.

        Returns:
            The ending point (one unit past the last point).

        """
        return numpy.datetime64(  # type: ignore[call-overload]
            super().end(),
            self.resolution,
        )

    def as_resolution(self, resolution: str) -> Period:
        """Convert the period to a different time resolution.

        Args:
            resolution: The new resolution string (e.g., 'D', 'h', 's').

        Returns:
            A new period with the specified resolution.

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
        """Get the string representation of the period."""
        return f'Period(begin={self.begin!r}, end={self.last!r}, within=True)'

    def __str__(self) -> str:
        """Get the string representation of the period."""
        return f'[{self.begin}, {self.end()})'

    def __hash__(self) -> int:
        """Compute the hash value of the period."""
        return hash((self.begin, self.last, self.resolution))

    def duration(self) -> numpy.timedelta64:  # type: ignore[override]
        """Calculate the duration of the period.

        Returns:
            The duration as a numpy timedelta64 (end - begin).

        """
        return numpy.timedelta64(  # type: ignore[call-overload]
            super().duration(),
            self.resolution,
        )

    def is_null(self) -> bool:
        """Check if the period is ill-formed.

        Returns:
            True if the period length is zero or less.

        """
        return super().is_null()

    def __eq__(self, lhs: object) -> bool:
        """Equal comparison between periods."""
        if not isinstance(lhs, Period):
            return NotImplemented
        if self.resolution != lhs.resolution:
            return False
        return super().__eq__(lhs)

    def __ne__(self, rhs: object) -> bool:
        """Not-equal comparison between periods."""
        return not self.__eq__(rhs)

    def __lt__(self, lhs: object) -> bool:
        """Less-than comparison between periods."""
        if not isinstance(lhs, Period):
            return NotImplemented
        if self.resolution != lhs.resolution:
            return False
        return super().__lt__(lhs)

    def contains(  # type: ignore[override]
        self,
        other: numpy.datetime64 | Period,
    ) -> bool:
        """Check if this period contains the given point or period.

        Args:
            other: The datetime64 point or Period to check for containment.

        Returns:
            * True if other is a date and is inside the period (zero length
              periods contain no points)
            * True if other is a period and this period fully contains (or
              equals) the other period

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
        """Check if two periods are next to each other without a gap.

        In the example below, p1 and p2 are adjacent, but p3 is not adjacent
        with either of p1 or p2.

        .. code-block:: text

            [-p1-)
                [-p2-)
                [-p3-)

        Args:
            other: The other period to check adjacency with.

        Returns:
            True if the other period is adjacent to this period.

        """
        if self.resolution != other.resolution:
            other = other.as_resolution(self.resolution, )
        return super().is_adjacent(other)

    def is_before(  # type: ignore[override]
            self,
            point: numpy.datetime64,
    ) -> bool:
        """Check if the entire period is prior to the passed point.

        Test if end <= point. In the example below, points 4 and 5 return true.

        .. code-block:: text

                [---------])
            ^   ^    ^     ^   ^
            1   2    3     4   5

        Args:
            point: The datetime64 point to compare against.

        Returns:
            True if the entire period is before the point.

        """
        return super().is_before(_datetime64_to_int64(
            point,
            self.resolution,
        ))

    def is_after(  # type: ignore[override]
            self,
            point: numpy.datetime64,
    ) -> bool:
        """Check if the entire period is after the passed point.

        Test if point < start. In the example below, only point 1 evaluates to
        true.

        .. code-block:: text

                [---------])
            ^   ^    ^     ^   ^
            1   2    3     4   5

        Args:
            point: The datetime64 point to compare against.

        Returns:
            True if the entire period is after the point.

        """
        return super().is_after(_datetime64_to_int64(
            point,
            self.resolution,
        ))

    def intersects(self, other: Period) -> bool:  # type: ignore[override]
        """Check if the periods overlap in any way.

        In the example below, p1 intersects with p2, p4, and p6.

        .. code-block:: text

                [---p1---)
                        [---p2---)
                        [---p3---)
            [---p4---)
            [-p5-)
                    [-p6-)

        Args:
            other: The other period to test for intersection.

        Returns:
            True if the periods overlap.

        """
        if self.resolution != other.resolution:
            other = other.as_resolution(self.resolution, )
        return super().intersects(other)

    def intersection(self, other: Period) -> Period:  # type: ignore[override]
        """Calculate the period of intersection.

        Args:
            other: The other period to intersect with.

        Returns:
            The intersection period, or a null period if no intersection exists.

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
        """Calculate the union of intersecting periods.

        Args:
            other: The other period to merge with.

        Returns:
            The merged period if periods intersect, or a null period otherwise.

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
                 periods: NDArrayStructured | NDArray2DDateTime
                 | core.PeriodList,
                 dtype: numpy.dtype | None = None) -> None:
        """Initialize a PeriodList instance."""
        if isinstance(periods, core.PeriodList):
            if dtype is None:
                raise ValueError('dtype must be specified when passing in a '
                                 'pyinterp.core.PeriodList')
            self._instance = periods
            self._datetime64 = dtype
        else:
            if not numpy.issubdtype(periods.dtype, numpy.datetime64):
                raise TypeError('periods must be a numpy.datetime64 array')
            if len(periods.shape) != 2:  # noqa: PLR2004
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
        """Get the number of periods in the list."""
        return len(self._instance)

    def __getstate__(self) -> tuple[Any, ...]:
        """Get the state of the object for serialization."""
        return (self._datetime64, self._dtype, self._timedelta64,
                self._instance)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        """Restore the state of the object from serialization."""
        (self._datetime64, self._dtype, self._timedelta64,
         self._instance) = state

    @property
    def periods(self) -> NDArrayPeriodType:
        """Get the periods in the list.

        Returns:
            The periods as a structured numpy array.

        """
        return self._instance.periods.astype(self._dtype)

    def __repr__(self) -> str:
        """Get the string representation of this instance."""
        return repr(self.periods)

    def are_periods_sorted_and_disjointed(self) -> bool:
        """Check if the periods are sorted and disjoint.

        Returns:
            True if periods are sorted chronologically and do not overlap.

        """
        return self._instance.are_periods_sorted_and_disjointed()

    def join_adjacent_periods(
        self,
        epsilon: numpy.timedelta64 | None = None,
        inplace: bool = False,
    ) -> PeriodList:
        """Join adjacent periods together.

        Args:
            epsilon: The maximum gap between periods to consider them adjacent.
            inplace: If True, modify this object in place rather than creating
            a new one.

        Returns:
            A PeriodList with adjacent periods joined together.

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
        """Filter to only periods within the given period.

        Args:
            period: The period to use as a filter.

        Returns:
            A new PeriodList containing only periods within the given period.

        """
        return PeriodList(self._instance.within(period.as_base_class()),
                          dtype=self._datetime64)

    def intersection(self, period: Period) -> PeriodList:
        """Filter to only periods that intersect the given period.

        Args:
            period: The period to test for intersection.

        Returns:
            A new PeriodList containing only periods that intersect the given
            period.

        """
        return PeriodList(self._instance.intersection(period.as_base_class()),
                          dtype=self._datetime64)

    def duration(self) -> NDArray1DTimeDelta:
        """Calculate the duration of each period in the list.

        Returns:
            An array of timedelta64 values representing each period's duration.

        """
        periods = self.periods
        return periods['last'] - periods['begin']

    def filter(self, min_duration: numpy.timedelta64) -> PeriodList:
        """Filter to only periods longer than the given duration.

        Args:
            min_duration: The minimum duration threshold.

        Returns:
            A new PeriodList containing only periods with
            duration >= min_duration.

        """
        min_duration = min_duration.astype(self._timedelta64)
        return PeriodList(self._instance.filter(min_duration.astype('int64')),
                          dtype=self._datetime64)

    def sort(self) -> PeriodList:
        """Sort the periods chronologically in place.

        Returns:
            This PeriodList instance with sorted periods.

        """
        self._instance.sort()
        return self

    def merge(self, other: PeriodList) -> PeriodList:
        """Merge another PeriodList into this one.

        Args:
            other: The PeriodList to merge into this one.

        Returns:
            This PeriodList instance with the other periods merged in.

        """
        if self._datetime64 != other._datetime64:
            periods = other.periods.astype(self._datetime64)
            other = PeriodList(periods, dtype=self._datetime64)
        self._instance.merge(other._instance)
        return self

    def cross_a_period(self, dates: NDArray1DDateTime) -> NDArray1DBool:
        """Search for dates that do not traverse any managed period.

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
            A boolean array where True indicates the date crosses a period
            boundary.

        """
        return self._instance.cross_a_period(dates.astype(self._datetime64))

    def belong_to_a_period(self, dates: NDArray1DDateTime) -> NDArray1DBool:
        """Search for dates that belong to any managed period.

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
            A boolean array where True indicates the date belongs to a period.

        """
        return self._instance.belong_to_a_period(dates.astype(
            self._datetime64))

    def is_it_close(self,
                    date: numpy.datetime64,
                    epsilon: numpy.timedelta64 | None = None) -> bool:
        """Determine whether a date is close to any managed period.

        Args:
            date: The timestamp to test.
            epsilon: Maximum time difference to consider a date as close to a
                period. Defaults to zero if not specified.

        Returns:
            True if the date is within epsilon of any period boundary.

        """
        epsilon = epsilon or numpy.timedelta64(0, 's')
        return self._instance.is_it_close(
            date.astype(self._datetime64).astype(numpy.int64),
            epsilon.astype(self._timedelta64).astype(numpy.int64))

# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Axis
====
"""
from typing import Type, Union
import re
import warnings
import numpy as np
from . import core


class TemporalAxis(core.TemporalAxis):
    """Time axis.
    """
    #: Pattern to parse numpy time units
    PATTERN = re.compile(r"\[([^\]]*)\]").search

    #: Numpy time units
    RESOLUTION = [
        "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as"
    ]

    #: Numpy time unit meanings
    UNITS = [
        "year", "month", "week", "day", "hour", "minute", "second",
        "millisecond", "microsecond", "nanosecond", "picosecond",
        "femtosecond", "attosecond"
    ]

    def __init__(self, values: np.ndarray):
        """
        Create a coordinate axis from values.

        Args:
            values (numpy.ndarray): Items representing the datetimes or
            timedeltas of the axis.

        Raises:
            TypeError: if the array data type is not a datetime64 subtype.

        Examples:

            >>> import datetime
            >>> import numpy as np
            >>> import pyinterp
            >>> start = datetime.datetime(2000, 1, 1)
            >>> values = np.array([
            ...     start + datetime.timedelta(hours=index)
            ...     for index in range(86400)
            ... ],
            ...                   dtype="datetime64[us]")
            >>> axis = pyinterp.TemporalAxis(values)
            >>> axis
            TemporalAxis(array(['2000-01-01T00:00:00.000000',
                                '2000-01-01T01:00:00.000000',
                                '2000-01-01T02:00:00.000000',
                                ...,
                                '2009-11-08T21:00:00.000000',
                                '2009-11-08T22:00:00.000000',
                                '2009-11-08T23:00:00.000000'],
                            dtype='datetime64[us]'))
            >>> values = np.array([
            ...     datetime.timedelta(hours=index)
            ...     for index in range(86400)
            ... ],
            ...                   dtype="timedelta64[us]")
            >>> axis = pyinterp.TemporalAxis(values)
            >>> axis
            TemporalAxis(array([0,
                                3600000000,
                                7200000000,
                                ...,
                                311029200000000,
                                311032800000000,
                                311036400000000],
                         dtype='timedelta64[us]'))
        """
        self._assert_issubdtype(values.dtype)
        super().__init__(values.astype("int64"))
        self.dtype = values.dtype
        self.object = self._object(self.dtype)
        self.resolution = self._npdate_resolution(str(self.dtype))

    @staticmethod
    def _assert_issubdtype(dtype: np.dtype) -> None:
        if not np.issubdtype(dtype,
                             np.dtype("datetime64")) and not np.issubdtype(
                                 dtype, np.dtype("timedelta64")):
            raise TypeError("values must be a numpy datetime/timedelta array")

    @staticmethod
    def _object(dtype: np.dtype) -> Type:
        """Get the object type handled by this class."""
        data_type = str(dtype)
        return getattr(np, data_type[:data_type.index('64') + 2])

    def safe_cast(self, values: np.ndarray) -> np.ndarray:
        """Convert the dates of the vector in the same unit as the time axis
        defined in this instance.

        Args:
            values (numpy.ndarray): Values to convert.

        Returns:
            numpy.ndarray: values converted.

        Raises:
            UserWarning: if the implicit conversion from the unit of dates
                provided to the unit of the axis, truncates the dates
                (e.g. converting microseconds to seconds).
        """
        self._assert_issubdtype(values.dtype)
        resolution = self._npdate_resolution(str(values.dtype))
        source_idx = self.RESOLUTION.index(resolution)
        target_idx = self.RESOLUTION.index(self.resolution)
        if source_idx != target_idx:
            if source_idx > target_idx:
                source = self.UNITS[source_idx]
                target = self.UNITS[target_idx]
                warnings.warn(
                    "implicit conversion turns "
                    f"{source} into {target}", UserWarning)
            return values.astype(
                f"{self.object.__name__}[{self.resolution}]").astype("int64")
        return values.astype("int64")

    def back(self) -> Union[np.datetime64, np.timedelta64]:
        """Get the last value of this axis.

        Returns:
            numpy.datetime64, numpy.timedelta64: The last value.
        """
        return self.object(super().back(), self.resolution)

    def find_index(self,
                   coordinates: np.ndarray,
                   bounded: bool = False) -> np.ndarray:
        """Given coordinate positions, find what grid elements contains them,
        or is closest to them.

        Args:
            coordinates (numpy.ndararray): Positions in this coordinate system
            bounded (bool, optional): True if you want to obtain the closest
                value to a coordinate outside the axis definition range.

        Returns:
            numpy.ndarray: index of the grid points containing them or -1 if
            the bounded parameter is set to false and if one of the searched
            indexes is out of the definition range of the axis, otherwise the
            index of the closest value of the coordinate is returned.
        """
        return super().find_index(
            coordinates.astype(self.dtype).astype("int64"), bounded)

    def find_indexes(self, coordinates: np.ndarray) -> np.ndarray:
        """For all coordinate positions, search for the axis elements around
        them. This means that for n coordinate ``ix`` of the provided array,
        the method searches the indexes ``i0`` and ``i1`` as fallow:

        .. code::

            self[i0] <= coordinates[ix] <= self[i1]

        The provided coordinates located outside the axis definition range are
        set to ``-1``.

        Args:
            coordinates (numpy.ndarray): Positions in this coordinate system.
        Returns:
            numpy.ndarray: A matrix of shape ``(n, 2)``. The first column of
            the matrix contains the indexes ``i0`` and the second column the
            indexes ``i1`` found.
        """
        return super().find_indexes(
            coordinates.astype(self.dtype).astype("int64"))

    def front(self) -> Union[np.datetime64, np.timedelta64]:
        """Get the first value of this axis.

        Returns:
            numpy.datetime64, numpy.timedelta64: The first value.
        """
        return self.object(super().front(), self.resolution)

    @classmethod
    def _npdate_resolution(cls, dtype) -> str:
        """Gets the numpy date time resolution."""
        match = cls.PATTERN(dtype)
        assert match is not None
        return match.group(1)

    def increment(self) -> np.timedelta64:
        """Get increment value if is_regular().

        Returns
            numpy.timedelta64: Increment value.

        Raises:
            RuntimeError: if this instance does not represent a regular axis.
        """
        return np.timedelta64(super().increment(), self.resolution)

    def max_value(self) -> Union[np.datetime64, np.timedelta64]:
        """Get the maximum value of this axis.

        Returns:
            numpy.datetime64, numpy.timedelta64: The maximum value.
        """
        return self.object(super().max_value(), self.resolution)

    def min_value(self) -> Union[np.datetime64, np.timedelta64]:
        """Get the minimum value of this axis.

        Returns:
            numpy.datetime64, numpy.timedelta64 : The minimum value.
        """
        return self.object(super().min_value(), self.resolution)

    def __repr__(self):
        array = repr(self[:]).split("\n")
        array[1:] = [(" " * 13) + item for item in array[1:]]
        return "TemporalAxis(" + "\n".join(array) + ")"

    def __setstate__(self, state):
        """Restore the state of this object.

        Args:
            state (tuple): State of the object.
        """
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError("invalid state")
        super().__setstate__(state[1])
        self.dtype = state[0]
        self.object = self._object(self.dtype)
        self.resolution = self._npdate_resolution(str(self.dtype))

    def __getstate__(self):
        """Returns the state of this object.

        Returns:
            tuple: The state of this object.
        """
        return (self.dtype, super().__getstate__())

    def __getitem__(self, *args):
        """Get the values of this axis.
        
        Args:
            *args: Variable length argument list.

        Returns:
            numpy.ndarray: Values of this axis.
        """
        result = super().__getitem__(*args)
        if isinstance(result, int):
            return self.object(result, self.resolution)
        return result.astype(self.dtype)

"""
Axis
====
"""
import re
import warnings
import numpy as np
from . import core


class TemporalAxis(core.TemporalAxis):
    """Time axis
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
            values (numpy.ndarray): Dates representing the dates of the time
                axis.

        Raises:
            TypeError: if the array data type is not a datetime64 subtype.

        Examples:

            >>> import datetime
            >>> import numpy as np
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
        """
        if not np.issubdtype(values.dtype, np.dtype("datetime64")):
            raise TypeError("values must be a datetime64 array")
        super().__init__(values.astype("int64"))
        self.dtype = values.dtype
        self.resolution = self._datetime64_resolution(str(self.dtype))

    def safe_cast(self, values: np.ndarray) -> np.ndarray:
        """Convert the dates of the vector in the same unit as the time axis
        defined in this instance.

        Args:
            values (numpy.ndarray): Values to convert

        Return:
            numpy.ndarray: values converted

        Raises:
            UserWarning: if the implicit conversion from the unit of dates
                provided to the unit of the axis, truncates the dates
                (e.g. converting microseconds to seconds).
        """
        if not np.issubdtype(values.dtype, np.dtype("datetime64")):
            raise TypeError("values must be a datetime64 array")
        resolution = self._datetime64_resolution(str(values.dtype))
        source_idx = self.RESOLUTION.index(resolution)
        target_idx = self.RESOLUTION.index(self.resolution)
        if source_idx > target_idx:
            source = self.UNITS[source_idx]
            target = self.UNITS[target_idx]
            warnings.warn(f"implicit conversion turns {source} into {target}",
                          UserWarning)
        return values.astype(f"datetime64[{self.resolution}]").astype("int64")

    def back(self) -> np.datetime64:
        """Get the last value of this axis

        Return:
            numpy.datetime64: The last value
        """
        return np.datetime64(super().back(), self.resolution)

    def find_index(self, coordinates: np.ndarray,
                   bounded: bool = False) -> np.ndarray:
        """Given coordinate positions, find what grid elements contains them,
        or is closest to them.

        Args:
            coordinates (numpy.ndararray): Positions in this coordinate system
            bounded (bool, optional): True if you want to obtain the closest
                value to a coordinate outside the axis definition range.

        Return:
            numpy.ndarray: index of the grid points containing them or -1 if
            the bounded parameter is set to false and if one of the searched
            indexes is out of the definition range of the axis, otherwise the
            index of the closest value of the coordinate is returned.
        """
        return super().find_index(
            coordinates.astype(self.dtype).astype("int64"), bounded)

    def front(self) -> np.datetime64:
        """Get the first value of this axis

        Return:
            numpy.datetime64: The first value
        """
        return np.datetime64(super().front(), self.resolution)

    @classmethod
    def _datetime64_resolution(cls, dtype) -> str:
        """Gets the date time resolution"""
        match = cls.PATTERN(dtype)
        assert match is not None
        return match.group(1)

    def increment(self) -> np.timedelta64:
        """Get increment value if is_regular()

        Returns
            numpy.timedelta64: Increment value

        Raises:
            RuntimeError: if this instance does not represent a regular axis
        """
        return np.timedelta64(super().increment(), self.resolution)

    def max_value(self) -> np.datetime64:
        """Get the maximum value of this axis

        Return:
            numpy.datetime64: The maximum value
        """
        return np.datetime64(super().max_value(), self.resolution)

    def min_value(self) -> np.datetime64:
        """Get the minimum value of this axis

        Return:
            numpy.datetime64: The minimum value
        """
        return np.datetime64(super().min_value(), self.resolution)

    def __repr__(self):
        array = repr(self[:]).split("\n")
        array[1:] = [(" " * 13) + item for item in array[1:]]
        return "TemporalAxis(" + "\n".join(array) + ")"

    def __setstate__(self, state):
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError("invalid state")
        super().__setstate__(state[1])
        self.dtype = state[0]
        self.resolution = self._datetime64_resolution(str(self.dtype))

    def __getstate__(self):
        return (self.dtype, super().__getstate__())

    def __getitem__(self, *args):
        return super().__getitem__(*args).astype(self.dtype)

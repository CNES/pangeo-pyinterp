"""
Axis
====
"""
import re
import numpy as np
from . import core


class TemporalAxis(core.TemporalAxis):
    """Time axis"""
    def __init__(self, values: np.ndarray):
        """
        Create a coordinate axis from values.

        Args:
            values (numpy.ndarray): Dates representing the dates of the time
                axis.

        Raises:
            TypeError: if the array data type is not a datetime64 subtype.
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

        Returns:
            numpy.ndarray: values converted
        """
        return values.astype(f"datetime64[{self.resolution}]").astype("int64")

    def back(self) -> np.datetime64:
        """Get the last value of this axis

        Returns:
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

        Returns:
            numpy.ndarray: index of the grid points containing them or -1 if
            the bounded parameter is set to false and if one of the searched
            indexes is out of the definition range of the axis, otherwise the
            index of the closest value of the coordinate is returned.
        """
        return super().find_index(
            coordinates.astype(self.dtype).astype("int64"), bounded)

    def front(self) -> np.datetime64:
        """Get the first value of this axis

        Returns:
            numpy.datetime64: The first value
        """
        return np.datetime64(super().front(), self.resolution)

    @staticmethod
    def _datetime64_resolution(dtype) -> str:
        """Gets the date time resolution"""
        pattern = re.compile(r"\[([^\]]*)\]")
        match = pattern.search(dtype)
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

        Returns:
            numpy.datetime64: The maximum value
        """
        return np.datetime64(super().max_value(), self.resolution)

    def min_value(self) -> np.datetime64:
        """Get the minimum value of this axis

        Returns:
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

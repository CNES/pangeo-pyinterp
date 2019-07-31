# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
RTree spatial index
-------------------
"""
from typing import Optional, Tuple
import sys
import numpy as np
from . import core
from . import geodetic


class RTree:
    """RTree spatial index for geodetic scalar values

    Args:
        system (pyinterp.geodetic.System, optional): WGS of the
            coordinate system used to transform equatorial spherical positions
            (longitudes, latitudes, altitude) into ECEF coordinates. If not set
            the geodetic system used is WGS-84. Default to ``None``.
        dtype (numpy.dtype, optional): Data type of the instance to create.
    """

    def __init__(self,
                 system: Optional[geodetic.System] = None,
                 dtype: Optional[np.dtype] = np.dtype("float64")):
        if dtype == np.dtype("float64"):
            self._instance = core.RTreeFloat64(system)
        elif dtype == np.dtype("float32"):
            self._instance = core.RTreeFloat32(system)
        else:
            raise ValueError(f"dtype {dtype} not handled by the object")
        self.dtype = dtype

    def bounds(
            self
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Returns the box able to contain all values stored in the container.

        Return:
            tuple: A tuple containing two tuples defining 3 coordinates
            (longitude, latitude, altitude) capable of containing all the
            values stored in the container or None if there are no values
            in the container.
        """
        return self._instance.bounds()

    def clear(self) -> None:
        """Removes all values stored in the container.
        """
        return self._instance.clear()

    def __len__(self):
        return self._instance.__len__()

    def __bool__(self):
        return self._instance.__bool__()

    def packing(self, coordinates: np.ndarray, values: np.ndarray) -> None:
        """The tree is created using packing algorithm (The old data is erased
        before construction.)

        Args:
            coordinates (numpy.ndarray): A matrix ``(n, 2)`` to add points
                defined by their longitudes and latitudes or a matrix
                ``(n, 3)`` to add points defined by their longitudes, latitudes
                and altitudes.
            values (numpy.ndarray): An array of size ``(n)`` containing the
                values associated with the coordinates provided
        """
        self._instance.packing(coordinates, values)

    def insert(self, coordinates: np.ndarray, values: np.ndarray) -> None:
        """Insert new data into the search tree.

        Args:
            coordinates (numpy.ndarray): A matrix ``(n, 2)`` to add points
                defined by their longitudes and latitudes or a matrix
                ``(n, 3)`` to add points defined by their longitudes, latitudes
                and altitudes.
            values (numpy.ndarray): An array of size ``(n)`` containing the
                values associated with the coordinates provided
        """
        self._instance.insert(coordinates, values)

    def query(self,
              coordinates: np.ndarray,
              k: Optional[int] = 4,
              within: Optional[bool] = True,
              num_threads: Optional[int] = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Search for the nearest K nearest neighbors of a given point.

        Args:
            coordinates (numpy.ndarray): A matrix ``(n, 2)`` to search points
                defined by their longitudes and latitudes or a matrix
                ``(n, 3)`` to search points defined by their longitudes,
                latitudes and altitudes.
            k (int, optional): The number of nearest neighbors to be searched.
                Defaults to ``4``.
            within (bool, optional): If true, the method ensures that the
                neighbors found are located within the point of interest.
                Defaults to ``false``.
            num_threads (int, optional): The number of threads to use for the
                computation. If 0 all CPUs are used. If 1 is given, no parallel
                computing code is used at all, which is useful for debugging.
                Defaults to ``0``.
        Return:
            tuple: A tuple containing a matrix describing for each provided
            position, the distance, in meters, between the provided position
            and the found neighbors and a matrix containing the value of the
            different neighbors found for all provided positions.
        """
        return self._instance.query(coordinates, k, within, num_threads)

    def inverse_distance_weighting(
            self,
            coordinates: np.ndarray,
            radius: Optional[float] = sys.float_info.max,
            k: Optional[int] = 4,
            p: Optional[int] = 2,
            within: Optional[bool] = True,
            num_threads: Optional[int] = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolation of the value at the requested position by inverse
        distance weighting method.

        Args:
            coordinates (numpy.ndarray): A matrix ``(n, 2)`` to interpolate
                points defined by their longitudes and latitudes or a matrix
                ``(n, 3)`` to interpolate points defined by their longitudes,
                latitudes and altitudes.
            radius (float, optional): The maximum radius of the search (m).
                Defaults The maximum distance between two points.
            k (int, optional): The number of nearest neighbors to be used for
                calculating the interpolated value. Defaults to ``4``.
            p (float, optional): The power parameters. Defaults to ``2``.
            within (bool, optional): If true, the method ensures that the
                neighbors found are located around the point of interest. In
                other words, this parameter ensures that the calculated values
                will not be extrapolated. Defaults to ``true``.
            num_threads (int, optional): The number of threads to use for the
                computation. If 0 all CPUs are used. If 1 is given, no parallel
                computing code is used at all, which is useful for debugging.
                Defaults to ``0``.
        Return:
            tuple: The interpolated value and the number of neighbors used in
            the calculation.
        """
        return self._instance.inverse_distance_weighting(
            coordinates, radius, k, p, within, num_threads)

    def __getstate__(self) -> Tuple:
        return (self.dtype, self._instance.__getstate__())

    def __setstate__(self, state: Tuple):
        if len(state) != 2:
            raise ValueError("invalid state")
        _class = RTree(None, state[0])
        self.dtype = _class.dtype
        _class._instance.__setstate__(state[1])
        self._instance = _class._instance

# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Binned statistic
----------------
"""
from typing import Optional
import numpy as np
from . import core


class NearestBivariate:
    """
    Discretizes the data into a regular grid (computes a binned approximation)
    using the nearest binning technique.

    Args:
        x (pyinterp.core.Axis) : Definition of the bin edges for the X axis of
            the grid.
        y (pyinterp.core.Axis) : Definition of the bin edges for the Y axis of
            the grid.
        dtype (numpy.dtype, optional): Data type of the instance to create.
    """
    def __init__(self,
                 x: core.Axis,
                 y: core.Axis,
                 dtype: Optional[np.dtype] = np.dtype("float64")):
        if dtype == np.dtype("float64"):
            self._instance = core.binning.NearestBivariateFloat64(x, y)
        elif dtype == np.dtype("float32"):
            self._instance = core.binning.NearestBivariateFloat32(x, y)
        else:
            raise ValueError(f"dtype {dtype} not handled by the object")
        self.dtype = dtype

    @property
    def x(self) -> core.Axis:
        """Gets the bin edges for the X Axis of the grid"""
        return self._instance.x

    @property
    def y(self) -> core.Axis:
        """Gets the bin edges for the Y Axis of the grid"""
        return self._instance.y

    def clear(self) -> None:
        """Clears the data inside each bin."""
        self._instance.clear()

    def __repr__(self):
        result = [
            "<%s.%s>" % (self.__class__.__module__, self.__class__.__name__)
        ]
        result.append("Axis:")
        result.append(f"  x: {self._instance.x}")
        result.append(f"  y: {self._instance.y}")
        return "\n".join(result)

    def push(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        """Push new samples into the defined bins.

        Args:
            x (numpy.ndarray): X coordinates of the samples
            y (numpy.ndarray): Y coordinates of the samples
            z (numpy.ndarray): New samples to push into the defined bins.
        """
        self._instance.push(
            np.asarray(x).flatten(),
            np.asarray(y).flatten(),
            np.asarray(z).flatten())

    def variable(self, statistics: Optional[str] = 'mean') -> np.ndarray:
        """Gets the regular grid containing the calculated statistics.

        Args:
            statistics (str or iterable, optional) : The statistics to compute
                The following statistics are available:

                    * ``count`` : compute the count of points within each bin.
                    * ``kurtosis`` : compute the kurtosis of values for points
                      within each bin.
                    * ``max`` : compute the maximum of values for point within
                      each bin.
                    * ``mean`` : compute the mean of values for points within
                      each bin.
                    * ``median`` : compute the median of values for points
                      within each bin.
                    * ``min`` : compute the minimum of values for points within
                      each bin.
                    * ``skewness`` : compute the skewness within each bin.
                    * ``sum`` : compute the sum of values for points within
                      each bin.
                    * ``variance`` : compute the variance within each bin.

        Return:
            numpy.ndarray: The dataset representing the calculated
            statistical variable.
        """
        try:
            data = getattr(self._instance, statistics)()
            if statistics == 'count':
                data = data.astype(np.int64)
            else:
                data[~np.isfinite(data)] = np.nan
            return data
        except AttributeError:
            raise ValueError(f"The statistical variable {item} is unknown.")

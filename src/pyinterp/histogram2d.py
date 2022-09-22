# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Histogram 2D
------------
"""
from typing import Optional, Union
import copy

import dask.array.core
import numpy

from . import core


class Histogram2D:
    """Group a number of more or less continuous values into a smaller number
    of "bins" located on a grid.

    This class will build for each pixel of the defined grid, a histogram. This
    histogram will be used to compute the statistics.

    Histogram used uses the algorithm described in the paper `A Streaming
    Parallel Decision Tree Algorithm`. Therefore, if the number of observations
    to be taken into account in a pixel exceeds the maximum number of bins, the
    calculated statistics will be an approximate value of the exact statistical
    variable. This algorithm is useful if you want to know the statistical
    distribution per pixel or the value of a quantile, like the median.
    Otherwise, use the :py:class:`pyinterp.Binning2D` class.

    Args:
        x: Definition of the bin centers for the X axis of the grid.
        y: Definition of the bin centers for the Y axis of the grid.
        bin_counts: The number of bins to use. If not set, the number of
            bins is 100.
        dtype: Data type of the instance to create.

    .. note ::

        The axes define the centers of the different cells where the
        statistics will be calculated, as shown in the figure below.

        .. figure:: ../pictures/coordinates.svg
            :align: center
            :width: 50%

        In this example, to calculate the statistics in the different cells
        defined, the coordinates of the axes must be shifted by half a grid
        step, 0.5 in this example.

    .. note ::

        Yael Ben-Haim and Elad Tom-Tov,
        A Streaming Parallel Decision Tree Algorithm,
        Journal of Machine Learning Research, 11, 28, 849-872
        http://jmlr.org/papers/v11/ben-haim10a.html
    """

    def __init__(self,
                 x: core.Axis,
                 y: core.Axis,
                 bin_counts: Optional[int] = None,
                 dtype: Optional[numpy.dtype] = numpy.dtype('float64')):
        if dtype == numpy.dtype('float64'):
            self._instance = core.Histogram2DFloat64(x, y, bin_counts)
        elif dtype == numpy.dtype('float32'):
            self._instance = core.Histogram2DFloat32(x, y, bin_counts)
        else:
            raise ValueError(f'dtype {dtype} not handled by the object')
        self.dtype = dtype

    @property
    def x(self) -> core.Axis:
        """Gets the bin centers for the X Axis of the grid."""
        return self._instance.x

    @property
    def y(self) -> core.Axis:
        """Gets the bin centers for the Y Axis of the grid."""
        return self._instance.y

    def clear(self) -> None:
        """Clears the data inside each bin."""
        self._instance.clear()

    def __repr__(self) -> str:
        """Called by the ``repr()`` built-in function to compute the string
        representation of this instance."""
        result = [f'<{self.__class__.__module__}.{self.__class__.__name__}>']
        result.append('Axis:')
        result.append(f'  x: {self._instance.x}')
        result.append(f'  y: {self._instance.y}')
        return '\n'.join(result)

    def __add__(self, other: 'Histogram2D') -> 'Histogram2D':
        """Overrides the default behavior of the ``+`` operator."""
        if self.dtype != other.dtype:
            raise ValueError('dtype mismatch')
        result = copy.copy(self)
        result._instance += other._instance  # type: ignore
        return result

    def push(self, x: numpy.ndarray, y: numpy.ndarray,
             z: numpy.ndarray) -> None:
        """Push new samples into the defined bins.

        Args:
            x: X coordinates of the samples.
            y: Y coordinates of the samples.
            z: New samples to push into the defined bins.
        """
        x = numpy.asarray(x).ravel()
        y = numpy.asarray(y).ravel()
        z = numpy.asarray(z).ravel()
        self._instance.push(x, y, z)

    def push_delayed(
        self,
        x: Union[numpy.ndarray, dask.array.core.Array],
        y: Union[numpy.ndarray, dask.array.core.Array],
        z: Union[numpy.ndarray, dask.array.core.Array],
    ) -> dask.array.core.Array:
        """Push new samples into the defined bins from dask array.

        Args:
            x: X coordinates of the samples.
            y: Y coordinates of the samples.
            z: New samples to push into the defined bins.
        Returns:
            The calculation graph producing the update of the grid from the
            provided samples. Running the graph will return an instance of this
            class containing the statistics calculated for all processed
            samples.

        .. seealso ::

            :py:meth:`push <pyinterp.Histogram2D.push>`
        """
        x = dask.array.core.asarray(x)
        y = dask.array.core.asarray(y)
        z = dask.array.core.asarray(z)

        def _process_block(x, y, z, x_axis, y_axis, dtype):
            hist2d = Histogram2D(x_axis, y_axis, dtype=dtype)
            hist2d.push(x, y, z)
            return numpy.array([hist2d], dtype='object')

        return dask.array.core.map_blocks(_process_block,
                                          x.ravel(),
                                          y.ravel(),
                                          z.ravel(),
                                          self.x,
                                          self.y,
                                          self.dtype,
                                          dtype='object').sum()

    def variable(self, statistics: str = 'mean', *args) -> numpy.ndarray:
        """Gets the regular grid containing the calculated statistics.

        Args:
            statistics: The statistics to compute
                The following statistics are available:

                    * ``count`` : compute the count of points within each bin.
                    * ``kurtosis`` : compute the kurtosis of values for points
                      within each bin.
                    * ``max`` : compute the maximum of values for point within
                      each bin.
                    * ``mean`` : compute the mean of values for points within
                      each bin.
                    * ``min`` : compute the minimum of values for points within
                      each bin.
                    * ``skewness`` : compute the skewness of values for points
                    * ``variance`` : compute the variance within each bin.
            args: Additional arguments to pass to the statistics
                function. For example, ``quantile`` requires a ``q``
                argument that specifies the quantile to compute.

        Returns:
            The dataset representing the calculated statistical variable.
        """
        try:
            return getattr(self._instance, statistics)(*args)
        except AttributeError as exc:
            raise ValueError(
                f'The statistical variable {statistics} is unknown.') from exc

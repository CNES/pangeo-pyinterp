# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Data binning."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union
import copy

import dask.array.core
import numpy

from . import core, geodetic

if TYPE_CHECKING:
    from .typing import NDArray, NDArray1D, NDArray2D

#: The supported data types for the binning 2D
Binning2DTyped = Union[core.Binning2DFloat64, core.Binning2DFloat32]


class Binning2D:
    """Group continuous values into bins located on a grid.

    Create a 2D binning structure that groups a number of more or less
    continuous values into a smaller number of "bins" located on a grid for
    statistical calculations.

    Args:
        x: Definition of the bin centers for the X axis of the grid.
        y: Definition of the bin centers for the Y axis of the grid.
        wgs: WGS of the coordinate system used to manipulate geographic
            coordinates. If this parameter is not set, the handled
            coordinates will be considered as Cartesian coordinates.
            Otherwise, ``x`` and ``y`` are considered to represents the
            longitudes and latitudes.
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

    """

    def __init__(self,
                 x: core.Axis,
                 y: core.Axis,
                 wgs: geodetic.Spheroid | None = None,
                 dtype: numpy.dtype | None = None) -> None:
        """Initialize a Binning2D instance."""
        dtype = dtype or numpy.dtype('float64')
        if dtype == numpy.dtype('float64'):
            self._instance: Binning2DTyped = core.Binning2DFloat64(x, y, wgs)
        elif dtype == numpy.dtype('float32'):
            self._instance = core.Binning2DFloat32(x, y, wgs)
        else:
            raise ValueError(f'dtype {dtype} not handled by the object')
        self.dtype = dtype

    @property
    def x(self) -> core.Axis:
        """Get the bin centers for the X Axis of the grid."""
        return self._instance.x

    @property
    def y(self) -> core.Axis:
        """Get the bin centers for the Y Axis of the grid."""
        return self._instance.y

    @property
    def wgs(self) -> core.geodetic.Spheroid | None:
        """Get the geodetic system handled by the grid."""
        return self._instance.wgs

    def clear(self) -> None:
        """Clear the data inside each bin."""
        self._instance.clear()

    def __repr__(self) -> str:
        """Compute the string representation of this instance.

        Called by the ``repr()`` built-in function.
        """
        result = [f'<{self.__class__.__module__}.{self.__class__.__name__}>']
        result.append('Axis:')
        result.append(f'  x: {self._instance.x}')
        result.append(f'  y: {self._instance.y}')
        return '\n'.join(result)

    def __add__(self, other: Binning2D) -> Binning2D:
        """Override the default behavior of the ``+`` operator."""
        result = copy.copy(self)
        if type(result._instance) != type(other._instance):  # noqa: E721
            raise TypeError('Binning2D instance must be of the same type')
        result._instance += other._instance  # type: ignore[operator]
        return result

    def push(self,
             x: NDArray,
             y: NDArray,
             z: NDArray,
             simple: bool = True) -> None:
        r"""Push new samples into the defined bins.

        Args:
            x: X coordinates of the samples
            y: Y coordinates of the samples
            z: New samples to push into the defined bins.
            simple: If true, a simple binning 2D is used otherwise a linear
                binning 2d is applied. See the full description of the algorithm
                below.

        .. _bilinear_binning:

        The figure below is a graphical presentation of how a sample data
        point :math:`x` distributes its weight to neighboring grid points.

        :math:`A` is the area of the grid cell. :math:`\\alpha`,
        :math:`\\beta`, :math:`\\gamma` and :math:`\\delta` are the areas of
        the different sub-rectangles. :math:`g_{00}`, :math:`g_{01}`,
        :math:`g_{10}` and :math:`g_{11}` are the grid points identified around
        point :math:`x`. :math:`w_{00}`, :math:`w_{01}`, :math:`w_{10}` and
        :math:`w_{11}` are the weights associated with the grid points.

        .. figure:: ../pictures/binning_2d.svg
            :align: center

        For simple binning, the point :math:`x` gives all its weight to its
        nearest grid point. In this example, the lower left grid point takes
        the weight equal to 1, that is :math:`w_{00}=1`.

        In the case of linear binning, the contribution from :math:`x` is
        distributed among each of the four surrounding grid points according to
        the areas of the opposite sub-rectangle induced by the position of the
        point.

        .. seealso::

            Peter Hall, M.P. Wand,
            On the Accuracy of Binned Kernel Density Estimators,
            Journal of Multivariate Analysis,
            Volume 56, Issue 2,
            1996,
            Pages 165-184,

        """
        x = numpy.asarray(x).ravel()
        y = numpy.asarray(y).ravel()
        z = numpy.asarray(z).ravel()
        self._instance.push(x, y, z, simple)

    def push_delayed(self,
                     x: NDArray | dask.array.core.Array,
                     y: NDArray | dask.array.core.Array,
                     z: NDArray | dask.array.core.Array,
                     simple: bool = True) -> dask.array.core.Array:
        """Push new samples into the defined bins from dask array.

        Args:
            x: X coordinates of the samples.
            y: Y coordinates of the samples.
            z: New samples to push into the defined bins.
            simple: If true, a simple binning 2D is used
                otherwise a linear binning 2d is applied. See the full
                description of the algorithm :ref:`here <bilinear_binning>`.

        Returns:
            The calculation graph producing the update of the grid from the
            provided samples. Running the graph will return an instance of this
            class containing the statistics calculated for all processed
            samples.

        .. seealso ::

            :py:meth:`push <pyinterp.Binning2D.push>`

        """
        x = dask.array.core.asarray(x)
        y = dask.array.core.asarray(y)
        z = dask.array.core.asarray(z)

        def _process_block(
            x: NDArray1D,
            y: NDArray1D,
            z: NDArray1D,
            x_axis: core.Axis,
            y_axis: core.Axis,
            wgs: geodetic.Spheroid | None,
            simple: bool,
        ) -> numpy.ndarray:
            binning = Binning2D(x_axis, y_axis, wgs)
            binning.push(x, y, z, simple)
            return numpy.array([binning], dtype='object')

        return dask.array.core.map_blocks(_process_block,
                                          x.ravel(),
                                          y.ravel(),
                                          z.ravel(),
                                          self.x,
                                          self.y,
                                          self.wgs,
                                          simple,
                                          dtype='object').sum()

    def variable(self, statistics: str = 'mean') -> NDArray2D:
        """Get the regular grid containing the calculated statistics.

        Return the grid with the requested statistical variable computed from
        the binned data.

        Args:
            statistics: The statistics to compute. The following statistics are
                available:

                    * ``count`` : compute the count of points within each bin.
                    * ``kurtosis`` : compute the kurtosis of values for points
                      within each bin.
                    * ``max`` : compute the maximum of values for point within
                      each bin.
                    * ``mean`` : compute the mean of values for points within
                      each bin.
                    * ``min`` : compute the minimum of values for points within
                      each bin.
                    * ``skewness`` : compute the skewness within each bin.
                    * ``sum`` : compute the sum of values for points within
                      each bin.
                    * ``sum_of_weights`` : compute the sum of weights within
                      each bin.
                    * ``variance`` : compute the variance within each bin.

        Returns:
            The dataset representing the calculated statistical variable.

        """
        try:
            return getattr(self._instance, statistics)()
        except AttributeError as exc:
            raise ValueError(
                f'The statistical variable {statistics} is unknown.') from exc


class Binning1D:
    """Group continuous values into bins located on a vector.

    Create a 1D binning structure that groups a number of more or less
    continuous values into a smaller number of "bins" located on a vector for
    statistical calculations.

    Args:
        x: Definition of the bin centers for the X axis of the grid.
        range: The lower and upper range of the bins. If not provided, range
            is simply ``(x.min_value(), x.max_value())``.
        dtype: Data type of the instance to create.

    .. note ::

        The axe define the centers of the different bins where the
        statistics will be calculated.

    """

    def __init__(self,
                 x: core.Axis,
                 range: tuple[float, float] | None = None,
                 dtype: numpy.dtype | None = None) -> None:
        """Initialize a Binning1D instance."""
        dtype = dtype or numpy.dtype('float64')
        if dtype == numpy.dtype('float64'):
            self._instance: (core.Binning1DFloat64
                             | core.Binning1DFloat32) = core.Binning1DFloat64(
                                 x, range)
        elif dtype == numpy.dtype('float32'):
            self._instance = core.Binning1DFloat32(x, range)
        else:
            raise ValueError(f'dtype {dtype} not handled by the object')
        self.dtype = dtype

    @property
    def x(self) -> core.Axis:
        """Get the bin centers for the X Axis of the grid."""
        return self._instance.x

    def range(self) -> tuple[float, float]:
        """Get the lower and upper range of the bins."""
        return self._instance.range()

    def clear(self) -> None:
        """Clear the data inside each bin."""
        self._instance.clear()

    def __repr__(self) -> str:
        """Compute the string representation of this instance.

        Called by the ``repr()`` built-in function.
        """
        result = [f'<{self.__class__.__module__}{self.__class__.__name__}>']
        result.append('Axis:')
        result.append(f'  {self._instance.x}')
        result.append('Range:')
        result.append(f'  {self._instance.range()}')
        return '\n'.join(result)

    def __add__(self, other: Binning1D) -> Binning1D:
        """Override the default behavior of the ``+`` operator."""
        result = copy.copy(self)
        if type(result._instance) != type(other._instance):  # noqa: E721
            raise TypeError('Binning1D instance must be of the same type')
        result._instance += other._instance  # type: ignore[operator]
        return result

    def push(
        self,
        x: NDArray,
        z: NDArray,
        weights: NDArray | None = None,
    ) -> None:
        """Push new samples into the defined bins.

        Args:
            x: X coordinates of the samples
            z: New samples to push into the defined bins.
            weights: An array of weights, of the same shape as ``z``. Each
                value in a only contributes its associated weight towards the
                bin count (instead of 1).

        """
        x = numpy.asarray(x).ravel()
        z = numpy.asarray(z).ravel()
        self._instance.push(x, z, weights)

    def push_delayed(
        self,
        x: NDArray | dask.array.core.Array,
        z: NDArray | dask.array.core.Array,
        weights: NDArray | dask.array.core.Array | None = None,
    ) -> dask.array.core.Array:
        """Push new samples into the defined bins from dask array.

        Args:
            x: X coordinates of the samples.
            z: New samples to push into the defined bins.
            weights: An array of weights, of the same shape as ``z``. Each
                value in a only contributes its associated weight towards the
                bin count (instead of 1).

        Returns:
            The calculation graph producing the update of the vector from the
            provided samples. Running the graph will return an instance of this
            class containing the statistics calculated for all processed
            samples.

        .. seealso ::

            :py:meth:`push <pyinterp.Binning1D.push>`

        """
        x = dask.array.core.asarray(x)
        z = dask.array.core.asarray(z)
        if weights is not None:
            weights = dask.array.core.asarray(weights).ravel()

        def _process_block(
            x: NDArray1D,
            z: NDArray1D,
            weights: NDArray1D | None,
            x_axis: core.Axis,
        ) -> numpy.ndarray:
            binning = Binning1D(x_axis)
            binning.push(x, z, weights)
            return numpy.array([binning], dtype='object')

        return dask.array.core.map_blocks(_process_block,
                                          x.ravel(),
                                          z.ravel(),
                                          weights,
                                          self.x,
                                          dtype='object').sum()

    def variable(self, statistics: str = 'mean') -> NDArray1D:
        """Get the regular grid containing the calculated statistics.

        Return the grid with the requested statistical variable computed from
        the binned data.

        Args:
            statistics: The statistics to compute.

        Returns:
            numpy.ndarray: The dataset representing the calculated
            statistical variable.

        .. seealso ::

            The :py:meth:`pyinterp.Binning2D.variable` method describes the
            accessible statistical variables.

        """
        try:
            return getattr(self._instance, statistics)()
        except AttributeError as exc:
            raise ValueError(
                f'The statistical variable {statistics} is unknown.') from exc

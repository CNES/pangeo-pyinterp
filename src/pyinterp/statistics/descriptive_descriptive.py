# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Descriptive statistics
----------------------
"""
from typing import Any, Iterable, Optional, Union
import copy

import dask.array.core
import numpy

from .. import core


def _delayed(
    attr: str,
    values: dask.array.core.Array,
    weights: Optional[dask.array.core.Array] = None,
    axis: Optional[Iterable[int]] = None,
) -> Union[core.DescriptiveStatisticsFloat64,
           core.DescriptiveStatisticsFloat32]:
    """Calculate the descriptive statistics of a dask array."""
    if weights is not None and values.shape != weights.shape:
        raise ValueError('values and weights must have the same shape')

    def _process_block(attr, x, w, axis):
        instance = getattr(core, attr)(values=x, weights=w, axis=axis)
        return numpy.array([instance], dtype='object')

    drop_axis = list(range(values.ndim))[1:]

    return dask.array.core.map_blocks(
        _process_block,
        attr,
        values,
        weights,
        axis,
        drop_axis=drop_axis,
        dtype='object').sum().compute()  # type: ignore


class DescriptiveStatistics:
    """Univariate descriptive statistics.

    Calculates the incremental descriptive statistics from the provided values.
    The calculation of the statistics is done when the constructor is invoked.
    Different methods allow to extract the calculated statistics.

    Args:
        values: Array containing numbers whose statistics are desired.

            . note::

                NaNs are automatically ignored.

        weights: An array of weights associated with the values. If not
            provided, all values are assumed to have equal weight.
        axis: Axis or axes along which to compute the statistics. If not
            provided, the statistics are computed over the flattened array.
        dtype: Data type of the returned array. By default, the data type is
            ``numpy.float64``.

    . seealso::

        Pébay, P., Terriberry, T.B., Kolla, H. et al.
        Numerically stable, scalable formulas for parallel and online
        computation of higher-order multivariate central moments
        with arbitrary weights.
        Comput Stat 31, 1305–1325,
        2016,
        https://doi.org/10.1007/s00180-015-0637-z
    """

    def __init__(self,
                 values: Union[dask.array.core.Array, numpy.ndarray],
                 weights: Optional[Union[dask.array.core.Array,
                                         numpy.ndarray]] = None,
                 axis: Optional[Union[int, Iterable[int]]] = None,
                 dtype: Optional[numpy.dtype] = None) -> None:
        if isinstance(axis, int):
            axis = (axis, )
        dtype = dtype or numpy.dtype('float64')
        if dtype == numpy.dtype('float64'):
            attr = 'DescriptiveStatisticsFloat64'
        elif dtype == numpy.dtype('float32'):
            attr = 'DescriptiveStatisticsFloat32'
        else:
            raise ValueError(f'dtype {dtype} not handled by the object')
        if isinstance(values, dask.array.core.Array) or isinstance(
                weights, dask.array.core.Array):
            self._instance = _delayed(
                attr, dask.array.core.asarray(values),
                dask.array.core.asarray(weights)
                if weights is not None else None, axis)
        else:
            self._instance: Union[core.DescriptiveStatisticsFloat64,
                                  core.DescriptiveStatisticsFloat32] = getattr(
                                      core, attr)(values, weights, axis)

    def copy(self) -> 'DescriptiveStatistics':
        """Creates a copy of the current descriptive statistics container.

        Returns:
            Returns a copy of the current descriptive statistics container.
        """
        cls = type(self)
        result = getattr(cls, '__new__')(cls)
        result._instance = self._instance.__copy__()
        return result

    def __iadd__(self, other: Any) -> 'DescriptiveStatistics':
        """Adds a new descriptive statistics container to the current one.

        Returns:
            Returns itself.
        """
        if isinstance(other, DescriptiveStatistics):
            if type(self._instance) != type(other._instance):  # noqa: E721
                raise TypeError(
                    'Descriptive statistics must have the same type')
            self._instance += other._instance  # type: ignore
            return self
        raise TypeError('unsupported operand type(s) for +='
                        f": '{type(self)}' and '{type(other)}'")

    def __add__(self, other: Any) -> 'DescriptiveStatistics':
        """Adds a new descriptive statistics container to the current one.

        Returns:
            DescriptiveStatistics: Returns a new descriptive statistics
                container.
        """
        if isinstance(other, DescriptiveStatistics):
            if type(self._instance) != type(other._instance):  # noqa: E721
                raise TypeError(
                    'Descriptive statistics must have the same type')
            result = copy.copy(self)
            result += other
            return result
        raise TypeError('unsupported operand type(s) for +='
                        f": '{type(self)}' and '{type(other)}'")

    def count(self) -> numpy.ndarray:
        """Returns the count of samples.

        Returns:
            The count of samples.
        """
        return self._instance.count()

    def kurtosis(self) -> numpy.ndarray:
        """Returns the kurtosis of samples.

        Returns:
            The kurtosis of samples.
        """
        return self._instance.kurtosis()

    def max(self) -> numpy.ndarray:
        """Returns the maximum of samples.

        Returns:
            The maximum of samples.
        """
        return self._instance.max()

    def mean(self) -> numpy.ndarray:
        """Returns the mean of samples.

        Returns:
            The mean of samples.
        """
        return self._instance.mean()

    def min(self) -> numpy.ndarray:
        """Returns the minimum of samples.

        Returns:
            The minimum of samples.
        """
        return self._instance.min()

    def skewness(self) -> numpy.ndarray:
        """Returns the skewness of samples.

        Returns:
            The skewness of samples.
        """
        return self._instance.skewness()

    def sum(self) -> numpy.ndarray:
        """Returns the sum of samples.

        Returns:
            The sum of samples.
        """
        return self._instance.sum()

    def sum_of_weights(self) -> numpy.ndarray:
        """Returns the sum of weights.

        Returns:
            The sum of weights.
        """
        return self._instance.sum_of_weights()

    def var(self, ddof: int = 0) -> numpy.ndarray:
        """Returns the variance of samples.

        Args:
            ddof: Means Delta Degrees of Freedom. The divisor used in
                calculations is N - ddof, where N represents the number of
                elements. By default ddof is zero.

        Returns:
            The variance of samples.
        """
        return self._instance.variance(ddof)

    def std(self, ddof: int = 0) -> numpy.ndarray:
        """Returns the standard deviation of samples.

        Args:
            ddof: Means Delta Degrees of Freedom. The divisor used in
                calculations is N - ddof, where N represents the number of
                elements. By default ddof is zero.

        Returns:
            The standard deviation of samples.
        """
        return numpy.sqrt(self.var(ddof=ddof))

    def array(self) -> numpy.ndarray:
        """Returns the different statistical variables calculated in a numpy
        structured table with the following fields:

        - ``count``: Number of samples.
        - ``kurtosis``: Kurtosis of samples.
        - ``max``: Maximum of samples.
        - ``mean``: Mean of samples.
        - ``min``: Minimum of samples.
        - ``skewness``: Skewness of samples.
        - ``sum_of_weights``: Sum of weights.
        - ``sum``: Sum of samples.
        - ``var``: Variance of samples (ddof is equal to zero).

        Returns:
            The different statistical variables calculated in a numpy
            structured table.
        """
        dreal = 'f8' if isinstance(self._instance,
                                   core.DescriptiveStatisticsFloat64) else 'f4'
        dtype = [('count', 'u8'), ('kurtosis', dreal), ('max', dreal),
                 ('mean', dreal), ('min', dreal), ('skewness', dreal),
                 ('sum_of_weights', dreal), ('sum', dreal), ('var', dreal)]
        fields = [item[0] for item in dtype]
        field = fields.pop()
        buffer = getattr(self, field)()
        result = numpy.empty(buffer.shape, dtype=dtype)
        result[field] = buffer
        for field in fields:
            result[field] = getattr(self, field)()
        return result

    def __str__(self) -> str:
        array, shape = self._instance.__getstate__()
        return str(array.reshape(shape))

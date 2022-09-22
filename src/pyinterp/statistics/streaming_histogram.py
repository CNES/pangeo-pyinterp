# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Calculate statistics of a stream of values
------------------------------------------
"""
from typing import Any, Iterable, Optional, Union

import dask.array.core
import numpy

from .. import core


def _delayed(
    attr: str,
    values: dask.array.core.Array,
    weights: Optional[dask.array.core.Array] = None,
    axis: Optional[Iterable[int]] = None,
    bin_count: Optional[int] = None,
) -> Union[core.StreamingHistogramFloat64, core.StreamingHistogramFloat32]:
    """Calculate the descriptive statistics of a dask array."""
    if weights is not None and values.shape != weights.shape:
        raise ValueError('values and weights must have the same shape')

    def _process_block(attr, x, w, axis, bin_count):
        instance = getattr(core, attr)(values=x,
                                       weights=w,
                                       axis=axis,
                                       bin_count=bin_count)
        return numpy.array([instance], dtype='object')

    drop_axis = list(range(values.ndim))[1:]

    return dask.array.core.map_blocks(
        _process_block,
        attr,
        values,
        weights,
        axis,
        bin_count,
        drop_axis=drop_axis,
        dtype='object').sum().compute()  # type: ignore


class StreamingHistogram:
    """Streaming histogram.

    The bins in the histogram have no predefined size, so that as values are
    pushed into the histogram, bins are added and merged as soon as their
    numbers exceed the maximum allowed capacity. A particularly interesting
    feature of streaming histograms is that they can be used to approximate
    quantiles without sorting (or even storing) values individually. The
    histograms can be constructed independently and merged, making them usable
    with Dask.

    Args:
        values: Array containing numbers whose statistics are desired.

            .. note::

                NaNs are automatically ignored.

        weights: An array of weights associated with the values. If not
            provided, all values are assumed to have equal weight.
        axis: Axis or axes along which to compute the statistics. If not
            provided, the statistics are computed over the flattened array.
        bin_count: The maximum number of bins to use in the histogram. If
            the number of bins exceeds the number of values, the histogram
            will be trimmed. Default is ``None``, which will set the number
            of bins to 100.
        dtype: Data type of the returned array. By default, the data type
            is numpy.float64.

    .. seealso::

        Yael Ben-Haim and Elad Tom-Tov,
        A Streaming Parallel Decision Tree Algorithm,
        Journal of Machine Learning Research, 11, 28, 849-872,
        http://jmlr.org/papers/v11/ben-haim10a.html

    .. note::

        If you do not want to estimate the quantiles of the dataset, use the
        class :py:class:`DescriptiveStatistics <pyinterp.DescriptiveStatistics>`
        which will give you more accurate results.
    """

    def __init__(self,
                 values: Union[dask.array.core.Array, numpy.ndarray],
                 weights: Optional[Union[dask.array.core.Array,
                                         numpy.ndarray]] = None,
                 axis: Optional[Union[int, Iterable[int]]] = None,
                 bin_count: Optional[int] = None,
                 dtype: Optional[numpy.dtype] = None) -> None:
        if isinstance(axis, int):
            axis = (axis, )
        dtype = dtype or numpy.dtype('float64')
        if dtype == numpy.dtype('float64'):
            attr = 'StreamingHistogramFloat64'
        elif dtype == numpy.dtype('float32'):
            attr = 'StreamingHistogramFloat32'
        else:
            raise ValueError(f'dtype {dtype} not handled by the object')
        if isinstance(values, dask.array.core.Array) or isinstance(
                weights, dask.array.core.Array):
            self._instance = _delayed(attr,
                                      dask.array.core.asarray(values),
                                      weights=dask.array.core.asarray(weights)
                                      if weights is not None else None,
                                      axis=axis,
                                      bin_count=bin_count)
        else:
            self._instance: Union[core.StreamingHistogramFloat64,
                                  core.StreamingHistogramFloat32] = getattr(
                                      core, attr)(values,
                                                  weights=weights,
                                                  axis=axis,
                                                  bin_count=bin_count)

    def __iadd__(self, other: Any) -> 'StreamingHistogram':
        """Adds a new histogram to the current one.

        Args:
            The histogram to add to the current one.

        Returns:
            itself.
        """
        if isinstance(other, StreamingHistogram):
            if type(self._instance) != type(other._instance):  # noqa: E721
                raise TypeError('StreamingHistogram types must match')
            self._instance += other._instance  # type: ignore
        else:
            raise TypeError('unsupported operand type(s) for +='
                            f": '{type(self)}' and '{type(other)}'")
        return self

    def bins(self) -> numpy.ndarray:
        """Returns the histogram bins.

        Returns:
            The histogram bins.
        """
        return self._instance.bins()

    def size(self) -> numpy.ndarray:
        """Returns the number of bins allocated to calculate the histogram.

        If :py:meth:`size() <pyinterp.StreamingHistogram.size>` is equal to
        :py:meth:`count() <pyinterp.StreamingHistogram.count>` then the
        histogram used to calculate the statistics is un-compressed. Otherwise,
        the histogram is compressed, which means that the calculated statistical
        quantities are an approximation of the statistical variables.

        Returns:
            Number of bins allocated to calculate the histogram.
        """
        return self._instance.size()

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

    def sum_of_weights(self) -> numpy.ndarray:
        """Returns the sum of weights.

        Returns:
            The sum of weights.
        """
        return self._instance.sum_of_weights()

    def var(self) -> numpy.ndarray:
        """Returns the variance of samples.

        Returns:
            The variance of samples.
        """
        return self._instance.variance()

    def std(self) -> numpy.ndarray:
        """Returns the standard deviation of samples.

        Returns:
            The standard deviation of samples.
        """
        return numpy.sqrt(self.var())

    def quantile(self, q: float = 0.5) -> numpy.ndarray:
        """Returns the q quantile of samples.

        Args:
            q (float): Quantile to compute. Default is ``0.5`` (median).

        Returns:
            The q quantile of samples.
        """
        return self._instance.quantile(q)

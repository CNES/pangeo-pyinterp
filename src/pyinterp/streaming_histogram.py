from typing import Any, Iterable, Optional, Union
import dask.array as da
import numpy as np
from . import core


def delayed(
    attr: str,
    values: da.Array,
    weights: Optional[da.Array] = None,
    axis: Optional[Iterable[int]] = None,
    bin_count: Optional[int] = None,
) -> Union[core.StreamingHistogramFloat64,
           core.StreamingHistogramFloat32]:
    """Calculate the descriptive statistics of a dask array."""
    if weights is not None and values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")

    def _process_block(attr, x, w, axis):
        instance = getattr(core, attr)(values=x,
                                       weights=w,
                                       axis=axis,
                                       bin_count=bin_count)
        return np.array([instance], dtype="object")

    drop_axis = list(range(values.ndim))[1:]

    return da.map_blocks(_process_block,
                         attr,
                         values,
                         weights,
                         axis,
                         bin_count,
                         drop_axis=drop_axis,
                         dtype="object").sum().compute()  # type: ignore


class StreamingHistogram:
    """
    Streaming histogram.

    Args:
        values (numpy.ndarray, dask.Array): Array containing numbers whose
            statistics are desired.

            .. note::

                NaNs are automatically ignored.

        weights (numpy.ndarray, dask.Array, optional): An array of weights
            associated with the values. If not provided, all values are assumed
            to have equal weight.
        axis (iterable, optional): Axis or axes along which to compute the
            statistics. If not provided, the statistics are computed over the
            flattened array.
        bin_count (int, optional): The number of bins to use in the histogram.
        dtype (numpy.dtype, optional): Data type of the returned array. By
            default, the data type is numpy.float64.
    """
    def __init__(self,
                 values: Union[da.Array, np.ndarray],
                 weights: Optional[Union[da.Array, np.ndarray]] = None,
                 axis: Optional[Iterable[int]] = None,
                 bin_count: Optional[int] = None,
                 dtype: Optional[np.dtype] = None) -> None:
        dtype = dtype or np.dtype("float64")
        if dtype == np.dtype("float64"):
            attr = f"StreamingHistogramFloat64"
        elif dtype == np.dtype("float32"):
            attr = f"StreamingHistogramFloat32"
        else:
            raise ValueError(f"dtype {dtype} not handled by the object")
        if isinstance(values, da.Array) or isinstance(weights, da.Array):
            self._instance = delayed(
                attr, da.asarray(values),
                da.asarray(weights) if weights is not None else None, axis)
        else:
            self._instance: Union[core.StreamingHistogramFloat64,
                                  core.StreamingHistogramFloat32] = getattr(
                                      core, attr)(values,
                                                  weights=weights,
                                                  axis=axis,
                                                  bin_count=bin_count)

    def __iadd__(self, other: Any) -> "StreamingHistogram":
        self._instance += other
        return self

    def size(self) -> np.ndarray:
        """
        Returns the number of bins allocated to calculate the histogram.

        If size() is equal to count() then the histrogram used to calculate
        the statistics is uncompressed. Otherwise, the histogram is compressed.

        Returns:
            numpy.ndarray: Returns the number of bins allocated to calculate
            the histogram.
        """
        return self._instance.size()

    def count(self) -> np.ndarray:
        """
        Returns the count of samples.
        
        Returns:
            numpy.ndarray: Returns the count of samples.
        """
        return self._instance.count()

    def kurtosis(self) -> np.ndarray:
        """
        Returns the kurtosis of samples.
        
        Returns:
            numpy.ndarray: Returns the kurtosis of samples.
        """
        return self._instance.kurtosis()

    def max(self) -> np.ndarray:
        """
        Returns the maximum of samples.
        
        Returns:
            numpy.ndarray: Returns the maximum of samples.
        """
        return self._instance.max()

    def mean(self) -> np.ndarray:
        """
        Returns the mean of samples.
        
        Returns:
            numpy.ndarray: Returns the mean of samples.
        """
        return self._instance.mean()

    def min(self) -> np.ndarray:
        """
        Returns the minimum of samples.
        
        Returns:
            numpy.ndarray: Returns the minimum of samples.
        """
        return self._instance.min()

    def skewness(self) -> np.ndarray:
        """
        Returns the skewness of samples.
        
        Returns:
            numpy.ndarray: Returns the skewness of samples.
        """
        return self._instance.skewness()

    def sum_of_weights(self) -> np.ndarray:
        """
        Returns the sum of weights.
        
        Returns:
            numpy.ndarray: Returns the sum of weights.
        """
        return self._instance.sum_of_weights()

    def var(self) -> np.ndarray:
        """
        Returns the variance of samples.
        
        Returns:
            numpy.ndarray: Returns the variance of samples.
        """
        return self._instance.variance()

    def std(self) -> np.ndarray:
        """
        Returns the standard deviation of samples.

        Returns:
            numpy.ndarray: Returns the standard deviation of samples.
        """
        return np.sqrt(self.var())

    def quantile(self, q: float) -> np.ndarray:
        """
        Returns the q quantile of samples.

        Args:
            q (float): Quantile to compute.

        Returns:
            numpy.ndarray: Returns the q quantile of samples.
        """
        return self._instance.quantile(q)

    def __str__(self) -> str:
        array, shape = self._instance.__getstate__()
        return str(array.reshape(shape))

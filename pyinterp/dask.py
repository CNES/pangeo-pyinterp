# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Distributed computation support for statistics using Dask.

This module provides functions to compute statistics on dask arrays
using the pyinterp statistics classes. Dask is an optional dependency.

Example usage:

    >>> import dask.array as da
    >>> import numpy as np
    >>> import pyinterp
    >>> import pyinterp.dask as dask_stats
    >>>
    >>> # Create a dask array
    >>> x = da.random.random((10000,), chunks=1000)
    >>>
    >>> # Compute descriptive statistics
    >>> stats = dask_stats.descriptive_statistics(x)
    >>> print(stats.mean())
    >>>
    >>> # Compute quantiles using TDigest
    >>> digest = dask_stats.tdigest(x)
    >>> print(digest.quantile(0.5))
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from . import core


if TYPE_CHECKING:
    import dask.array

__all__ = [
    "binning1d",
    "binning2d",
    "descriptive_statistics",
    "histogram2d",
    "tdigest",
]


def _check_dask_available() -> None:
    """Check that dask is available."""
    try:
        import dask.array  # noqa: F401, PLC0415
    except ImportError as exc:
        msg = (
            "dask is required for distributed computation. "
            "Install it with: pip install dask[array]"
        )
        raise ImportError(msg) from exc


def _validate_dask_array(
    arr: object,
    name: str,
) -> dask.array.Array:
    """Validate that input is a dask array.

    Args:
        arr: Input array to validate.
        name: Name of the parameter for error messages.

    Returns:
        The validated dask array.

    Raises:
        TypeError: If the input is not a dask array.

    """
    import dask.array as da  # noqa: PLC0415 (to avoid import issues)

    if not isinstance(arr, da.Array):
        msg = f"{name} must be a dask array, got {type(arr).__name__}"
        raise TypeError(msg)
    return arr


def _validate_shapes_match(
    values: dask.array.Array,
    weights: dask.array.Array | None,
) -> None:
    """Validate that values and weights have matching shapes.

    Args:
        values: Values array.
        weights: Optional weights array.

    Raises:
        ValueError: If shapes don't match.

    """
    if weights is not None and values.shape != weights.shape:
        msg = (
            f"values and weights must have the same shape, "
            f"got {values.shape} and {weights.shape}"
        )
        raise ValueError(msg)


def descriptive_statistics(
    values: dask.array.Array,
    weights: dask.array.Array | None = None,
    axis: list[int] | None = None,
    *,
    dtype: str | type | np.dtype | None = None,
) -> core.DescriptiveStatisticsHolder:
    """Compute descriptive statistics on a dask array.

    This function computes statistics (mean, variance, skewness, kurtosis,
    etc.) on a dask array by processing each block independently and then
    merging the results.

    Args:
        values: Input dask array of values.
        weights: Optional dask array of weights with the same shape as values.
        axis: Axis or axes along which to compute statistics. If None,
            statistics are computed over all axes.
        dtype: Data type for computation. Can be "float32", "float64",
            np.float32, np.float64, or None (defaults to float64).

    Returns:
        A DescriptiveStatistics instance containing the computed statistics.

    Raises:
        ImportError: If dask is not installed.
        TypeError: If inputs are not dask arrays.
        ValueError: If values and weights have different shapes.

    Example:
        >>> import dask.array as da
        >>> import pyinterp.dask as dask_stats
        >>> values = da.random.random((10000,), chunks=1000)
        >>> stats = dask_stats.descriptive_statistics(values)
        >>> print(f"Mean: {stats.mean():.4f}")
        >>> print(f"Std: {np.sqrt(stats.variance()):.4f}")

    """
    _check_dask_available()
    import dask.array as da  # noqa: PLC0415 (to avoid import issues)

    values = _validate_dask_array(values, "values")
    if weights is not None:
        weights = _validate_dask_array(weights, "weights")
        _validate_shapes_match(values, weights)

    def _process_block(
        values_block: np.ndarray,
        weights_block: np.ndarray | None,
        axis: list[int] | None,
        dtype: str | type | np.dtype | None,
        block_id: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Process a single block and return statistics as object array."""
        stats = core.DescriptiveStatistics(
            values_block,
            weights=weights_block,
            axis=axis,
            dtype=dtype,
        )
        result = np.empty((1,), dtype=object)
        result[0] = stats
        return result

    # Create dummy weights if needed to enable block alignment
    if weights is None:
        blocks = da.map_blocks(
            _process_block,
            values,
            None,
            axis,
            dtype,
            dtype=object,
            drop_axis=list(range(values.ndim)),
            new_axis=0,
        )
    else:
        blocks = da.map_blocks(
            _process_block,
            values,
            weights,
            axis,
            dtype,
            dtype=object,
            drop_axis=list(range(values.ndim)),
            new_axis=0,
        )

    # Compute all blocks and merge
    results = blocks.compute()

    # Merge all results using in-place addition
    merged = results[0]
    for item in results[1:]:
        merged += item

    return merged


def tdigest(
    values: dask.array.Array,
    weights: dask.array.Array | None = None,
    axis: list[int] | None = None,
    compression: int = 100,
    *,
    dtype: str | type | np.dtype | None = None,
) -> core.TDigestHolder:
    """Compute quantile estimates on a dask array using T-Digest.

    This function uses the T-Digest algorithm to compute approximate quantiles
    on a dask array by processing each block independently and then merging
    the results.

    Args:
        values: Input dask array of values.
        weights: Optional dask array of weights with the same shape as values.
        axis: Axis or axes along which to compute quantiles. If None,
            quantiles are computed over all axes.
        compression: T-Digest compression parameter. Higher values give
            more accurate results but use more memory. Default is 100.
        dtype: Data type for computation. Can be "float32", "float64",
            np.float32, np.float64, or None (defaults to float64).

    Returns:
        A TDigest instance that can be used to compute quantiles.

    Raises:
        ImportError: If dask is not installed.
        TypeError: If inputs are not dask arrays.
        ValueError: If values and weights have different shapes.

    Example:
        >>> import dask.array as da
        >>> import pyinterp.dask as dask_stats
        >>> values = da.random.random((10000,), chunks=1000)
        >>> digest = dask_stats.tdigest(values)
        >>> print(f"Median: {digest.quantile(0.5):.4f}")
        >>> print(f"Q25: {digest.quantile(0.25):.4f}")
        >>> print(f"Q75: {digest.quantile(0.75):.4f}")

    """
    _check_dask_available()
    import dask.array as da  # noqa: PLC0415 (to avoid import issues)

    values = _validate_dask_array(values, "values")
    if weights is not None:
        weights = _validate_dask_array(weights, "weights")
        _validate_shapes_match(values, weights)

    def _process_block(
        values_block: np.ndarray,
        weights_block: np.ndarray | None,
        axis: list[int] | None,
        compression: int,
        dtype: str | type | np.dtype | None,
        block_id: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Process a single block and return TDigest as object array."""
        digest = core.TDigest(
            values_block,
            weights=weights_block,
            axis=axis,
            compression=compression,
            dtype=dtype,
        )
        result = np.empty((1,), dtype=object)
        result[0] = digest
        return result

    if weights is None:
        blocks = da.map_blocks(
            _process_block,
            values,
            None,
            axis,
            compression,
            dtype,
            dtype=object,
            drop_axis=list(range(values.ndim)),
            new_axis=0,
        )
    else:
        blocks = da.map_blocks(
            _process_block,
            values,
            weights,
            axis,
            compression,
            dtype,
            dtype=object,
            drop_axis=list(range(values.ndim)),
            new_axis=0,
        )

    # Compute all blocks and merge
    results = blocks.compute()

    # Merge all results using in-place addition
    merged = results[0]
    for item in results[1:]:
        merged += item

    return merged


def binning1d(
    x: dask.array.Array,
    z: dask.array.Array,
    binning: core.Binning1DHolder,
    weights: dask.array.Array | None = None,
) -> core.Binning1DHolder:
    """Accumulate values into 1D bins from a dask array.

    This function processes a dask array in parallel, binning values according
    to the x coordinates and accumulating statistics in each bin.

    Args:
        x: Dask array of x coordinates.
        z: Dask array of values to bin.
        binning: A Binning1D instance defining the bins. A copy is made
            internally, so the original is not modified.
        weights: Optional dask array of weights with the same shape as z.

    Returns:
        A new Binning1D instance with accumulated statistics.

    Raises:
        ImportError: If dask is not installed.
        TypeError: If inputs are not dask arrays.
        ValueError: If x and z have different shapes, or if weights shape
            doesn't match.

    Example:
        >>> import dask.array as da
        >>> import numpy as np
        >>> import pyinterp
        >>> import pyinterp.dask as dask_stats
        >>>
        >>> # Create bins and data
        >>> axis = pyinterp.Axis(np.linspace(0, 10, 11))
        >>> binning = pyinterp.Binning1D(axis)
        >>>
        >>> # Create dask arrays
        >>> x = da.random.uniform(0, 10, size=(10000,), chunks=1000)
        >>> z = da.random.random((10000,), chunks=1000)
        >>>
        >>> # Compute binned statistics
        >>> result = dask_stats.binning1d(x, z, binning)
        >>> print(result.mean())

    """
    _check_dask_available()

    import dask.array as da  # noqa: PLC0415 (to avoid import issues)

    x = _validate_dask_array(x, "x")
    z = _validate_dask_array(z, "z")
    if x.shape != z.shape:
        msg = f"x and z must have the same shape, got {x.shape} and {z.shape}"
        raise ValueError(msg)
    if weights is not None:
        weights = _validate_dask_array(weights, "weights")
        if weights.shape != z.shape:
            msg = (
                f"weights and z must have the same shape, "
                f"got {weights.shape} and {z.shape}"
            )
            raise ValueError(msg)

    # Get axis and range from the binning instance for creating new instances
    axis = binning.x
    bin_range = binning.range()

    def _process_block(
        x_block: np.ndarray,
        z_block: np.ndarray,
        weights_block: np.ndarray | None,
        axis: core.Axis,
        bin_range: tuple[float, float] | None,
        block_id: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Process a single block and return binning as object array."""
        # Create a fresh binning instance for this block
        local_binning = core.Binning1D(copy.copy(axis), range=bin_range)
        weights = weights_block.ravel() if weights_block is not None else None
        local_binning.push(
            x_block.ravel(),
            z_block.ravel(),
            weights=weights,
        )
        result = np.empty((1,), dtype=object)
        result[0] = local_binning
        return result

    if weights is None:
        blocks = da.map_blocks(
            _process_block,
            x,
            z,
            None,
            axis,
            bin_range,
            dtype=object,
            drop_axis=list(range(x.ndim)),
            new_axis=0,
        )
    else:
        blocks = da.map_blocks(
            _process_block,
            x,
            z,
            weights,
            axis,
            bin_range,
            dtype=object,
            drop_axis=list(range(x.ndim)),
            new_axis=0,
        )

    # Compute all blocks and merge
    results = blocks.compute()

    # Merge all results using in-place addition
    merged = results[0]
    for item in results[1:]:
        merged += item

    return merged


def binning2d(
    x: dask.array.Array,
    y: dask.array.Array,
    z: dask.array.Array,
    binning: core.Binning2DHolder,
    simple: bool = True,
) -> core.Binning2DHolder:
    """Accumulate values into 2D bins from dask arrays.

    This function processes dask arrays in parallel, binning values according
    to the x and y coordinates and accumulating statistics in each bin.

    Args:
        x: Dask array of x coordinates.
        y: Dask array of y coordinates.
        z: Dask array of values to bin.
        binning: A Binning2D instance defining the bins. A copy is made
            internally, so the original is not modified.
        simple: If True, use simple binning (nearest neighbor). If False,
            use linear interpolation to distribute values among neighboring
            bins. Default is True.

    Returns:
        A new Binning2D instance with accumulated statistics.

    Raises:
        ImportError: If dask is not installed.
        TypeError: If inputs are not dask arrays.
        ValueError: If x, y, and z have different shapes.

    Example:
        >>> import dask.array as da
        >>> import numpy as np
        >>> import pyinterp
        >>> import pyinterp.dask as dask_stats
        >>>
        >>> # Create bins and data
        >>> x_axis = pyinterp.Axis(np.linspace(0, 10, 11))
        >>> y_axis = pyinterp.Axis(np.linspace(0, 10, 11))
        >>> binning = pyinterp.Binning2D(x_axis, y_axis)
        >>>
        >>> # Create dask arrays
        >>> x = da.random.uniform(0, 10, size=(10000,), chunks=1000)
        >>> y = da.random.uniform(0, 10, size=(10000,), chunks=1000)
        >>> z = da.random.random((10000,), chunks=1000)
        >>>
        >>> # Compute binned statistics
        >>> result = dask_stats.binning2d(x, y, z, binning)
        >>> print(result.mean())

    """
    _check_dask_available()

    import dask.array as da  # noqa: PLC0415 (to avoid import issues)

    x = _validate_dask_array(x, "x")
    y = _validate_dask_array(y, "y")
    z = _validate_dask_array(z, "z")

    if x.shape != y.shape or x.shape != z.shape:
        msg = (
            f"x, y, and z must have the same shape, "
            f"got {x.shape}, {y.shape}, and {z.shape}"
        )
        raise ValueError(msg)

    # Get axes and spheroid from the binning instance
    x_axis = binning.x
    y_axis = binning.y
    spheroid = binning.spheroid

    def _process_block(
        x_block: np.ndarray,
        y_block: np.ndarray,
        z_block: np.ndarray,
        x_axis: core.Axis,
        y_axis: core.Axis,
        spheroid: core.geometry.geographic.Spheroid | None,
        simple: bool,
        block_id: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Process a single block and return binning as object array."""
        # Create a fresh binning instance for this block
        local_binning = core.Binning2D(
            copy.copy(x_axis),
            copy.copy(y_axis),
            spheroid=spheroid,
        )
        local_binning.push(
            x_block.ravel(),
            y_block.ravel(),
            z_block.ravel(),
            simple=simple,
        )
        result = np.empty((1,), dtype=object)
        result[0] = local_binning
        return result

    blocks = da.map_blocks(
        _process_block,
        x,
        y,
        z,
        x_axis,
        y_axis,
        spheroid,
        simple,
        dtype=object,
        drop_axis=list(range(x.ndim)),
        new_axis=0,
    )

    # Compute all blocks and merge
    results = blocks.compute()

    # Merge all results using in-place addition
    merged = results[0]
    for item in results[1:]:
        merged += item

    return merged


def histogram2d(
    x: dask.array.Array,
    y: dask.array.Array,
    z: dask.array.Array,
    histogram: core.Histogram2DHolder,
) -> core.Histogram2DHolder:
    """Accumulate values into a 2D histogram from dask arrays.

    This function processes dask arrays in parallel, accumulating values
    into a 2D histogram based on x and y coordinates.

    Args:
        x: Dask array of x coordinates.
        y: Dask array of y coordinates.
        z: Dask array of values to accumulate.
        histogram: A Histogram2D instance defining the grid. A copy is made
            internally, so the original is not modified.

    Returns:
        A new Histogram2D instance with accumulated values.

    Raises:
        ImportError: If dask is not installed.
        TypeError: If inputs are not dask arrays.
        ValueError: If x, y, and z have different shapes.

    Example:
        >>> import dask.array as da
        >>> import numpy as np
        >>> import pyinterp
        >>> import pyinterp.dask as dask_stats
        >>>
        >>> # Create histogram and data
        >>> x_axis = pyinterp.Axis(np.linspace(0, 10, 11))
        >>> y_axis = pyinterp.Axis(np.linspace(0, 10, 11))
        >>> hist = pyinterp.Histogram2D(x_axis, y_axis)
        >>>
        >>> # Create dask arrays
        >>> x = da.random.uniform(0, 10, size=(10000,), chunks=1000)
        >>> y = da.random.uniform(0, 10, size=(10000,), chunks=1000)
        >>> z = da.random.random((10000,), chunks=1000)
        >>>
        >>> # Compute histogram
        >>> result = dask_stats.histogram2d(x, y, z, hist)
        >>> print(result.mean())
        >>> print(result.quantile(0.5))

    """
    _check_dask_available()

    import dask.array as da  # noqa: PLC0415 (to avoid import issues)

    x = _validate_dask_array(x, "x")
    y = _validate_dask_array(y, "y")
    z = _validate_dask_array(z, "z")

    if x.shape != y.shape or x.shape != z.shape:
        msg = (
            f"x, y, and z must have the same shape, "
            f"got {x.shape}, {y.shape}, and {z.shape}"
        )
        raise ValueError(msg)

    # Get axes from the histogram instance
    x_axis = histogram.x
    y_axis = histogram.y

    def _process_block(
        x_block: np.ndarray,
        y_block: np.ndarray,
        z_block: np.ndarray,
        x_axis: core.Axis,
        y_axis: core.Axis,
        block_id: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Process a single block and return histogram as object array."""
        # Create a fresh histogram instance for this block
        local_hist = core.Histogram2D(
            copy.copy(x_axis),
            copy.copy(y_axis),
        )
        local_hist.push(
            x_block.ravel(),
            y_block.ravel(),
            z_block.ravel(),
        )
        result = np.empty((1,), dtype=object)
        result[0] = local_hist
        return result

    blocks = da.map_blocks(
        _process_block,
        x,
        y,
        z,
        x_axis,
        y_axis,
        dtype=object,
        drop_axis=list(range(x.ndim)),
        new_axis=0,
    )

    # Compute all blocks and merge
    results = blocks.compute()

    # Merge all results using in-place addition
    merged = results[0]
    for item in results[1:]:
        merged += item

    return merged

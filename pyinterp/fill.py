# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Fill Methods Wrapper.

This module provides convenient wrapper functions around pyinterp.core.fill
methods that accept both configuration objects and simplified keyword
arguments.

For most use cases, you can simply pass keyword arguments:
>>> iterations, residual = gauss_seidel(grid, max_iterations=100, epsilon=1e-5)

For advanced configuration, pass a config object:
>>> from pyinterp.config import fill
>>> config = fill.GaussSeidel().with_max_iterations(100).with_epsilon(1e-5)
>>> iterations, residual = gauss_seidel(grid, config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from . import core
from .core.config import fill


if TYPE_CHECKING:
    from .type_hints import NDArray2DFloat32, NDArray2DFloat64

__all__ = [
    "fft_inpaint",
    "gauss_seidel",
    "loess",
    "matrix",
    "multigrid",
    "vector",
]

#: First guess initialization methods
FirstGuessMethod = Literal["zero", "zonal_average"]

#: Loess value type options
LoessValueType = Literal["all", "defined", "undefined"]

# First guess mapping
_FIRST_GUESS_MAP = {
    "zero": fill.FirstGuess.ZERO,
    "zonal_average": fill.FirstGuess.ZONAL_AVERAGE,
}

# Loess value type mapping
_LOESS_VALUE_TYPE_MAP = {
    "all": fill.LoessValueType.ALL,
    "defined": fill.LoessValueType.DEFINED,
    "undefined": fill.LoessValueType.UNDEFINED,
}

# TypeVars for generic config creation
_FillConfig = TypeVar(
    "_FillConfig",
    fill.FFTInpaint,
    fill.GaussSeidel,
    fill.Loess,
    fill.Multigrid,
)

if TYPE_CHECKING:
    _FloatArrayT = TypeVar(
        "_FloatArrayT",
        NDArray2DFloat32,
        NDArray2DFloat64,
    )
else:
    _FloatArrayT = TypeVar("_FloatArrayT")


def _build_config(
    config_class: type[_FillConfig],
    method_map: dict[str, tuple[str, dict | None]] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> _FillConfig:
    """Build a configuration object from keyword arguments.

    Args:
        config_class: The configuration class to instantiate
        method_map: Dict mapping parameter names to method names and optional
            value maps
        **kwargs: Keyword arguments to apply

    Returns:
        Configured object

    """
    cfg = config_class()
    method_map = method_map or {}

    for param_name, value in kwargs.items():
        if value is None or param_name not in method_map:
            continue

        method_name, value_map = method_map[param_name]
        method = getattr(cfg, method_name)
        cfg = method(value_map[value] if value_map else value)

    return cast("_FillConfig", cfg)


def fft_inpaint(
    grid: _FloatArrayT,
    config: fill.FFTInpaint | None = None,
    *,
    max_iterations: int | None = None,
    epsilon: float | None = None,
    sigma: float | None = None,
    first_guess: FirstGuessMethod | None = None,
    is_periodic: bool | None = None,
    num_threads: int | None = None,
) -> tuple[int, float]:
    """Fill missing values using FFT-based inpainting.

    Args:
        grid: 2D grid with missing values (NaN)
        config: Configuration object (if provided, keyword args are ignored)
        max_iterations: Maximum number of iterations
        epsilon: Convergence criterion
        sigma: Smoothing parameter for Gaussian filter
        first_guess: Initial guess method
        is_periodic: Whether to assume periodic boundaries
        num_threads: Number of threads (0 = auto)

    Returns:
        Tuple of (number of iterations, final residual)

    Examples:
        Simple usage:

        >>> iterations, residual = fft_inpaint(
        ...     grid, max_iterations=100, epsilon=1e-5
        ... )

        Advanced usage with config object:

        >>> from pyinterp.core.config import fill
        >>> config = (
        ...     fill.FFTInpaint()
        ...     .with_max_iterations(100)
        ...     .with_epsilon(1e-5)
        ...     .with_sigma(0.5)
        ... )
        >>> iterations, residual = fft_inpaint(grid, config)

    """
    # If config is provided, use it directly
    if config is not None:
        return core.fill.fft_inpaint(grid, config)

    # Create config from keyword arguments
    method_map = {
        "max_iterations": ("with_max_iterations", None),
        "epsilon": ("with_epsilon", None),
        "sigma": ("with_sigma", None),
        "first_guess": ("with_first_guess", _FIRST_GUESS_MAP),
        "is_periodic": ("with_is_periodic", None),
        "num_threads": ("with_num_threads", None),
    }
    cfg = _build_config(
        fill.FFTInpaint,
        method_map,
        max_iterations=max_iterations,
        epsilon=epsilon,
        sigma=sigma,
        first_guess=first_guess,
        is_periodic=is_periodic,
        num_threads=num_threads,
    )

    return core.fill.fft_inpaint(grid, cfg)


def gauss_seidel(
    grid: _FloatArrayT,
    config: fill.GaussSeidel | None = None,
    *,
    max_iterations: int | None = None,
    epsilon: float | None = None,
    relaxation: float | None = None,
    first_guess: FirstGuessMethod | None = None,
    is_periodic: bool | None = None,
    num_threads: int | None = None,
) -> tuple[int, float]:
    """Fill missing values using Gauss-Seidel relaxation.

    Args:
        grid: 2D grid with missing values (NaN)
        config: Configuration object (if provided, keyword args are ignored)
        max_iterations: Maximum number of iterations
        epsilon: Convergence criterion
        relaxation: Relaxation parameter (0 < relaxation <= 2)
        first_guess: Initial guess method
        is_periodic: Whether to assume periodic boundaries
        num_threads: Number of threads (0 = auto)

    Returns:
        Tuple of (number of iterations, final residual)

    Examples:
        >>> iterations, residual = gauss_seidel(
        ...     grid, max_iterations=1000, epsilon=1e-4, relaxation=1.5
        ... )

    """
    # If config is provided, use it directly
    if config is not None:
        return core.fill.gauss_seidel(grid, config)

    # Create config from keyword arguments
    method_map = {
        "max_iterations": ("with_max_iterations", None),
        "epsilon": ("with_epsilon", None),
        "relaxation": ("with_relaxation", None),
        "first_guess": ("with_first_guess", _FIRST_GUESS_MAP),
        "is_periodic": ("with_is_periodic", None),
        "num_threads": ("with_num_threads", None),
    }
    cfg = _build_config(
        fill.GaussSeidel,
        method_map,
        max_iterations=max_iterations,
        epsilon=epsilon,
        relaxation=relaxation,
        first_guess=first_guess,
        is_periodic=is_periodic,
        num_threads=num_threads,
    )

    return core.fill.gauss_seidel(grid, cfg)


def loess(
    data: _FloatArrayT,
    config: fill.Loess | None = None,
    *,
    nx: int | None = None,
    ny: int | None = None,
    max_iterations: int | None = None,
    epsilon: float | None = None,
    value_type: LoessValueType | None = None,
    first_guess: FirstGuessMethod | None = None,
    is_periodic: bool | None = None,
    num_threads: int | None = None,
) -> _FloatArrayT:
    """Fill missing values using LOESS (locally weighted regression).

    Args:
        data: 2D grid with missing values (NaN)
        config: Configuration object (if provided, keyword args are ignored)
        nx: Window size in X direction
        ny: Window size in Y direction
        max_iterations: Maximum number of iterations. If the value is 1, a
            single-pass LOESS is performed, the first guess option is ignored.
        epsilon: Convergence criterion
        value_type: Which values to use in regression
        first_guess: Initial guess method
        is_periodic: Whether to assume periodic boundaries
        num_threads: Number of threads (0 = auto)

    Returns:
        Filled grid

    Examples:
        >>> filled_grid = loess(
        ...     data, nx=5, ny=5, max_iterations=10, value_type="defined"
        ... )

    """
    # If config is provided, use it directly
    if config is not None:
        return core.fill.loess(data, config)

    # Create config from keyword arguments
    method_map = {
        "nx": ("with_nx", None),
        "ny": ("with_ny", None),
        "max_iterations": ("with_max_iterations", None),
        "epsilon": ("with_epsilon", None),
        "value_type": ("with_value_type", _LOESS_VALUE_TYPE_MAP),
        "first_guess": ("with_first_guess", _FIRST_GUESS_MAP),
        "is_periodic": ("with_is_periodic", None),
        "num_threads": ("with_num_threads", None),
    }
    cfg = _build_config(
        fill.Loess,
        method_map,
        nx=nx,
        ny=ny,
        max_iterations=max_iterations,
        epsilon=epsilon,
        value_type=value_type,
        first_guess=first_guess,
        is_periodic=is_periodic,
        num_threads=num_threads,
    )

    return core.fill.loess(data, cfg)


def multigrid(
    grid: _FloatArrayT,
    config: fill.Multigrid | None = None,
    *,
    max_iterations: int | None = None,
    epsilon: float | None = None,
    pre_smooth: int | None = None,
    post_smooth: int | None = None,
    first_guess: FirstGuessMethod | None = None,
    is_periodic: bool | None = None,
    num_threads: int | None = None,
) -> tuple[int, float]:
    """Fill missing values using multigrid method.

    Args:
        grid: 2D grid with missing values (NaN)
        config: Configuration object (if provided, keyword args are ignored)
        max_iterations: Maximum number of iterations
        epsilon: Convergence criterion
        pre_smooth: Number of pre-smoothing iterations
        post_smooth: Number of post-smoothing iterations
        first_guess: Initial guess method
        is_periodic: Whether to assume periodic boundaries
        num_threads: Number of threads (0 = auto)

    Returns:
        Tuple of (number of iterations, final residual)

    Examples:
        >>> iterations, residual = multigrid(
        ...     grid,
        ...     max_iterations=100,
        ...     epsilon=1e-5,
        ...     pre_smooth=2,
        ...     post_smooth=2,
        ... )

    """
    # If config is provided, use it directly
    if config is not None:
        return core.fill.multigrid(grid, config)

    # Create config from keyword arguments
    method_map = {
        "max_iterations": ("with_max_iterations", None),
        "epsilon": ("with_epsilon", None),
        "pre_smooth": ("with_pre_smooth", None),
        "post_smooth": ("with_post_smooth", None),
        "first_guess": ("with_first_guess", _FIRST_GUESS_MAP),
        "is_periodic": ("with_is_periodic", None),
        "num_threads": ("with_num_threads", None),
    }
    cfg = _build_config(
        fill.Multigrid,
        method_map,
        max_iterations=max_iterations,
        epsilon=epsilon,
        pre_smooth=pre_smooth,
        post_smooth=post_smooth,
        first_guess=first_guess,
        is_periodic=is_periodic,
        num_threads=num_threads,
    )

    return core.fill.multigrid(grid, cfg)


def matrix(grid: _FloatArrayT, fill_value: float | None = None) -> None:
    """Fill a 2D array by linear interpolation.

    Args:
        grid: 2D array to fill
        fill_value: Value to use to determine missing values (if None, use NaN)

    Note:
        This function modifies the grid in-place.

    Examples:
        >>> import numpy as np
        >>> grid = np.arange(100, dtype=np.float32).reshape(10, 10)
        >>> grid[::2, ::2] = np.nan
        >>> matrix(grid)

    """
    core.fill.matrix(
        grid, fill_value if fill_value is not None else float("nan")
    )


def vector(array: _FloatArrayT, fill_value: float | None = None) -> None:
    """Fill a 1D array by linear interpolation.

    Args:
        array: 1D array to fill
        fill_value: Value to use to determine missing values (if None, use NaN)

    Note:
        This function modifies the array in-place.

    Examples:
        >>> import numpy as np
        >>> arr = np.arange(100, dtype=np.float32)
        >>> arr[10:20] = np.nan
        >>> vector(arr)

    """
    core.fill.vector(
        array, fill_value if fill_value is not None else float("nan")
    )

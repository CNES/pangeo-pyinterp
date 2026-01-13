# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Regular Grid Interpolator.

This module provides convenient wrapper functions around pyinterp.core
interpolation functions that accept both string-based method names and
configuration objects.

For most use cases, you can simply pass a string method name:
>>> result = bivariate(grid, x, y, "bilinear")

For advanced configuration, pass a config object:
>>> from pyinterp.config import windowed
>>> config = windowed.Bivariate.bicubic().with_half_window_size_x(10)
>>> result = bivariate(grid, x, y, config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar, cast, get_args, overload

from . import core
from .core.config import geometric, windowed


if TYPE_CHECKING:
    from .core import GridHolder
    from .type_hints import (
        NDArray1DFloat32,
        NDArray1DFloat64,
        NDArray1DNumeric,
        NDArray1DNumericWithTime,
    )

__all__ = [
    "bivariate",
    "quadrivariate",
    "trivariate",
    "univariate",
    "univariate_derivative",
]

#: Geometric interpolation methods (simple, non-windowed)
GeometricMethods = Literal["bilinear", "idw", "nearest"]

#: Windowed interpolation methods (with window functions)
WindowedMethods = Literal[
    "akima",
    "akima_periodic",
    "bicubic",
    "bilinear",
    "c_spline",
    "c_spline_not_a_knot",
    "c_spline_periodic",
    "linear",
    "polynomial",
    "steffen",
]

#: All valid interpolation method names
InterpolationMethods = GeometricMethods | WindowedMethods

#: Boundary mode strings
BoundaryMode = Literal["shrink", "undef"]

#: Axis configuration strings
AxisConfigStr = Literal["linear", "nearest"]

# Extract valid method names for runtime validation
_GEOMETRIC_METHODS = get_args(GeometricMethods)
_WINDOWED_METHODS = get_args(WindowedMethods)

# Boundary mode string to enum mapping
_BOUNDARY_MAP = {
    "undef": windowed.BoundaryConfig.undef,
    "shrink": windowed.BoundaryConfig.shrink,
}

_AXIS_MAP = {
    "linear": windowed.AxisConfig.linear,
    "nearest": windowed.AxisConfig.nearest,
}

# TypeVars for generic config creation
_GeometricConfig = TypeVar(
    "_GeometricConfig",
    geometric.Bivariate,
    geometric.Trivariate,
    geometric.Quadrivariate,
)

_WindowedConfig = TypeVar(
    "_WindowedConfig",
    windowed.Univariate,
    windowed.Bivariate,
    windowed.Trivariate,
    windowed.Quadrivariate,
)


def _make_geometric_config(
    method: GeometricMethods,
    class_type: type[_GeometricConfig],
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
) -> _GeometricConfig:
    """Create a geometric interpolation configuration."""
    if method not in _GEOMETRIC_METHODS:
        raise ValueError(
            f"Unknown geometric method: '{method}'. "
            f"Valid methods: {', '.join(_GEOMETRIC_METHODS)}"
        )

    factory = getattr(class_type, method)
    config = factory()

    return cast(
        "_GeometricConfig",
        config.with_bounds_error(bounds_error).with_num_threads(num_threads),
    )


@overload
def _make_windowed_config(
    method: WindowedMethods,
    class_type: type[windowed.Univariate],
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
    half_window_size: int | None = None,
    boundary_mode: BoundaryMode | None = None,
) -> windowed.Univariate: ...


@overload
def _make_windowed_config(
    method: WindowedMethods,
    class_type: type[windowed.Bivariate],
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
    half_window_size_x: int | None = None,
    half_window_size_y: int | None = None,
    boundary_mode: BoundaryMode | None = None,
) -> windowed.Bivariate: ...


@overload
def _make_windowed_config(
    method: WindowedMethods,
    class_type: type[windowed.Trivariate],
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
    half_window_size_x: int | None = None,
    half_window_size_y: int | None = None,
    boundary_mode: BoundaryMode | None = None,
    third_axis: AxisConfigStr | None = None,
) -> windowed.Trivariate: ...


@overload
def _make_windowed_config(
    method: WindowedMethods,
    class_type: type[windowed.Quadrivariate],
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
    half_window_size_x: int | None = None,
    half_window_size_y: int | None = None,
    boundary_mode: BoundaryMode | None = None,
    third_axis: AxisConfigStr | None = None,
    fourth_axis: AxisConfigStr | None = None,
) -> windowed.Quadrivariate: ...


def _make_windowed_config(
    method: WindowedMethods,
    class_type: type[_WindowedConfig],
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
    half_window_size: int | None = None,
    half_window_size_x: int | None = None,
    half_window_size_y: int | None = None,
    boundary_mode: BoundaryMode | None = None,
    third_axis: AxisConfigStr | None = None,
    fourth_axis: AxisConfigStr | None = None,
) -> _WindowedConfig:
    if method not in _WINDOWED_METHODS:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Valid: {', '.join(_WINDOWED_METHODS)}"
        )

    config: _WindowedConfig = getattr(class_type, method)()
    config = config.with_bounds_error(bounds_error).with_num_threads(
        num_threads
    )

    if boundary_mode is not None:
        config = config.with_boundary_mode(_BOUNDARY_MAP[boundary_mode]())
    if half_window_size is not None:
        assert isinstance(config, windowed.Univariate)
        config = config.with_half_window_size(half_window_size)
    if half_window_size_x is not None:
        assert not isinstance(config, windowed.Univariate)
        config = config.with_half_window_size_x(half_window_size_x)
    if half_window_size_y is not None:
        assert not isinstance(config, windowed.Univariate)
        config = config.with_half_window_size_y(half_window_size_y)
    if third_axis is not None:
        assert isinstance(config, windowed.Trivariate | windowed.Quadrivariate)
        config = config.with_third_axis(_AXIS_MAP[third_axis]())
    if fourth_axis is not None:
        assert isinstance(config, windowed.Quadrivariate)
        config = config.with_fourth_axis(_AXIS_MAP[fourth_axis]())

    return config


def _validate_no_windowed_options(
    method: str,
    half_window_size: int | None = None,
    half_window_size_x: int | None = None,
    half_window_size_y: int | None = None,
    boundary_mode: BoundaryMode | None = None,
    third_axis: AxisConfigStr | None = None,
    fourth_axis: AxisConfigStr | None = None,
) -> None:
    """Validate that windowed options aren't used with geometric methods."""
    options = {
        "half_window_size": half_window_size,
        "half_window_size_x": half_window_size_x,
        "half_window_size_y": half_window_size_y,
        "boundary_mode": boundary_mode,
        "third_axis": third_axis,
        "fourth_axis": fourth_axis,
    }
    invalid_opts = [
        name for name, value in options.items() if value is not None
    ]

    if invalid_opts:
        raise TypeError(
            f"Options {invalid_opts} are not valid for geometric method "
            f"'{method}'. These options are only available for windowed "
            f"methods: {', '.join(_WINDOWED_METHODS)}"
        )


# Univariate interpolation
@overload
def univariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    method: windowed.Univariate,
) -> NDArray1DFloat32 | NDArray1DFloat64: ...


@overload
def univariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    method: WindowedMethods = "linear",
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
    half_window_size: int | None = None,
    boundary_mode: BoundaryMode | None = None,
) -> NDArray1DFloat32 | NDArray1DFloat64: ...


def univariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    method: windowed.Univariate | WindowedMethods = "linear",
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
    half_window_size: int | None = None,
    boundary_mode: BoundaryMode | None = None,
) -> NDArray1DFloat32 | NDArray1DFloat64:
    """Univariate interpolation.

    Args:
        grid: The 1D grid to interpolate from
        x: X coordinates at which to interpolate
        method: Interpolation method (config object or string)
        bounds_error: If True, raise error for out-of-bounds coordinates
        num_threads: Number of threads to use (0 = auto)
        half_window_size: Half window size for the interpolation
        boundary_mode: Boundary handling mode

    Return:
        Interpolated values

    Examples:
        Simple usage:

        >>> result = univariate(grid, x, "linear")
        >>> result = univariate(grid, x, "c_spline", window_size=10)

        Advanced usage with config objects:

        >>> from pyinterp.core.config import windowed
        >>> config = windowed.Univariate.c_spline().with_half_window_size(10)
        >>> result = univariate(grid, x, config)

    """
    # If method is a config object, use it directly
    if isinstance(method, windowed.Univariate):
        return core.univariate(grid, x, method)

    # String-based method - create config (univariate only supports windowed)
    method = cast("WindowedMethods", method)
    config = _make_windowed_config(
        method,
        windowed.Univariate,
        bounds_error=bounds_error,
        num_threads=num_threads,
        half_window_size=half_window_size,
        boundary_mode=boundary_mode,
    )

    return core.univariate(grid, x, config)


def univariate_derivative(
    grid: GridHolder,
    x: NDArray1DNumeric,
    method: windowed.Univariate | WindowedMethods = "linear",
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
    half_window_size: int | None = None,
    boundary_mode: BoundaryMode | None = None,
) -> NDArray1DFloat32 | NDArray1DFloat64:
    """Calculate derivatives on a 1D grid.

    Args:
        grid: The 1D grid containing data
        x: X coordinates at which to calculate derivatives
        method: Interpolation method (config object or string)
        bounds_error: If True, raise error for out-of-bounds coordinates
        num_threads: Number of threads to use (0 = auto)
        half_window_size: Half window size for the derivative calculation
        boundary_mode: Boundary handling mode

    Return:
        Derivative values

    Examples:
        Simple usage:

        >>> derivative = univariate_derivative(grid, x, "c_spline")
        >>> derivative = univariate_derivative(grid, x, "akima", window_size=7)

        Advanced usage with config objects:

        >>> from pyinterp.core.config import windowed
        >>> config = windowed.Univariate.c_spline().with_half_window_size(10)
        >>> derivative = univariate_derivative(grid, x, config)

    """
    # If method is a config object, use it directly
    if not isinstance(method, str):
        return core.univariate_derivative(grid, x, method)

    # String-based method - create config
    method = cast("WindowedMethods", method)
    config = _make_windowed_config(
        method,
        windowed.Univariate,
        bounds_error=bounds_error,
        num_threads=num_threads,
        half_window_size=half_window_size,
        boundary_mode=boundary_mode,
    )

    return core.univariate_derivative(grid, x, config)


# Bivariate interpolation
@overload
def bivariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    y: NDArray1DNumeric,
    method: geometric.Bivariate | windowed.Bivariate,
) -> NDArray1DFloat32 | NDArray1DFloat64: ...


@overload
def bivariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    y: NDArray1DNumeric,
    method: GeometricMethods = "bilinear",
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
) -> NDArray1DFloat32 | NDArray1DFloat64: ...


@overload
def bivariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    y: NDArray1DNumeric,
    method: WindowedMethods,
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
    half_window_size_x: int | None = None,
    half_window_size_y: int | None = None,
    boundary_mode: BoundaryMode | None = None,
) -> NDArray1DFloat32 | NDArray1DFloat64: ...


def bivariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    y: NDArray1DNumeric,
    method: geometric.Bivariate
    | windowed.Bivariate
    | InterpolationMethods = "bilinear",
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
    half_window_size_x: int | None = None,
    half_window_size_y: int | None = None,
    boundary_mode: BoundaryMode | None = None,
) -> NDArray1DFloat32 | NDArray1DFloat64:
    """Bivariate interpolation.

    Args:
        grid: The 2D grid to interpolate from
        x: X coordinates at which to interpolate
        y: Y coordinates at which to interpolate
        method: Interpolation method (config object or string)
        bounds_error: If True, raise error for out-of-bounds coordinates
        num_threads: Number of threads to use (0 = auto)
        half_window_size_x: Half window size for X axis (windowed methods only)
        half_window_size_y: Half window size for Y axis (windowed methods only)
        boundary_mode: Boundary handling mode (windowed methods only)

    Return:
        Interpolated values

    Examples:
        Simple usage:

        >>> result = bivariate(grid, x, y, "bilinear")
        >>> result = bivariate(grid, x, y, "bicubic", window_size_x=10)

        Advanced usage with config objects:

        >>> from pyinterp.core.config import windowed
        >>> config = windowed.Bivariate.bicubic().with_half_window_size_x(10)
        >>> result = bivariate(grid, x, y, config)

    """
    config: geometric.Bivariate | windowed.Bivariate

    # If method is a config object, use it directly
    if not isinstance(method, str):
        return core.bivariate(grid, x, y, method)

    # String-based method - create config
    if method in _GEOMETRIC_METHODS:
        # Validate no windowed options for geometric methods
        _validate_no_windowed_options(
            method,
            half_window_size_x=half_window_size_x,
            half_window_size_y=half_window_size_y,
            boundary_mode=boundary_mode,
        )
        method = cast("GeometricMethods", method)
        config = _make_geometric_config(
            method,
            geometric.Bivariate,
            bounds_error=bounds_error,
            num_threads=num_threads,
        )
    else:
        # Windowed method
        method = cast("WindowedMethods", method)
        config = _make_windowed_config(
            method,
            windowed.Bivariate,
            bounds_error=bounds_error,
            num_threads=num_threads,
            half_window_size_x=half_window_size_x,
            half_window_size_y=half_window_size_y,
            boundary_mode=boundary_mode,
        )

    return core.bivariate(grid, x, y, config)


@overload
def trivariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    y: NDArray1DNumeric,
    z: NDArray1DNumericWithTime,
    method: geometric.Trivariate | windowed.Trivariate,
) -> NDArray1DFloat32 | NDArray1DFloat64: ...


@overload
def trivariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    y: NDArray1DNumeric,
    z: NDArray1DNumericWithTime,
    method: GeometricMethods = "bilinear",
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
) -> NDArray1DFloat32 | NDArray1DFloat64: ...


@overload
def trivariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    y: NDArray1DNumeric,
    z: NDArray1DNumericWithTime,
    method: WindowedMethods,
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
    half_window_size_x: int | None = None,
    half_window_size_y: int | None = None,
    boundary_mode: BoundaryMode | None = None,
    third_axis: AxisConfigStr | None = None,
) -> NDArray1DFloat32 | NDArray1DFloat64: ...


def trivariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    y: NDArray1DNumeric,
    z: NDArray1DNumericWithTime,
    method: geometric.Trivariate
    | windowed.Trivariate
    | InterpolationMethods = "bilinear",
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
    half_window_size_x: int | None = None,
    half_window_size_y: int | None = None,
    boundary_mode: BoundaryMode | None = None,
    third_axis: AxisConfigStr | None = None,
) -> NDArray1DFloat32 | NDArray1DFloat64:
    """Trivariate interpolation.

    Args:
        grid: The 3D grid to interpolate from
        x: X coordinates at which to interpolate
        y: Y coordinates at which to interpolate
        z: Z coordinates at which to interpolate
        method: Interpolation method (config object or string)
        bounds_error: If True, raise error for out-of-bounds coordinates
        num_threads: Number of threads to use (0 = auto)
        half_window_size_x: Half window size for X axis (windowed methods only)
        half_window_size_y: Half window size for Y axis (windowed methods only)
        boundary_mode: Boundary handling mode (windowed methods only)
        third_axis: Interpolation method for Z axis (windowed methods only)

    Return:
        Interpolated values

    Raise:
        TypeError: If windowed options are provided with geometric methods

    Examples:
        >>> result = trivariate(grid, x, y, z, "bilinear")
        >>> result = trivariate(grid, x, y, z, "bicubic", third_axis="linear")

    """
    config: geometric.Trivariate | windowed.Trivariate

    # If method is a config object, use it directly
    if not isinstance(method, str):
        return core.trivariate(grid, x, y, z, method)

    # String-based method - create config
    if method in _GEOMETRIC_METHODS:
        # Validate no windowed options for geometric methods
        _validate_no_windowed_options(
            method,
            half_window_size_x=half_window_size_x,
            half_window_size_y=half_window_size_y,
            boundary_mode=boundary_mode,
            third_axis=third_axis,
        )
        method = cast("GeometricMethods", method)
        config = _make_geometric_config(
            method,
            geometric.Trivariate,
            bounds_error=bounds_error,
            num_threads=num_threads,
        )
    else:
        # Windowed method
        method = cast("WindowedMethods", method)
        config = _make_windowed_config(
            method,
            windowed.Trivariate,
            bounds_error=bounds_error,
            num_threads=num_threads,
            half_window_size_x=half_window_size_x,
            half_window_size_y=half_window_size_y,
            boundary_mode=boundary_mode,
            third_axis=third_axis,
        )

    return core.trivariate(grid, x, y, z, config)


@overload
def quadrivariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    y: NDArray1DNumeric,
    z: NDArray1DNumericWithTime,
    u: NDArray1DNumeric,
    method: geometric.Quadrivariate | windowed.Quadrivariate,
) -> NDArray1DFloat32 | NDArray1DFloat64: ...


@overload
def quadrivariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    y: NDArray1DNumeric,
    z: NDArray1DNumericWithTime,
    u: NDArray1DNumeric,
    method: GeometricMethods = "bilinear",
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
) -> NDArray1DFloat32 | NDArray1DFloat64: ...


@overload
def quadrivariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    y: NDArray1DNumeric,
    z: NDArray1DNumericWithTime,
    u: NDArray1DNumeric,
    method: WindowedMethods,
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
    half_window_size_x: int | None = None,
    half_window_size_y: int | None = None,
    boundary_mode: BoundaryMode | None = None,
    third_axis: AxisConfigStr | None = None,
    fourth_axis: AxisConfigStr | None = None,
) -> NDArray1DFloat32 | NDArray1DFloat64: ...


def quadrivariate(
    grid: GridHolder,
    x: NDArray1DNumeric,
    y: NDArray1DNumeric,
    z: NDArray1DNumericWithTime,
    u: NDArray1DNumeric,
    method: geometric.Quadrivariate
    | windowed.Quadrivariate
    | InterpolationMethods = "bilinear",
    *,
    bounds_error: bool = False,
    num_threads: int = 0,
    half_window_size_x: int | None = None,
    half_window_size_y: int | None = None,
    boundary_mode: BoundaryMode | None = None,
    third_axis: AxisConfigStr | None = None,
    fourth_axis: AxisConfigStr | None = None,
) -> NDArray1DFloat32 | NDArray1DFloat64:
    """Quadrivariate interpolation.

    Args:
        grid: The 4D grid to interpolate from
        x: X coordinates at which to interpolate
        y: Y coordinates at which to interpolate
        z: Z coordinates at which to interpolate
        u: U coordinates at which to interpolate
        method: Interpolation method (config object or string)
        bounds_error: If True, raise error for out-of-bounds coordinates
        num_threads: Number of threads to use (0 = auto)
        half_window_size_x: Half window size for X axis (windowed methods only)
        half_window_size_y: Half window size for Y axis (windowed methods only)
        boundary_mode: Boundary handling mode (windowed methods only)
        third_axis: Interpolation method for Z axis (windowed methods only)
        fourth_axis: Interpolation method for U axis (windowed methods only)

    Return:
        Interpolated values

    Raise:
        TypeError: If windowed options are provided with geometric methods

    Examples:
        >>> result = quadrivariate(grid, x, y, z, u, "bilinear")
        >>> result = quadrivariate(
        ...     grid,
        ...     x,
        ...     y,
        ...     z,
        ...     u,
        ...     "bicubic",
        ...     third_axis="linear",
        ...     fourth_axis="linear",
        ... )

    """
    config: geometric.Quadrivariate | windowed.Quadrivariate

    # If method is a config object, use it directly
    if not isinstance(method, str):
        return core.quadrivariate(grid, x, y, z, u, method)

    # String-based method - create config
    if method in _GEOMETRIC_METHODS:
        # Validate no windowed options for geometric methods
        _validate_no_windowed_options(
            method,
            half_window_size_x=half_window_size_x,
            half_window_size_y=half_window_size_y,
            boundary_mode=boundary_mode,
            third_axis=third_axis,
            fourth_axis=fourth_axis,
        )
        method = cast("GeometricMethods", method)
        config = _make_geometric_config(
            method,
            geometric.Quadrivariate,
            bounds_error=bounds_error,
            num_threads=num_threads,
        )
    else:
        # Windowed method
        method = cast("WindowedMethods", method)
        config = _make_windowed_config(
            method,
            windowed.Quadrivariate,
            bounds_error=bounds_error,
            num_threads=num_threads,
            half_window_size_x=half_window_size_x,
            half_window_size_y=half_window_size_y,
            boundary_mode=boundary_mode,
            third_axis=third_axis,
            fourth_axis=fourth_axis,
        )

    return core.quadrivariate(grid, x, y, z, u, config)

# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""XArray backend.

Build interpolation objects from xarray.DataArray instances
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, cast

import numpy as np


try:
    import xarray as xr
except ImportError as exc:
    raise ImportError(
        "pyinterp.backends.xarray requires xarray, install it with "
        "`pip install pyinterp[xarray]`"
    ) from exc

from .. import cf, core
from ..core.config import geometric
from ..regular_grid_interpolator import (
    _GEOMETRIC_METHODS,
    InterpolationMethods,
    _make_geometric_config,
    bivariate,
    quadrivariate,
    trivariate,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping

    from ..type_hints import (
        NDArray1D,
        NDArray1DDateTime64,
        NDArray1DFloat64,
        NDArray1DNumeric,
        NDArray1DNumericWithTime,
    )

__all__ = [
    "DataArrayAccessor",
    "DatasetAccessor",
    "Grid2D",
    "Grid3D",
    "Grid4D",
    "RegularGridInterpolator",
]

#: Two dimensional grid.
TWO_DIMENSIONS = 2

#: Three dimensional grid.
THREE_DIMENSIONS = 3

#: Four dimensional grid.
FOUR_DIMENSIONS = 4

#: Index of the longitude axis in a 2D, 3D, or 4D grid.
LONGITUDE_AXIS_INDEX = 0

#: Index of the temporal axis in a 3D or 4D grid.
TEMPORAL_AXIS_INDEX = 2

#: Interpolation on the grid defined by 1D coordinates, by number of
#: dimensions.
_OUTER_FUNCTIONS: dict[int, Callable[..., np.ndarray]] = {
    TWO_DIMENSIONS: core.bivariate_outer,
    THREE_DIMENSIONS: core.trivariate_outer,
    FOUR_DIMENSIONS: core.quadrivariate_outer,
}

#: Configuration of the geometric interpolators, by number of dimensions.
_GEOMETRIC_CONFIGS: dict[int, Any] = {
    TWO_DIMENSIONS: geometric.Bivariate,
    THREE_DIMENSIONS: geometric.Trivariate,
    FOUR_DIMENSIONS: geometric.Quadrivariate,
}

#: Options of the interpolation on the grid defined by 1D coordinates.
_OUTER_OPTIONS = frozenset({"bounds_error", "num_threads"})


class AxisIdentifier:
    """Identify axes defining longitudes and latitudes in a CF file.

    This class determines which dimensions in a data array correspond to
    longitude and latitude coordinates based on CF conventions.

    Args:
        data_array: The data array to be identified.

    """

    def __init__(self, data_array: xr.DataArray) -> None:
        """Initialize the AxisIdentifier with the provided data array."""
        self.data_array = data_array

    def _axis(self, units: cf.AxisUnit) -> str | None:
        """Return the name of the dimension that defines an axis.

        Args:
            units: The units of the axis

        Returns:
            The name of the coordinate

        """
        for name, coord in self.data_array.coords.items():
            if hasattr(coord, "units") and coord.units in units:
                return str(name)
        return None

    def longitude(self) -> str | None:
        """Return the name of the dimension that defines a longitude axis.

        Returns:
            The name of the longitude coordinate

        """
        return self._axis(cf.AxisLongitudeUnit())

    def latitude(self) -> str | None:
        """Return the name of the dimension that defines a latitude axis.

        Returns:
            The name of the latitude coordinates

        """
        return self._axis(cf.AxisLatitudeUnit())


def _identify_temporal_axis(
    data_array: xr.DataArray,
    dims: Iterable[Hashable],
) -> Hashable | None:
    """Identify the temporal axis in the data array."""
    for dim in dims:
        # Check coordinate associated with the dimension
        if dim in data_array.coords:
            coord = data_array.coords[dim]
            # Robust datetime check using numpy
            if np.issubdtype(
                coord.dtype,
                np.datetime64,
            ) or np.issubdtype(
                coord.dtype,
                np.timedelta64,
            ):
                # Support is limited to a single temporal axis; return as soon
                # as one is found. Any additional temporal axes will be
                # disregarded.
                return dim
    return None


@dataclasses.dataclass(frozen=True)
class _DimInfo:
    """Information about a dimension in the data array."""

    #: Data array
    _data_array: xr.DataArray
    #: Tuple of dimension names in standard order.
    dims: tuple[Hashable, ...]
    #: True if longitude was identified (at index 0).
    has_longitude: bool = False
    #: True if temporal axis was identified (at index 2).
    has_temporal: bool = False
    #: Indicates whether the dimension names differ in order from those in the
    #: provided data array.
    should_be_transposed: bool = False

    def axis(self, index: int) -> core.Axis | core.TemporalAxis:
        """Get dimension name at the specified index."""
        values = self.data_array.coords[self.dims[index]].values
        if index == LONGITUDE_AXIS_INDEX and self.has_longitude:
            return core.Axis(values, period=360.0)
        if index == TEMPORAL_AXIS_INDEX and self.has_temporal:
            return core.TemporalAxis(values)
        return core.Axis(values)

    @property
    def data_array(self) -> xr.DataArray:
        """Get the associated data array."""
        if self.should_be_transposed:
            return self._data_array.transpose(*self.dims)
        return self._data_array

    @property
    def datetime64(self) -> Hashable:
        """Get the temporal axis information if present."""
        if not self.has_temporal:
            raise AttributeError("No temporal axis present")
        return self.dims[2]


def _get_canonical_dimensions(
    data_array: xr.DataArray,
    ndims: int = 2,
) -> _DimInfo:
    """Get the name of dimensions that define the grid axes in canonical order.

    Identifies longitude, latitude, and temporal axes using CF conventions.
    Returns dimensions ordered as (Longitude, Latitude, Time, ...Others) to
    standardize grid processing.

    Target positions:
    - Index 0: Longitude (if present)
    - Index 1: Latitude (if present)
    - Index 2: Temporal axis (if present and ndims >= 3)

    Args:
        data_array: Provided data array.
        ndims: Number of dimensions expected for the variable.

    Returns:
        A _DimInfo instance containing ordered dimension names and flags
        indicating presence of longitude and temporal axes.

    Raises:
        ValueError: If the number of dimensions doesn't match ndims, or if an
            identified axis is not a dimension of the variable.

    """
    if data_array.ndim != ndims:
        raise ValueError(
            f"The number of dimensions of the variable is incorrect. "
            f"Expected {ndims}, found {data_array.ndim}."
        )

    current_dims = list(data_array.dims)

    # Identify lon/lat axes
    ident = AxisIdentifier(data_array)
    lon_dim = ident.longitude()
    lat_dim = ident.latitude()

    # Identify temporal axis (only one supported)
    time_dim = _identify_temporal_axis(data_array, current_dims)

    has_longitude = lon_dim is not None
    has_temporal = False

    special_dims = {lon_dim, lat_dim, time_dim}
    remaining_dims = [d for d in current_dims if d not in special_dims]

    final_dims: list[Hashable] = []

    # Slot 0: Longitude
    if has_longitude:
        final_dims.append(lon_dim)
    elif remaining_dims:
        final_dims.append(remaining_dims.pop(0))

    # Slot 1: Latitude
    if lat_dim is not None:
        final_dims.append(lat_dim)
    elif remaining_dims:
        final_dims.append(remaining_dims.pop(0))

    # Slot 2 : Time
    if ndims >= THREE_DIMENSIONS:
        if time_dim is not None:
            final_dims.append(time_dim)
            has_temporal = True
        elif remaining_dims:
            final_dims.append(remaining_dims.pop(0))

    # Fill remaining slots with whatever is left
    final_dims.extend(remaining_dims)

    # The identified axes must be dimensions of the variable, not auxiliary
    # coordinates such as the 2D longitudes and latitudes of a curvilinear
    # grid.
    if len(final_dims) != ndims:
        raise ValueError(
            f"Unable to identify the {ndims} grid axes of the variable: "
            f"found {tuple(final_dims)} for the dimensions "
            f"{tuple(current_dims)}. The longitude, latitude and time "
            "coordinates must be dimension coordinates."
        )

    return _DimInfo(
        data_array,
        tuple(final_dims),
        has_longitude,
        has_temporal,
        tuple(current_dims) != tuple(final_dims),
    )


def _check_coords(
    coords: dict[Hashable, Any],
    dims: tuple[Hashable, ...],
) -> None:
    """Check that the coordinates are those of the grid dimensions.

    Raises:
        TypeError: If coords is not a dict.
        IndexError: If the coordinates are not those of the dimensions.

    """
    if not isinstance(coords, dict):
        raise TypeError("coords must be an instance of dict")
    if len(coords) != len(dims):
        raise IndexError(
            f"Number of coordinates ({len(coords)}) doesn't match "
            f"number of dimensions ({len(dims)})"
        )
    unknown = set(coords) - set(dims)
    if unknown:
        raise IndexError(
            "axes not handled by this grid: "
            + ", ".join([str(item) for item in unknown])
        )


def _coords(
    coords: dict[Hashable, Any],
    dims: tuple[Hashable, ...],
    datetime64: tuple[Hashable, core.TemporalAxis] | None = None,
) -> tuple[tuple[NDArray1D | NDArray1DDateTime64, ...], tuple[int, ...]]:
    """Get the arguments to provide to grid interpolation functions.

    The coordinates can be scalars or array-like of any shape: they are
    broadcast against each other and flattened.

    Returns:
        The flattened coordinates, in the order of the grid dimensions, and
        the shape to give to the interpolated values.

    """
    _check_coords(coords, dims)
    values = np.broadcast_arrays(*(np.asarray(coords[dim]) for dim in dims))
    shape = values[0].shape
    return tuple(
        _cast_coordinate(dim, value.ravel(), datetime64)
        for dim, value in zip(dims, values, strict=True)
    ), shape


def _cast_coordinate(
    dim: Hashable,
    values: np.ndarray,
    datetime64: tuple[Hashable, core.TemporalAxis] | None,
) -> NDArray1D | NDArray1DDateTime64:
    """Cast 1D coordinates to the type expected by the axis of a grid."""
    if datetime64 is not None and dim == datetime64[0]:
        # Cast temporal coordinates
        dates = (
            values if values.dtype.kind == "M" else values.astype("datetime64")
        )
        return datetime64[1].cast_to_temporal_axis(
            cast("NDArray1DDateTime64", dates)
        )
    return cast("NDArray1D", values.astype(np.float64, copy=False))


class _GridHolder:
    """Base class for grid holders."""

    def __init__(
        self,
        grid: core.GridHolder,
        dims: tuple[Hashable, ...],
    ) -> None:
        """Initialize the grid holder."""
        self._dims = dims
        self._instance = grid
        self._datetime64: tuple[Hashable, core.TemporalAxis] | None = None
        if self._instance.has_temporal_axis:
            self._datetime64 = (
                dims[2],
                cast("core.TemporalAxis", self._instance.z),
            )

    def _outer(
        self,
        coords: dict[Hashable, Any],
        method: InterpolationMethods = "bilinear",
        *,
        bounds_error: bool = False,
        num_threads: int = 0,
    ) -> np.ndarray:
        """Interpolate on the grid defined by 1D coordinates.

        The points to interpolate at are all the combinations of the
        coordinates (outer product). Each coordinate is located once on its
        axis, which is much faster than interpolating the broadcast
        coordinates, and gives the same results.

        Args:
            coords: Mapping from dimension names to the 1D coordinates of the
                grid to interpolate at.
            method: Geometric interpolation method: ``bilinear``, ``idw`` or
                ``nearest``.
            bounds_error: If True, raise an error for a coordinate out of the
                grid.
            num_threads: Number of threads to use, 0 for all CPUs.

        Returns:
            The interpolated values, with one dimension per coordinate in the
            order of the dimensions of the grid.

        """
        _check_coords(coords, self._dims)
        ndim = len(self._dims)
        axes = tuple(
            _cast_coordinate(
                dim, np.asarray(coords[dim]).reshape(-1), self._datetime64
            )
            for dim in self._dims
        )
        config = _make_geometric_config(
            cast("Any", method),
            _GEOMETRIC_CONFIGS[ndim],
            bounds_error=bounds_error,
            num_threads=num_threads,
        )
        return _OUTER_FUNCTIONS[ndim](self._instance, *axes, config).reshape(
            tuple(item.size for item in axes), order="F"
        )

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Delegate attribute access to the underlying grid instance."""
        return getattr(self._instance, name)

    def __repr__(self) -> str:
        return repr(self._instance)

    def __reduce__(self) -> tuple:
        """Support pickle serialization (e.g. for Dask scatter/multiprocessing).

        Returns:
            Tuple of (callable, args) allowing reconstruction of this instance.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __reduce__"
        )


class Grid2D(_GridHolder):
    """Build a Grid2D from Xarray data.

    Create a 2D grid interpolation object from the provided Xarray data array.
    The longitude and latitude axes are identified from the CF attributes and
    the data is transposed to the (longitude, latitude) order if needed; a
    longitude axis is created as periodic (360 degrees).

    Args:
        data_array: Provided data

    Raises:
        ValueError: if the number of dimensions is different of 2.

    """

    def __init__(self, data_array: xr.DataArray) -> None:
        """Initialize the 2D grid from an Xarray data array."""
        object.__setattr__(self, "_data_array", data_array)
        canonical_dimensions = _get_canonical_dimensions(
            data_array, ndims=TWO_DIMENSIONS
        )
        grid = core.Grid(
            cast("core.Axis", canonical_dimensions.axis(0)),
            cast("core.Axis", canonical_dimensions.axis(1)),
            canonical_dimensions.data_array.values,
        )
        super().__init__(grid, canonical_dimensions.dims)

    def __reduce__(self) -> tuple:
        """Support pickle serialization (e.g. for Dask scatter/multiprocessing).

        Returns:
            Tuple of (callable, args) allowing reconstruction of this instance.

        """
        return (
            self.__class__,
            (self._data_array,),
        )

    def bivariate(
        self,
        coords: dict[Hashable, NDArray1DNumeric],
        method: InterpolationMethods = "bilinear",
        **kwargs: Any,  # noqa: ANN401
    ) -> np.ndarray:
        """Evaluate the interpolation defined for the given coordinates.

        Args:
            coords: Mapping from dimension names to the
                coordinates to interpolate. Coordinates can be scalars or
                array-like; they are broadcast against each other.
            method: Interpolation method. See
                :py:func:`pyinterp.bivariate` for more details.
            **kwargs: Additional keyword arguments provided to the
                interpolation method.

        Returns:
            The interpolated values, with the broadcast shape of the
            coordinates.

        Raises:
            IndexError: If coordinate dimensions don't match grid dimensions

        """
        (x, y), shape = _coords(coords, self._dims, self._datetime64)
        return bivariate(
            self._instance,
            cast("NDArray1DFloat64", x),
            cast("NDArray1DFloat64", y),
            method=method,
            **kwargs,
        ).reshape(shape)


class Grid3D(_GridHolder):
    """Build a Grid3D from Xarray data.

    Create a 3D grid interpolation object from the provided Xarray data array.
    Supports temporal axes via datetime64 coordinates.

    Args:
        data_array: Provided 3D data array

    Raises:
        ValueError: if the number of dimensions is different from 3.

    """

    def __init__(self, data_array: xr.DataArray) -> None:
        """Initialize the 3D grid from an Xarray data array."""
        object.__setattr__(self, "_data_array", data_array)
        canonical_dimensions = _get_canonical_dimensions(
            data_array, ndims=THREE_DIMENSIONS
        )

        grid = core.Grid(
            cast(
                "core.Axis",
                canonical_dimensions.axis(0),
            ),
            cast(
                "core.Axis",
                canonical_dimensions.axis(1),
            ),
            cast(
                "core.Axis | core.TemporalAxis",
                canonical_dimensions.axis(2),
            ),
            canonical_dimensions.data_array.values,
        )
        super().__init__(grid, canonical_dimensions.dims)

    def __reduce__(self) -> tuple:
        """Support pickle serialization (e.g. for Dask scatter/multiprocessing).

        Returns:
            Tuple of (callable, args) allowing reconstruction of this instance.

        """
        return (
            self.__class__,
            (object.__getattribute__(self, "_data_array"),),
        )

    def trivariate(
        self,
        coords: dict[Hashable, NDArray1DNumericWithTime],
        method: InterpolationMethods = "bilinear",
        **kwargs: Any,  # noqa: ANN401
    ) -> np.ndarray:
        """Evaluate the interpolation defined for the given coordinates.

        Args:
            coords: Mapping from dimension names to the
                coordinates to interpolate. Coordinates can be scalars or
                array-like; they are broadcast against each other.
                If the third axis is temporal, provide datetime64 array.
            method: Interpolation method. See
                :py:func:`pyinterp.trivariate` for more details.
            **kwargs: Additional keyword arguments provided to the
                interpolation method.

        Returns:
            The interpolated values, with the broadcast shape of the
            coordinates.

        Raises:
            IndexError: If coordinate dimensions don't match grid dimensions

        """
        (x, y, z), shape = _coords(coords, self._dims, self._datetime64)
        return trivariate(
            self._instance,
            cast("NDArray1DFloat64", x),
            cast("NDArray1DFloat64", y),
            cast("NDArray1DFloat64 | NDArray1DDateTime64", z),
            method=method,
            **kwargs,
        ).reshape(shape)


class Grid4D(_GridHolder):
    """Build a Grid4D from Xarray data.

    Create a 4D grid interpolation object from the provided Xarray data array.
    Supports temporal axes via datetime64 coordinates.

    Args:
        data_array: Provided 4D data array

    Raises:
        ValueError: if the number of dimensions is different from 4.

    """

    def __init__(self, data_array: xr.DataArray) -> None:
        """Initialize the 4D grid from an Xarray data array."""
        object.__setattr__(self, "_data_array", data_array)
        canonical_dimensions = _get_canonical_dimensions(
            data_array, ndims=FOUR_DIMENSIONS
        )

        grid = core.Grid(
            cast(
                "core.Axis",
                canonical_dimensions.axis(0),
            ),
            cast(
                "core.Axis",
                canonical_dimensions.axis(1),
            ),
            cast(
                "core.Axis | core.TemporalAxis",
                canonical_dimensions.axis(2),
            ),
            cast(
                "core.Axis",
                canonical_dimensions.axis(3),
            ),
            canonical_dimensions.data_array.values,
        )
        super().__init__(grid, canonical_dimensions.dims)

    def __reduce__(self) -> tuple:
        """Support pickle serialization (e.g. for Dask scatter/multiprocessing).

        Returns:
            Tuple of (callable, args) allowing reconstruction of this instance.

        """
        return (
            self.__class__,
            (object.__getattribute__(self, "_data_array"),),
        )

    def quadrivariate(
        self,
        coords: dict[Hashable, NDArray1DNumericWithTime],
        method: InterpolationMethods = "bilinear",
        **kwargs: Any,  # noqa: ANN401
    ) -> np.ndarray:
        """Evaluate the interpolation defined for the given coordinates.

        Args:
            coords: Mapping from dimension names to the
                coordinates to interpolate. Coordinates can be scalars or
                array-like; they are broadcast against each other.
                If the third axis is temporal, provide datetime64 array.
            method: Interpolation method. See
                :py:func:`pyinterp.quadrivariate` for more details.
            **kwargs: Additional keyword arguments provided to the
                interpolation method.

        Returns:
            The interpolated values, with the broadcast shape of the
            coordinates.

        Raises:
            IndexError: If coordinate dimensions don't match grid dimensions

        """
        (x, y, z, u), shape = _coords(coords, self._dims, self._datetime64)
        return quadrivariate(
            self._instance,
            cast("NDArray1DFloat64", x),
            cast("NDArray1DFloat64", y),
            cast("NDArray1DFloat64 | NDArray1DDateTime64", z),
            cast("NDArray1DFloat64", u),
            method=method,
            **kwargs,
        ).reshape(shape)


class RegularGridInterpolator:
    """Interpolate on a regular grid in arbitrary dimensions.

    Perform interpolation on a regular grid with uneven spacing support.
    Automatically detects geodetic coordinates (lon/lat) using CF conventions
    and temporal axes (datetime64).

    The data must be defined on a regular grid; the grid spacing however may be
    uneven. Linear, nearest neighbors, inverse distance weighting, and bicubic
    interpolation are supported.

    Args:
        array: The xarray DataArray defining the regular grid in ``n``
            dimensions. Must be 2D, 3D, or 4D.

    Raises:
        NotImplementedError: if the number of dimensions in the array is
            less than 2 or more than 4.

    Notes:
        **Automatic Detection:**

        The interpolator automatically detects:

        - **Geodetic coordinates**: If lon/lat are found via CF conventions
          (units attribute).
        - **Temporal axes**: If a coordinate has dtype='datetime64', it will
          be treated as a temporal axis with proper interpolation
        - **Dimension count**: Automatically selects Grid2D, Grid3D, or Grid4D

        **Geodetic Detection (CF Conventions):**

        Longitude axes are detected if the coordinate has units attribute
        matching: ``degrees_east``, ``degree_east``, ``degree_E``,
        ``degrees_E``, ``degreeE``, or ``degreesE``

        Latitude axes are detected if the coordinate has units attribute
        matching: ``degrees_north``, ``degree_north``, ``degree_N``,
        ``degrees_N``, ``degreeN``, or ``degreesN``

        **Temporal Detection:**

        Any coordinate with dtype containing 'datetime64' is automatically
        treated as a temporal axis.

    Examples:
        2D sea surface temperature

        >>> sst = xr.open_dataarray("sst.nc")  # (lon, lat)
        >>> interp = RegularGridInterpolator(sst)
        >>> result = interp(
        ...     {"lon": [10.5, 20.3], "lat": [45.2, -30.1]}, method="bilinear"
        ... )

        3D ocean temperature with depth

        >>> temp = xr.open_dataarray("temp.nc")  # (lon, lat, depth)
        >>> interp = RegularGridInterpolator(temp)
        >>> result = interp(
        ...     {"lon": [10.5], "lat": [45.2], "depth": [25.0]},
        ...     method="bilinear",
        ... )

        3D SST time series (automatic temporal handling)

        >>> sst_time = xr.open_dataarray("sst_time.nc")  # (lon, lat, time)
        >>> interp = RegularGridInterpolator(sst_time)
        >>> result = interp(
        ...     {
        ...         "lon": [10.5],
        ...         "lat": [45.2],
        ...         "time": np.array(["2020-01-01"], dtype="datetime64"),
        ...     },
        ...     method="bilinear",
        ... )

    """

    def __init__(self, array: xr.DataArray) -> None:
        """Initialize the interpolator from an Xarray data array.

        Args:
            array: The xarray DataArray to interpolate. Must be 2D, 3D, or 4D.

        Raises:
            NotImplementedError: If array is not 2D, 3D, or 4D.

        """
        ndim = len(array.shape)

        self._grid: Grid2D | Grid3D | Grid4D
        self._interp: Callable[..., Any]

        if ndim == TWO_DIMENSIONS:
            self._grid = Grid2D(array)
            self._interp = self._grid.bivariate
        elif ndim == THREE_DIMENSIONS:
            self._grid = Grid3D(array)
            self._interp = self._grid.trivariate
        elif ndim == FOUR_DIMENSIONS:
            self._grid = Grid4D(array)
            self._interp = self._grid.quadrivariate
        else:
            raise NotImplementedError(
                f"Only 2D, 3D, and 4D grids can be interpolated. "
                f"Got {ndim}D grid."
            )

    @property
    def ndim(self) -> int:
        """Get the number of array dimensions.

        Returns:
            Number of array dimensions (2, 3, or 4).

        """
        return len(self._grid._dims)

    @property
    def grid(self) -> Grid2D | Grid3D | Grid4D:
        """Get the instance handling the regular grid for interpolations.

        Returns:
            The underlying Grid2D, Grid3D, or Grid4D instance.

        """
        return self._grid

    def __call__(
        self,
        coords: dict,
        method: InterpolationMethods = "bilinear",
        **kwargs: Any,  # noqa: ANN401
    ) -> np.ndarray:
        """Interpolate at coordinates.

        Perform interpolation at the specified coordinates using the chosen
        method and parameters.

        Args:
            coords: Mapping from dimension names to the new coordinates.
                Coordinates can be scalars or array-like. For temporal axes,
                provide datetime64 arrays.
            method: The method of interpolation to perform. Supported methods
                depend on the grid type. Common methods include:

                - Geometric methods: ``nearest``, ``bilinear``, ``idw``
                - Windowed methods: ``akima``, ``akima_periodic``, ``bicubic``,
                  ``bilinear``, ``c_spline``, ``c_spline_not_a_knot``,
                  ``c_spline_periodic``, ``linear``, ``polynomial``,
                  ``steffen``.

            **kwargs: Additional keyword arguments passed to the interpolation
                function. Common options include:

                - ``bounds_error`` (bool): Raise error if coordinates are
                  out of bounds. Default: False (returns NaN).
                - ``num_threads`` (int): Number of threads for parallel
                  computation. 0 uses all CPUs. Default: 0.

                For windowed methods (bicubic, c_spline, etc.), additional
                options include:

                - ``half_window_size_x`` (int): Half window size in X direction
                - ``half_window_size_y`` (int): Half window size in Y direction
                - ``boundary_mode`` (str): Boundary handling mode
                  (``"shrink"``, ``"undef"``)

                For 3D/4D grids:

                - ``third_axis`` (str): Method for 3rd axis
                  (``"linear"``, ``"nearest"``)
                - ``fourth_axis`` (str): Method for 4th axis
                  (``"linear"``, ``"nearest"``)

        Returns:
            Interpolated values, with the broadcast shape of the
            coordinates.

        Raises:
            ValueError: If bounds_error=True and coordinates are out of bounds.
            IndexError: If coordinate dimensions don't match grid dimensions.

        Examples:
            Simple bilinear interpolation

            >>> result = interp(
            ...     {"lon": [10.5], "lat": [45.2]}, method="bilinear"
            ... )

            Bicubic with custom window size

            >>> result = interp(
            ...     {"lon": [10.5], "lat": [45.2]},
            ...     method="bicubic",
            ...     half_window_size_x=10,
            ...     half_window_size_y=10,
            ... )

            With bounds checking

            >>> result = interp(
            ...     {"lon": [10.5], "lat": [45.2]},
            ...     method="bilinear",
            ...     bounds_error=True,
            ... )

            Multi-threaded

            >>> result = interp(
            ...     {"lon": lon_array, "lat": lat_array},
            ...     method="bilinear",
            ...     num_threads=4,
            ... )

            3D with temporal axis

            >>> result = interp(
            ...     {
            ...         "lon": [10.5],
            ...         "lat": [45.2],
            ...         "time": np.array(["2020-01-01"], dtype="datetime64"),
            ...     },
            ...     method="bilinear",
            ... )

        """
        return self._interp(coords, method=method, **kwargs)


def _merge_indexers(
    coords: Mapping[Hashable, Any] | None,
    coords_kwargs: dict[str, Any],
    dims: tuple[Hashable, ...],
) -> tuple[dict[Hashable, Any], dict[str, Any]]:
    """Split the keyword arguments into indexers and interpolation options.

    A keyword argument named after a dimension of the variable is a
    coordinate to interpolate at; any other is an option of the interpolator.

    Returns:
        The indexers and the interpolation options.

    """
    indexers: dict[Hashable, Any] = dict(coords) if coords is not None else {}
    options: dict[str, Any] = {}
    for name, value in coords_kwargs.items():
        if name not in dims:
            options[name] = value
        elif name in indexers:
            raise ValueError(
                f"the coordinate {name!r} is given both in coords and as a "
                "keyword argument"
            )
        else:
            indexers[name] = value
    return indexers, options


def _as_indexer(
    name: Hashable,
    value: Any,  # noqa: ANN401
    source: xr.DataArray,
) -> xr.DataArray:
    """Convert a coordinate to interpolate at into a data array.

    The conversion follows :py:meth:`xarray.DataArray.interp`: a data array is
    kept as it is, a scalar gives a scalar coordinate and a one-dimensional
    array gives a new dimension named after the coordinate.

    Raises:
        ValueError: If the coordinate is a multidimensional array that is not
            a data array.

    """
    if not isinstance(value, xr.DataArray):
        array = np.asarray(value)
        if array.ndim > 1:
            raise ValueError(
                f"the coordinate {name!r} has {array.ndim} dimensions: give a "
                "multidimensional coordinate as a DataArray to name its "
                "dimensions"
            )
        value = (
            xr.DataArray(array)
            if array.ndim == 0
            else xr.DataArray(array, dims=(name,), coords={name: array})
        )
    # Dates given as strings are converted to the type of the axis, so that
    # the coordinates of the result are dates.
    if name in source.coords:
        dtype = source.coords[name].dtype
        if dtype.kind in "mM" and value.dtype.kind not in "mM":
            value = value.astype(dtype)
    return value


def _is_outer(
    indexers: dict[Hashable, xr.DataArray],
    method: InterpolationMethods,
    options: dict[str, Any],
) -> bool:
    """Return whether the coordinates define a grid to interpolate at.

    This is the case when each coordinate is a scalar or a 1D array along its
    own dimension, and the interpolation method is geometric.
    """
    return (
        method in _GEOMETRIC_METHODS
        and set(options) <= _OUTER_OPTIONS
        and all(
            item.ndim == 0 or item.dims == (dim,)
            for dim, item in indexers.items()
        )
    )


def _interp_outer(
    grid: Grid2D | Grid3D | Grid4D,
    indexers: dict[Hashable, xr.DataArray],
    template: xr.DataArray,
    method: InterpolationMethods,
    options: dict[str, Any],
) -> np.ndarray:
    """Interpolate on the grid defined by scalar or 1D coordinates.

    Returns:
        The interpolated values, with the dimensions of the template.

    """
    dims = grid._dims
    values = grid._outer(
        {dim: item.values for dim, item in indexers.items()},
        method,
        **options,
    )
    # The scalar coordinates give no dimension to the result
    kept = [dim for dim in dims if indexers[dim].ndim]
    values = values.squeeze(
        axis=tuple(ix for ix, dim in enumerate(dims) if not indexers[dim].ndim)
    )
    return values.transpose([kept.index(dim) for dim in template.dims])


def _interp_data_array(
    source: xr.DataArray,
    indexers: dict[Hashable, Any],
    method: InterpolationMethods,
    options: dict[str, Any],
) -> xr.DataArray:
    """Interpolate a data array at the given coordinates.

    Returns:
        The interpolated values, labelled with the coordinates they were
        interpolated at.

    Raises:
        ValueError: If a dimension of the variable has no coordinate to
            interpolate at, or if a coordinate is not a dimension.

    """
    unknown = [name for name in indexers if name not in source.dims]
    if unknown:
        raise ValueError(
            f"the coordinates {tuple(unknown)} are not dimensions of the "
            f"variable {source.dims}"
        )
    missing = [dim for dim in source.dims if dim not in indexers]
    if missing:
        raise ValueError(
            f"the dimensions {tuple(missing)} have no coordinate to "
            "interpolate at: every dimension of the variable must be "
            "interpolated"
        )
    # The indexers are taken in the order of the dimensions of the variable,
    # which sets the order of the dimensions of the result.
    converted = {
        dim: _as_indexer(dim, indexers[dim], source) for dim in source.dims
    }
    broadcast = dict(
        zip(converted, xr.broadcast(*converted.values()), strict=True)
    )
    template = next(iter(broadcast.values()))

    interpolator = RegularGridInterpolator(source)
    if _is_outer(converted, method, options):
        values = _interp_outer(
            interpolator.grid, converted, template, method, options
        )
    else:
        values = interpolator(
            {dim: item.values for dim, item in broadcast.items()},
            method=method,
            **options,
        )

    result = xr.DataArray(
        values,
        dims=template.dims,
        coords={
            name: coord
            for item in broadcast.values()
            for name, coord in item.coords.items()
        },
        name=source.name,
        attrs=dict(source.attrs),
    )
    # A coordinate interpolated at points of other dimensions (a scalar or a
    # data array) is attached to the result as a non-dimension coordinate.
    for dim, item in converted.items():
        if dim not in result.dims:
            result.coords[dim] = item.variable
    for dim in source.dims:
        if dim in source.coords:
            result.coords[dim].attrs = dict(source.coords[dim].attrs)
    # Coordinates that depend on none of the interpolated dimensions still
    # describe the result.
    for name, coord in source.coords.items():
        if name not in result.coords and not set(coord.dims) & set(
            source.dims
        ):
            result.coords[name] = coord
    return result


@xr.register_dataarray_accessor("pyinterp")
class DataArrayAccessor:
    """Interpolate a data array with ``da.pyinterp``.

    The accessor is registered as soon as both :py:mod:`pyinterp` and xarray
    are imported, in either order.

    Args:
        data_array: The data array to interpolate.

    """

    def __init__(self, data_array: xr.DataArray) -> None:
        """Initialize the accessor."""
        self._data_array = data_array

    def interp(
        self,
        coords: Mapping[Hashable, Any] | None = None,
        method: InterpolationMethods = "bilinear",
        **coords_kwargs: Any,  # noqa: ANN401
    ) -> xr.DataArray:
        """Interpolate the data array at new coordinates.

        This method is a faster replacement of
        :py:meth:`xarray.DataArray.interp` for variables defined on 2D, 3D or
        4D regular grids. As there, a coordinate can be:

        - a scalar, which gives a scalar coordinate of the result;
        - a one-dimensional array, which gives a dimension named after the
          coordinate: several such coordinates define a grid (outer product);
        - a :py:class:`xarray.DataArray`, whose dimensions become those of the
          result: coordinates sharing a dimension are interpolated point by
          point.

        Unlike :py:meth:`xarray.DataArray.interp`, every dimension of the
        variable must be interpolated. Longitudes, latitudes and dates are
        identified as in :py:class:`RegularGridInterpolator`: longitudes are
        periodic and dates can be given as strings.

        Args:
            coords: Mapping from dimension names to the coordinates to
                interpolate at.
            method: Interpolation method. See
                :py:meth:`RegularGridInterpolator.__call__` for the supported
                methods.
            **coords_kwargs: The coordinates to interpolate at, for keyword
                arguments named after a dimension of the variable, or options
                of the interpolation method, such as ``bounds_error``,
                ``num_threads`` or ``half_window_size_x``. Use ``coords`` for
                a dimension whose name is that of an option.

        Returns:
            The interpolated values, labelled with the coordinates they were
            interpolated at, with the name and attributes of the variable.

        Raises:
            ValueError: If a dimension of the variable has no coordinate to
                interpolate at, or if a coordinate is not a dimension.

        Note:
            The data are loaded into memory and copied into a new grid at each
            call. To interpolate the same variable many times, build a
            :py:class:`RegularGridInterpolator` once and reuse it.

        Examples:
            Interpolate on a new grid

            >>> result = da.pyinterp.interp(
            ...     lon=np.arange(-180, 180, 0.25),
            ...     lat=np.arange(-80, 80, 0.25),
            ...     method="bicubic",
            ... )

            Interpolate along a track

            >>> track = da.pyinterp.interp(
            ...     lon=xr.DataArray(lons, dims="obs"),
            ...     lat=xr.DataArray(lats, dims="obs"),
            ...     time=xr.DataArray(dates, dims="obs"),
            ... )

        """
        indexers, options = _merge_indexers(
            coords, coords_kwargs, self._data_array.dims
        )
        return _interp_data_array(self._data_array, indexers, method, options)


@xr.register_dataset_accessor("pyinterp")
class DatasetAccessor:
    """Interpolate the variables of a dataset with ``ds.pyinterp``.

    The accessor is registered as soon as both :py:mod:`pyinterp` and xarray
    are imported, in either order.

    Args:
        dataset: The dataset to interpolate.

    """

    def __init__(self, dataset: xr.Dataset) -> None:
        """Initialize the accessor."""
        self._dataset = dataset

    def interp(
        self,
        coords: Mapping[Hashable, Any] | None = None,
        method: InterpolationMethods = "bilinear",
        **coords_kwargs: Any,  # noqa: ANN401
    ) -> xr.Dataset:
        """Interpolate the variables of the dataset at new coordinates.

        The variables defined on exactly the interpolated dimensions are
        interpolated as with :py:meth:`DataArrayAccessor.interp`. The
        variables that depend on none of them are kept as they are. The others,
        such as the bounds of a longitude axis, are dropped.

        Args:
            coords: Mapping from dimension names to the coordinates to
                interpolate at.
            method: Interpolation method. See
                :py:meth:`RegularGridInterpolator.__call__` for the supported
                methods.
            **coords_kwargs: The coordinates to interpolate at, for keyword
                arguments named after a dimension of the dataset, or options
                of the interpolation method.

        Returns:
            A dataset holding the interpolated variables and the variables
            kept, with the attributes of the dataset.

        Raises:
            ValueError: If no variable is defined on exactly the interpolated
                dimensions.

        """
        dataset = self._dataset
        indexers, options = _merge_indexers(
            coords, coords_kwargs, tuple(dataset.dims)
        )
        interpolated = set(indexers)

        variables: dict[Hashable, xr.DataArray] = {}
        for name, variable in dataset.data_vars.items():
            dims = set(variable.dims)
            if dims == interpolated:
                variables[name] = _interp_data_array(
                    variable, indexers, method, options
                )
            elif not dims & interpolated:
                variables[name] = variable
        if not any(
            set(dataset[name].dims) == interpolated for name in variables
        ):
            raise ValueError(
                "no variable of the dataset is defined on the dimensions "
                f"{tuple(indexers)}"
            )
        return xr.Dataset(variables, attrs=dict(dataset.attrs))

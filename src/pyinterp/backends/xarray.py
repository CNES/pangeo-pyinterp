# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
XArray
------

Build interpolation objects from xarray.DataArray instances
"""
from typing import Dict, Hashable, Optional, Tuple, Union
import pickle

import numpy
import xarray as xr

from .. import cf, core, grid, interpolator

__all__ = ['Grid2D', 'Grid3D', 'Grid4D', 'RegularGridInterpolator']


class AxisIdentifier:
    """Identification of the axes defining longitudes, latitudes in a CF file.

    Args:
        data_array: The data array to be identified.
    """

    def __init__(self, data_array: xr.DataArray):
        self.data_array = data_array

    def _axis(self, units: cf.AxisUnit) -> Optional[str]:
        """Returns the name of the dimension that defines an axis.

        Args:
            units: The units of the axis

        Returns:
            The name of the coordinate
        """
        for name, coord in self.data_array.coords.items():
            if hasattr(coord, 'units') and coord.units in units:
                return name
        return None

    def longitude(self) -> Optional[str]:
        """Returns the name of the dimension that defines a longitude axis.

        Returns:
            The name of the longitude coordinate
        """
        return self._axis(cf.AxisLongitudeUnit())

    def latitude(self) -> Optional[str]:
        """Returns the name of the dimension that defines a latitude axis.

        Returns:
            The name of the latitude coordinates
        """
        return self._axis(cf.AxisLatitudeUnit())


def _dims_from_data_array(data_array: xr.DataArray,
                          geodetic: bool,
                          ndims: Optional[int] = 2) -> Tuple[str, str]:
    """Gets the name of the dimensions that define the grid axes. the
    longitudes and latitudes of the data array.

    Args:
        data_array: Provided data array
        geodetic: True, if the axes of the grid represent longitudes and
            latitudes otherwise Cartesian
        ndims: Number of dimension expected for the variable

    Returns:
        Longitude and latitude names

    Raises:
        ValueError if the provided data array doesn't define a
            longitude/latitude axis
        ValueError if the number of dimensions is different from the number of
            dimensions of the grid provided.
    """
    size = len(data_array.shape)
    if size != ndims:
        raise ValueError(
            'The number of dimensions of the variable is incorrect. Expected '
            f'{ndims}, found {size}.')

    if not geodetic:
        return tuple(data_array.coords)[:2]

    ident = AxisIdentifier(data_array)
    lon = ident.longitude()
    if lon is None:
        raise ValueError("The dataset doesn't define a longitude axis")
    lat = ident.latitude()
    if lat is None:
        raise ValueError("The dataset doesn't define a latitude axis")
    return lon, lat


def _coords(
    coords: dict,
    dims: Tuple,
    datetime64: Optional[Tuple[Hashable, core.TemporalAxis]] = None,
) -> Tuple:
    """Get the list of arguments to provide to the grid interpolation
    functions.

    Args:
        coords: Mapping from dimension names to the new coordinates. New
            coordinate can be a scalar, array-like.
        dims: List of dimensions handled by the grid
        datetime64: Properties of the axis used

    Returns:
        The tuple of arguments decoded.

    Raises:
        TypeError if coords are not one instance of ``dict``
        IndexError if the number of coordinates is different from the
            number of grid dimensions
        IndexError if one of the coordinates is not used by this grid.
    """
    if not isinstance(coords, dict):
        raise TypeError('coords must be an instance of dict')
    if len(coords) != len(dims):
        raise IndexError('too many indices for array')
    unknown = set(coords) - set(dims)
    if unknown:
        raise IndexError('axes not handled by this grid: ' +
                         ', '.join([str(item) for item in unknown]))
    # Is it necessary to manage a time axis?
    if datetime64 is not None:
        temporal_dim, temporal_axis = datetime64
        return tuple(coords[dim] if dim != temporal_dim else temporal_axis.
                     safe_cast(coords[temporal_dim]) for dim in dims)
    return tuple(coords[dim] for dim in dims)


class Grid2D(grid.Grid2D):
    """Builds a Grid2D from the Xarray data provided.

    Args:
        data_array: Provided data
        increasing_axes: If this is true, check that the grid axes are
            increasing: the decreasing axes and the supplied grid will be
            flipped. Default to ``False``.
        geodetic: True, if the axes of the grid represent longitudes and
            latitudes. In this case, the constructor will try to determine
            the axes of longitudes and latitudes according to the value of
            the attribute ``units`` using the following algorithm:

            * if the axis unit is one of the values of the set ``degrees_east``,
              ``degree_east``, ``degree_E``, ``degrees_E``, ``degreeE`` or
              ``degreesE`` the axis
              represents a longitude,
            * if the axis unit is one of the values of the set
              ``degrees_north``, ``degree_north``, ``degree_N``, ``degrees_N``
              or ``degreesN`` the axis represents a latitude.

            If this option is false, the axes will be considered Cartesian.
            Default to ``True``.

    Raises:
        ValueError: if the provided data array doesn't define a
            longitude/latitude axis if ``geodetic`` is True.
        ValueError: if the number of dimensions is different of 2.
    """

    def __init__(self,
                 data_array: xr.DataArray,
                 increasing_axes: bool = False,
                 geodetic: bool = True):
        self._dims = _dims_from_data_array(data_array, geodetic)
        super().__init__(
            core.Axis(data_array.coords[self._dims[0]].values,
                      is_circle=geodetic),
            core.Axis(data_array.coords[self._dims[1]].values),
            data_array.transpose(*self._dims).values,
            increasing_axes='inplace' if increasing_axes else None)

    def bivariate(self, coords: dict, *args, **kwargs) -> numpy.ndarray:
        """Evaluate the interpolation defined for the given coordinates.

        Args:
            coords: Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.bivariate <pyinterp.bivariate>`
            **kwargs: List of keywords arguments provided to the interpolation
                method :py:meth:`pyinterp.bivariate <pyinterp.bivariate>`

        Returns:
            The interpolated values.
        """
        return interpolator.bivariate(self, *_coords(coords, self._dims),
                                      *args, **kwargs)

    def bicubic(self, coords: dict, *args, **kwargs) -> numpy.ndarray:
        """Evaluate the interpolation defined for the given coordinates.

        Args:
            coords: Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic <pyinterp.bicubic>`.
            **kwargs: List of keyword arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic <pyinterp.bicubic>`.

        Returns:
            The interpolated values.
        """
        return interpolator.bicubic(self, *_coords(coords, self._dims), *args,
                                    **kwargs)


class Grid3D(grid.Grid3D):
    """Builds a Grid3D from the Xarray data provided.

    Args:
        data_array: Provided data array
        increasing_axes: If this is true, check that the grid axes are
            increasing: the decreasing axes and the supplied grid will be
            flipped. Default to ``False``.
        geodetic: True, if the axes of the grid represent longitudes and
            latitudes. In this case, the constructor will try to determine
            the axes of longitudes and latitudes according to the value of
            the attribute ``units`` using the following algorithm:

            * if the axis unit is one of the values of the set
              ``degrees_east``, ``degree_east``, ``degree_E``, ``degrees_E``,
              ``degreeE`` or ``degreesE`` the axis represents a longitude,
            * if the axis unit is one of the values of the set
              ``degrees_north``, ``degree_north``, ``degree_N``, ``degrees_N``
              or ``degreesN`` the axis represents a latitude.

            If this option is false, the axes will be considered Cartesian.
            Default to ``True``.

    Raises:
        ValueError: if the provided data array doesn't define a
            longitude/latitude axis if ``geodetic`` is True.
        ValueError: if the number of dimensions is different of 3.
    """

    def __init__(self,
                 data_array: xr.DataArray,
                 increasing_axes: bool = False,
                 geodetic: bool = True):
        x, y = _dims_from_data_array(data_array, geodetic, ndims=3)
        z = (set(data_array.dims) - {x, y}).pop()
        self._dims = (x, y, z)
        # Should the grid manage a time axis?
        dtype = data_array.coords[z].dtype
        if 'datetime64' in dtype.name or 'timedelta64' in dtype.name:
            self._datetime64 = z, core.TemporalAxis(
                data_array.coords[z].values)
        else:
            self._datetime64 = None
        super().__init__(
            core.Axis(data_array.coords[x].values, is_circle=geodetic),
            core.Axis(data_array.coords[y].values),
            core.Axis(data_array.coords[z].values)
            if self._datetime64 is None else self._datetime64[1],
            data_array.transpose(x, y, z).values,
            increasing_axes='inplace' if increasing_axes else None)

    def trivariate(self, coords: dict, *args, **kwargs) -> numpy.ndarray:
        """Evaluate the interpolation defined for the given coordinates.

        Args:
            coords: Mapping from dimension names to the coordinates to
                interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.trivariate
                <pyinterp.trivariate>`.
            **kwargs: List of keywords arguments provided to the interpolation
                method :py:meth:`pyinterp.trivariate
                <pyinterp.trivariate>`.

        Returns:
            The interpolated values.
        """
        return interpolator.trivariate(
            self, *_coords(coords, self._dims, self._datetime64), *args,
            **kwargs)

    def bicubic(self, coords: dict, *args, **kwargs) -> numpy.ndarray:
        """Evaluate the interpolation defined for the given coordinates.

        Args:
            coords: Mapping from dimension names to the coordinates to
                interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic <pyinterp.bicubic>`.
            **kwargs: List of keyword arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic <pyinterp.bicubic>`.

        Returns:
            The interpolated values.
        """
        return interpolator.bicubic(
            self, *_coords(coords, self._dims, self._datetime64), *args,
            **kwargs)


class Grid4D(grid.Grid4D):
    """Builds a Grid4D from the Xarray data provided.

    Args:
        data_array: Provided data array.
        increasing_axes: If this is true, check that the grid axes are
            increasing: the decreasing axes and the supplied grid will be
            flipped. Default to ``False``.
        geodetic: True, if the axes of the grid represent longitudes and
            latitudes. In this case, the constructor will try to determine the
            axes of longitudes and latitudes according to the value of the
            attribute ``units`` using the following algorithm:

            * if the axis unit is one of the values of the set
              ``degrees_east``, ``degree_east``, ``degree_E``, ``degrees_E``,
              ``degreeE`` or ``degreesE`` the axis represents a longitude,
            * if the axis unit is one of the values of the set
              ``degrees_north``, ``degree_north``, ``degree_N``, ``degrees_N``
              or ``degreesN`` the axis represents a latitude.

            If this option is false, the axes will be considered Cartesian.
            Default to ``True``.

    Raises:
        ValueError: if the provided data array doesn't define a
            longitude/latitude axis if ``geodetic`` is True.
        ValueError: if the number of dimensions is different of 4.
    """

    def __init__(self,
                 data_array: xr.DataArray,
                 increasing_axes: bool = False,
                 geodetic: bool = True):
        x, y = _dims_from_data_array(data_array, geodetic, ndims=4)
        z, u = tuple(set(data_array.dims) - {x, y})

        # Should the grid manage a time axis?
        self._datetime64 = None
        dtype = data_array.coords[z].dtype
        if 'datetime64' in dtype.name:
            self._datetime64 = z, core.TemporalAxis(
                data_array.coords[z].values)
        dtype = data_array.coords[u].dtype
        if 'datetime64' in dtype.name:
            if self._datetime64 is not None:
                raise ValueError('unable to handle two time axes')
            self._datetime64 = u, core.TemporalAxis(
                data_array.coords[u].values)
            # The time axis is the Z axis.
            z, u = u, z

        # Names of the dimensions in the order of the tensor handled by the
        # library
        self._dims = (x, y, z, u)

        super().__init__(
            core.Axis(data_array.coords[x].values, is_circle=geodetic),
            core.Axis(data_array.coords[y].values),
            core.Axis(data_array.coords[z].values)
            if self._datetime64 is None else self._datetime64[1],
            core.Axis(data_array.coords[u].values),
            data_array.transpose(x, y, z, u).values,
            increasing_axes='inplace' if increasing_axes else None)

    def quadrivariate(self, coords: dict, *args, **kwargs) -> numpy.ndarray:
        """Evaluate the interpolation defined for the given coordinates.

        Args:
            coords: Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.quadrivariate
                <pyinterp.quadrivariate>`.
            **kwargs: List of keywords arguments provided to the interpolation
                method :py:meth:`pyinterp.quadrivariate
                <pyinterp.quadrivariate>`.

        Returns:
            The interpolated values.
        """
        return interpolator.quadrivariate(
            self, *_coords(coords, self._dims, self._datetime64), *args,
            **kwargs)

    def bicubic(self, coords: dict, *args, **kwargs) -> numpy.ndarray:
        """Evaluate the interpolation defined for the given coordinates.

        Args:
            coords: Mapping from dimension names to the coordinates to
                interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic <pyinterp.bicubic>`.
            **kwargs: List of keyword arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic <pyinterp.bicubic>`.

        Returns:
            The interpolated values.
        """
        return interpolator.bicubic(
            self, *_coords(coords, self._dims, self._datetime64), *args,
            **kwargs)


class RegularGridInterpolator:
    """Interpolation on a regular grid in arbitrary dimensions.

    The data must be defined on a regular grid; the grid spacing however may be
    uneven.  Linear, nearest neighbors, inverse distance weighting and bicubic
    interpolation are supported.

    Args:
        array: The array defining the regular grid in ``n`` dimensions.
        increasing_axes: If this is true, check that the grid axes are
            increasing: the decreasing axes and the supplied grid will be
            flipped. Default to ``False``.
        geodetic: True, if the axes of the grid represent longitudes and
            latitudes. In this case, the constructor will try to determine the
            axes of longitudes and latitudes according to the value of the
            attribute ``units`` using the following algorithm:

            * if the axis unit is one of the values of the set ``degrees_east``,
              ``degree_east``, ``degree_E``,
              ``degrees_E``, ``degreeE`` or ``degreesE`` the axis represents a
              longitude,
            * if the axis unit is one of the values of the set
              ``degrees_north``, ``degree_north``, ``degree_N``, ``degrees_N``
              or ``degreesN`` the axis represents a latitude.

            If this option is false, the axes will be considered Cartesian.
            Default to ``True``.

    Raises:
        ValueError: if the provided data array doesn't define a
            longitude/latitude axis if ``geodetic`` is True.
        NotImplementedError: if the number of dimensions in the array is
            less than 2 or more than 4.
    """

    def __init__(self,
                 array: xr.DataArray,
                 increasing_axes: bool = True,
                 geodetic: bool = True):
        if len(array.shape) == 2:
            self._grid = Grid2D(array,
                                increasing_axes=increasing_axes,
                                geodetic=geodetic)
            self._interp = self._grid.bivariate
        elif len(array.shape) == 3:
            self._grid = Grid3D(array,
                                increasing_axes=increasing_axes,
                                geodetic=geodetic)
            self._interp = self._grid.trivariate
        elif len(array.shape) == 4:
            self._grid = Grid4D(array,
                                increasing_axes=increasing_axes,
                                geodetic=geodetic)
            self._interp = self._grid.quadrivariate
        else:
            raise NotImplementedError(
                'Only the 2D, 3D or 4D grids can be interpolated.')

    def __getstate__(self) -> Tuple[bytes]:
        # Walk around a bug with pybind11 and pickle starting with Python 3.9
        # Serialize the object here with highest protocol.
        return (pickle.dumps((self._grid, self._interp),
                             protocol=pickle.HIGHEST_PROTOCOL), )

    def __setstate__(self, state: Tuple[bytes]) -> None:
        # Walk around a bug with pybind11 and pickle starting with Python 3.9
        # Deserialize the object here with highest protocol.
        self._grid, self._interp = pickle.loads(state[0])

    @property
    def ndim(self) -> int:
        """Gets the number of array dimensions.

        Returns:
            Number of array dimensions.
        """
        return self._grid.array.ndim

    @property
    def grid(self) -> Union[Grid2D, Grid3D, Grid4D]:
        """Gets the instance of handling the regular grid for interpolations.

        Returns:
            The regular grid.
        """
        return self._grid

    def __call__(self,
                 coords: Dict,
                 method: str = 'bilinear',
                 bounds_error: bool = False,
                 bicubic_kwargs: Optional[Dict] = None,
                 num_threads: int = 0) -> numpy.ndarray:
        """Interpolation at coordinates.

        Args:
            coords: Mapping from dimension names to the new coordinates.
                New coordinate can be an scalar, array-like.
            method: The method of interpolation to perform. Supported are
                ``bicubic``, ``bilinear``, ``nearest``, and
                ``inverse_distance_weighting``. Default to ``bilinear``.
            bounds_error: If True, when interpolated values are requested
                outside of the domain of the input data, a
                :py:class:`ValueError` is raised. If False, then `nan` is used.
            bicubic_kwargs: A dictionary of keyword arguments to pass on to the
                :py:func:`bicubic <pyinterp.bicubic>` function. This is useful
                to control the parameters of this interpolator: window size in
                x, y and the edge control of the calculation windows.
            num_threads: The number of threads to use for the computation. If 0
                all CPUs are used. If 1 is given, no parallel computing code is
                used at all, which is useful for debugging. Defaults to ``0``.
        Returns:
            New array on the new coordinates.
        """
        if method == 'bicubic':
            bicubic_kwargs = bicubic_kwargs or {}
            return self._grid.bicubic(coords,
                                      bounds_error=bounds_error,
                                      num_threads=num_threads,
                                      **bicubic_kwargs)
        return self._interp(coords,
                            interpolator=method,
                            bounds_error=bounds_error,
                            num_threads=num_threads)

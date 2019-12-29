# Copyright (c) 2020 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
XArray
------

Build interpolation objects from xarray.DataArray instances
"""
from typing import Iterable, Optional, Tuple, Union
import numpy as np
import xarray as xr
from .. import axis
from .. import cf
from .. import core
from .. import grid
from .. import interpolator

__all__ = ['Grid2D', 'Grid3D', 'Grid4D']


class AxisIdentifier:
    """Identification of the axes defining longitudes, latitudes in a
    CF file."""
    def __init__(self, data_array: xr.DataArray):
        self.data_array = data_array

    def _axis(self, units: cf.AxisUnit) -> Optional[str]:
        """Returns the name of the dimension that defines an axis.

        Args:
            array : xarray.DataArray
                The array defining the regular grid in n dimensions.

        Return:
            str, optional:
                The name of the coordinate
        """
        for name, coord in self.data_array.coords.items():
            if hasattr(coord, 'units') and coord.units in units:
                return name
        return None

    def longitude(self) -> Optional[str]:
        """Returns the name of the dimension that defines a longitude
        axis.

        Args:
            array : xarray.DataArray
                The array defining the regular grid in n dimensions.

        Return:
            str, optional:
                The name of the longitude coordinate
        """
        return self._axis(cf.AxisLongitudeUnit())

    def latitude(self) -> Optional[str]:
        """Returns the name of the dimension that defines a latitude
        axis.

        Args:
            array : xarray.DataArray
                The array defining the regular grid in n dimensions.

        Return:
            str, optional:
                The name of the latitude coordinate
        """
        return self._axis(cf.AxisLatitudeUnit())


def _dims_from_data_array(data_array: xr.DataArray,
                          geodetic: bool,
                          ndims: Optional[int] = 2) -> Tuple[str, str]:
    """
    Gets the name of the dimensions that define the grid axes.
    the longitudes and latitudes
    of the data array.

    Args:
        data_array (xarray.DataArray): Provided data array
        geodetic (bool): True, if the axes of the grid represent
            longitudes and latitudes otherwise Cartesian
        ndims (int, optional): Number of dimension expected for the variable

    Return:
        tuple: longitude and latitude names

    Raises:
        ValueError if the provided data array doesn't define a
            longitude/latitude axis
        ValueError if the number of dimensions is different from the number of
            dimensions of the grid provided.
    """
    size = len(data_array.shape)
    if size != ndims:
        raise ValueError(
            "The number of dimensions of the variable is incorrect. Expected "
            f"{ndims}, found {size}.")

    if not geodetic:
        return tuple(data_array.coords)[:2]

    ident = AxisIdentifier(data_array)
    lon = ident.longitude()
    if lon is None:
        raise ValueError("The dataset doesn't define a longitude axis")
    lat = ident.latitude()
    if lat is None:
        raise ValueError("The dataset doesn't define a longitude axis")
    return lon, lat


def _coords(coords: dict,
            dims: Tuple,
            datetime64: Optional[Tuple[str, axis.TemporalAxis]] = None
            ) -> Tuple:
    """
    Get the list of arguments to provide to the grid interpolation
    functions.

    Args:
        coords (dict): Mapping from dimension names to the
            new coordinates. New coordinate can be an scalar, array-like.
        dims (tuple): List of dimensions handled by the grid
        datetime64 (tuple, optional): Properties of the axis used

    Return:
        tuple: the tuple of arguments decoded.

    Raises:
        TypeError if coords is not on instance of ``dict``
        IndexError if the number of coordinates is different from the
            number of grid dimensions
        IndexError if one of the coordinates is not used by this grid.
    """
    if not isinstance(coords, dict):
        raise TypeError("coords must be an instance of dict")
    if len(coords) != len(dims):
        raise IndexError("too many indices for array")
    unknown = set(coords) - set(dims)
    if unknown:
        raise IndexError("axes not handled by this grid: " +
                         ", ".join([str(item) for item in unknown]))
    # Is it necessary to manage a time axis?
    if datetime64 is not None:
        dim, axis = datetime64
        coords[dim] = axis.safe_cast(coords[dim])
    return tuple(coords[dim] for dim in dims)


class Grid2D(grid.Grid2D):
    """Builds a Grid2D from the Xarray data provided.
    """
    def __init__(self,
                 data_array: xr.DataArray,
                 increasing_axes: bool = False,
                 geodetic: bool = True):
        """
        Initialize a new 2D Cartesian Grid.

        Args:
            data_array (xarray.DataArray): Provided data
            increasing_axes (bool, optional): If this is true, check that the
                grid axes are increasing: the decreasing axes and the supplied
                grid will be flipped. Default to ``False``.
            geodetic (bool, optional): True, if the axes of the grid represent
                longitudes and latitudes. In this case, the constructor will
                try to determine the axes of longitudes and latitudes according
                to the value of the attribute ``units`` using the following
                algorithm:

                * if the axis unit is one of the values of the set
                  ``degrees_east``, ``degree_east``, "degree_E``,
                  ``degrees_E``, ``degrees_E``, ``degreeE`` or ``degreesE``
                  the axis represents a longitude,
                * if the axis unit is one of the values of the set
                  ``degrees_north``, ``degree_north``, ``degree_N``,
                  ``degree_N``, ``degrees_N`` or ``degreesN`` the axis
                  represents a latitude.

                If this option is false, the axes will be considered Cartesian.
                Default to ``True``.

        Raises:
            ValueError: if the provided data array doesn't define a
                longitude/latitude axis if ``geodetic`` is True
            ValueError: if the number of dimensions is different of 2.
        """

        self._dims = _dims_from_data_array(data_array, geodetic)
        super(Grid2D, self).__init__(
            core.Axis(data_array.coords[self._dims[0]].values,
                      is_circle=geodetic),
            core.Axis(data_array.coords[self._dims[1]].values),
            data_array.transpose(*self._dims).values,
            increasing_axes='inplace' if increasing_axes else None)

    def bivariate(self, coords: dict, *args, **kwargs) -> np.ndarray:
        """Evaluate the interpolation defined for the given coordinates

        Args:
            coords (dict): Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.bivariate <pyinterp.bivariate>`
            **kwargs: List of keywords arguments provided to the interpolation
                method :py:meth:`pyinterp.bivariate <pyinterp.bivariate>`

        Return:
            np.ndarray: the interpolated values
        """
        return interpolator.bivariate(self, *_coords(coords, self._dims),
                                      *args, **kwargs)

    def bicubic(self, coords: dict, *args, **kwargs) -> np.ndarray:
        """Evaluate the interpolation defined for the given coordinates

        Args:
            coords (dict): Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic <pyinterp.bicubic>`
            **kwargs: List of keyword arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic <pyinterp.bicubic>`

        Return:
            np.ndarray: the interpolated values
        """
        return interpolator.bicubic(self, *_coords(coords, self._dims), *args,
                                    **kwargs)


class Grid3D(grid.Grid3D):
    """Builds a Grid3D from the Xarray data provided.
    """
    def __init__(self,
                 data_array: xr.DataArray,
                 increasing_axes: bool = False,
                 geodetic: bool = True):
        """

        Args:
            data_array (xarray.DataArray): Provided data array
            increasing_axes (bool, optional): If this is true, check that the
                grid axes are increasing: the decreasing axes and the supplied
                grid will be flipped. Default to ``False``.
            geodetic (bool, optional): True, if the axes of the grid represent
                longitudes and latitudes. In this case, the constructor will
                try to determine the axes of longitudes and latitudes according
                to the value of the attribute ``units`` using the following
                algorithm:

                * if the axis unit is one of the values of the set
                  ``degrees_east``, ``degree_east``, "degree_E``,
                  ``degrees_E``, ``degrees_E``, ``degreeE`` or ``degreesE``
                  the axis represents a longitude,
                * if the axis unit is one of the values of the set
                  ``degrees_north``, ``degree_north``, ``degree_N``,
                  ``degree_N``, ``degrees_N`` or ``degreesN`` the axis
                  represents a latitude.

                If this option is false, the axes will be considered Cartesian.
                Default to ``True``.

        Raises:
            ValueError: if the provided data array doesn't define a
                longitude/latitude axis if ``geodetic`` is True
            ValueError: if the number of dimensions is different of 3.
        """
        x, y = _dims_from_data_array(data_array, geodetic, ndims=3)
        z = (set(data_array.dims) - {x, y}).pop()
        self._dims = (x, y, z)
        # Should the grid manage a time axis?
        dtype = data_array.coords[z].dtype
        if "datetime64" in dtype.name:
            self._datetime64 = z, axis.TemporalAxis(
                data_array.coords[z].values)
        else:
            self._datetime64 = None
        super(Grid3D, self).__init__(
            core.Axis(data_array.coords[x].values, is_circle=geodetic),
            core.Axis(data_array.coords[y].values),
            core.Axis(data_array.coords[z].values)
            if self._datetime64 is None else self._datetime64[1],
            data_array.transpose(x, y, z).values,
            increasing_axes='inplace' if increasing_axes else None)

    def trivariate(self, coords: dict, *args, **kwargs) -> np.ndarray:
        """Evaluate the interpolation defined for the given coordinates

        Args:
            coords (dict): Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.trivariate
                <pyinterp.trivariate>`
            **kwargs: List of keywords arguments provided to the interpolation
                method :py:meth:`pyinterp.trivariate
                <pyinterp.trivariate>`

        Return:
            np.ndarray: the interpolated values
        """
        return interpolator.trivariate(
            self, *_coords(coords, self._dims, self._datetime64), *args,
            **kwargs)

    def bicubic(self, coords: dict, *args, **kwargs) -> np.ndarray:
        """Evaluate the interpolation defined for the given coordinates

        Args:
            coords (dict): Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic <pyinterp.bicubic>`
            **kwargs: List of keyword arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic <pyinterp.bicubic>`

        Return:
            np.ndarray: the interpolated values
        """
        return interpolator.bicubic(
            self, *_coords(coords, self._dims, self._datetime64), *args,
            **kwargs)


class Grid4D(grid.Grid4D):
    """Builds a Grid4D from the Xarray data provided.
    """
    def __init__(self,
                 data_array: xr.DataArray,
                 increasing_axes: bool = False,
                 geodetic: bool = True):
        """

        Args:
            data_array (xarray.DataArray): Provided data array
            increasing_axes (bool, optional): If this is true, check that the
                grid axes are increasing: the decreasing axes and the supplied
                grid will be flipped. Default to ``False``.
            geodetic (bool, optional): True, if the axes of the grid represent
                longitudes and latitudes. In this case, the constructor will
                try to determine the axes of longitudes and latitudes according
                to the value of the attribute ``units`` using the following
                algorithm:

                * if the axis unit is one of the values of the set
                  ``degrees_east``, ``degree_east``, "degree_E``,
                  ``degrees_E``, ``degrees_E``, ``degreeE`` or ``degreesE``
                  the axis represents a longitude,
                * if the axis unit is one of the values of the set
                  ``degrees_north``, ``degree_north``, ``degree_N``,
                  ``degree_N``, ``degrees_N`` or ``degreesN`` the axis
                  represents a latitude.

                If this option is false, the axes will be considered Cartesian.
                Default to ``True``.

        Raises:
            ValueError: if the provided data array doesn't define a
                longitude/latitude axis if ``geodetic`` is True
            ValueError: if the number of dimensions is different of 4.
        """
        x, y = _dims_from_data_array(data_array, geodetic, ndims=4)
        z, u = tuple(set(data_array.dims) - {x, y})
        self._dims = (x, y, z, u)
        z_values = data_array.coords[z].values
        u_values = data_array.coords[u].values

        # Should the grid manage a time axis?
        self._datetime64 = None
        dtype = data_array.coords[z].dtype
        if "datetime64" in dtype.name:
            self._datetime64 = z, axis.TemporalAxis(
                data_array.coords[z].values)
        dtype = data_array.coords[u].dtype
        if "datetime64" in dtype.name:
            if self._datetime64 is not None:
                raise ValueError("unable to handle two time axes")
            self._datetime64 = u, axis.TemporalAxis(
                data_array.coords[u].values)
            # The time axis is the Z axis.
            z, u = u, z
        super(Grid4D, self).__init__(
            core.Axis(data_array.coords[x].values, is_circle=geodetic),
            core.Axis(data_array.coords[y].values),
            core.Axis(data_array.coords[z].values)
            if self._datetime64 is None else self._datetime64[1],
            core.Axis(data_array.coords[u].values),
            data_array.transpose(x, y, z, u).values,
            increasing_axes='inplace' if increasing_axes else None)

    def quadrivariate(self, coords: dict, *args, **kwargs) -> np.ndarray:
        """Evaluate the interpolation defined for the given coordinates

        Args:
            coords (dict): Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.quadrivariate
                <pyinterp.quadrivariate>`
            **kwargs: List of keywords arguments provided to the interpolation
                method :py:meth:`pyinterp.quadrivariate
                <pyinterp.quadrivariate>`

        Return:
            np.ndarray: the interpolated values
        """
        return interpolator.quadrivariate(
            self, *_coords(coords, self._dims, self._datetime64), *args,
            **kwargs)

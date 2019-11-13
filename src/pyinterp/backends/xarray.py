# Copyright (c) 2019 CNES
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
from .. import cf
from .. import core
from .. import grid
from .. import interpolator

__all__ = ['Grid2D', 'Grid3D']


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


def _lon_lat_from_data_array(data_array: xr.DataArray,
                             ndims: Optional[int] = 2) -> Tuple[str, str]:
    """
    Gets the name of the dimensions that define the longitudes and latitudes
    of the data array.

    Args:
        data_array (xarray.DataArray): Provided data array
        ndims (int, optional): Number of dimension expected for the variable

    Return:
        tuple: longitude and latitude names

    Raises:
        ValueError if the provided data array doesn't define a
            longitude/latitude axis
        ValueError if the number of dimensions is different from the number of
            dimensions of the grid provided.
    """
    ident = AxisIdentifier(data_array)
    lon = ident.longitude()
    if lon is None:
        raise ValueError("The dataset doesn't define a longitude axis")
    lat = ident.latitude()
    if lat is None:
        raise ValueError("The dataset doesn't define a longitude axis")
    size = len(data_array.shape)
    if size != ndims:
        raise ValueError(
            "The number of dimensions of the variable is incorrect. Expected "
            f"{ndims}, found {size}.")
    return lon, lat


def _coords(coords: dict,
            dims: Iterable,
            datetime64: Optional[Tuple[str, np.dtype]] = None) -> Tuple:
    """
    Get the list of arguments to provide to the grid interpolation
    functions.

    Args:
        coords (dict): Mapping from dimension names to the
            new coordinates. New coordinate can be an scalar, array-like.
        dims (iterable): List of dimensions handled by the grid
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
        # In this case, it's checked that the unit between the time axis and
        # the data provided is identical.
        dim, dtype = datetime64
        if coords[dim].dtype != dtype:
            raise ValueError(
                f"the unit ({dtype!s}) of the time axis ({dim}) is different "
                f"from the time unit provided: {coords[dim].dtype!s}")
        coords[dim] = coords[dim].astype("float64")
    return tuple(coords[dim] for dim in dims)


class Grid2D(grid.Grid2D):
    """Builds a Grid2D from the Xarray data provided.

    Args:
        data_array (xarray.DataArray): Provided data
        increasing_axes (bool, optional): If this is true, check that the grid
            axes are increasing: the decreasing axes and the supplied grid will
            be flipped. Default to ``False``.
    """
    def __init__(self,
                 data_array: xr.DataArray,
                 increasing_axes: Optional[bool] = False):
        self._dims = _lon_lat_from_data_array(data_array)
        super(Grid2D, self).__init__(
            core.Axis(data_array.coords[self._dims[0]].values, is_circle=True),
            core.Axis(data_array.coords[self._dims[1]].values),
            data_array.transpose(*self._dims).values,
            increasing_axes='inplace' if increasing_axes else None)

    def bivariate(self, coords: dict, *args, **kwargs) -> np.ndarray:
        """Evaluate the interpolation defined for the given coordinates

        Args:
            coords (dict): Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.bivariate
                <pyinterp.interpolator.bivariate.bivariate>`
            **kwargs: List of keywords arguments provided to the interpolation
                method :py:meth:`pyinterp.bivariate
                <pyinterp.interpolator.bivariate.bivariate>`

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
                method :py:meth:`pyinterp.bicubic
                <pyinterp.interpolator.bicubic.bicubic>`
            **kwargs: List of keyword arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic
                <pyinterp.interpolator.bicubic.bicubic>`

        Return:
            np.ndarray: the interpolated values
        """
        return interpolator.bicubic(self, *_coords(coords, self._dims), *args,
                                    **kwargs)


class Grid3D(grid.Grid3D):
    """Builds a Grid3D from the Xarray data provided.

    Args:
        data_array (xarray.DataArray): Provided data array
        increasing_axes (bool, optional): If this is true, check that the grid
            axes are increasing: the decreasing axes and the supplied grid will
            be flipped. Default to ``False``.
    """
    def __init__(self,
                 data_array: xr.DataArray,
                 increasing_axes: Optional[bool] = False):
        x, y = _lon_lat_from_data_array(data_array, ndims=3)
        z = (set(data_array.dims) - {x, y}).pop()
        self._dims = (x, y, z)
        # If the grid has a time axis, its properties are stored in order to
        # check the consistency between the time axis and the data provided
        # during interpolation.
        dtype = data_array.coords[z].dtype
        self._datetime64 = (z, dtype) if "datetime64" in dtype.name else None
        super(Grid3D, self).__init__(
            core.Axis(data_array.coords[x].values, is_circle=True),
            core.Axis(data_array.coords[y].values),
            core.Axis(data_array.coords[z].astype("float64") if self.
                      _datetime64 else data_array.coords[z].values),
            data_array.transpose(x, y, z).values,
            increasing_axes='inplace' if increasing_axes else None)

    def time_unit(self) -> Optional[np.dtype]:
        """Gets the time units handled by this instance

        Return:
            np.dtype, optional: The unity of the temporal axis or None if
            the third dimension of this instance does not represent a time.
        """
        if self._datetime64:
            return self._datetime64[1]

    def trivariate(self, coords: dict, *args, **kwargs) -> np.ndarray:
        """Evaluate the interpolation defined for the given coordinates

        Args:
            coords (dict): Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.trivariate
                <pyinterp.interpolator.trivariate.trivariate>`
            **kwargs: List of keywords arguments provided to the interpolation
                method :py:meth:`pyinterp.trivariate
                <pyinterp.interpolator.trivariate.trivariate>`

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
                method :py:meth:`pyinterp.bicubic
                <pyinterp.interpolator.bicubic.bicubic>`
            **kwargs: List of keyword arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic
                <pyinterp.interpolator.bicubic.bicubic>`

        Return:
            np.ndarray: the interpolated values
        """
        return interpolator.bicubic(
            self, *_coords(coords, self._dims, self._datetime64), *args,
            **kwargs)

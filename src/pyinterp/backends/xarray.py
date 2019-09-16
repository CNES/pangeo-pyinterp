# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
XArray
------

Build interpolation objects from XArray Dataset instances
"""
from typing import Iterable, Optional, Tuple
import xarray as xr
from .. import cf
from .. import core
from .. import grid
from .. import bivariate
from .. import bicubic
from .. import trivariate


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

        Returns
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

        Returns
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

        Returns
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

    Returns:
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


def _coords(coords: dict, dims: Iterable):
    """
    Get the list of arguments to provide to the grid interpolation
    functions.

    Args:
        coords (dict): Mapping from dimension names to the
            new coordinates. New coordinate can be an scalar, array-like.
        dims (iterable): List of dimensions handled by the grid

    Returns:
        The list of arguments decoded.

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
    return tuple(coords[dim] for dim in dims)


class Grid2D(grid.Grid2D):
    """Builds a Grid2D from the Xarray data provided.

    Args:
        data_array (xarray.DataArray): Provided data
    """
    def __init__(self, data_array: xr.DataArray):
        self._dims = _lon_lat_from_data_array(data_array)
        super(Grid2D, self).__init__(
            core.Axis(data_array.coords[self._dims[0]].values, is_circle=True),
            core.Axis(data_array.coords[self._dims[1]].values),
            data_array.transpose(*self._dims).values)

    def bivariate(self, coords: dict, *args, **kwargs):
        """Evaluate the interpolation defined for the given coordinates

        Args:
            coords (dict): Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.bivariate.bivariate`
            **kwargs: List of keywords arguments provided to the interpolation
                method :py:meth:`pyinterp.bivariate.bivariate`

        Returns:
            The interpolated values
        """
        return bivariate.bivariate(self, *_coords(coords, self._dims), *args,
                                   **kwargs)

    def bicubic(self, coords: dict, *args, **kwargs):
        """Evaluate the interpolation defined for the given coordinates

        Args:
            coords (dict): Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic.bicubic`
            **kwargs: List of keyword arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic.bicubic`

        Returns:
            The interpolated values
        """
        return bicubic.bicubic(self, *_coords(coords, self._dims), *args,
                               **kwargs)


class Grid3D(grid.Grid3D):
    """Builds a Grid3D from the Xarray data provided.

    Args:
        data_array (xarray.DataArray): Provided data array
    """
    def __init__(self, data_array: xr.DataArray):
        x, y = _lon_lat_from_data_array(data_array, ndims=3)
        z = (set(data_array.coords) - {x, y}).pop()
        self._dims = (x, y, z)
        super(Grid3D, self).__init__(
            core.Axis(data_array.coords[x].values, is_circle=True),
            core.Axis(data_array.coords[y].values),
            core.Axis(data_array.coords[z].values),
            data_array.transpose(x, y, z).values)

    def trivariate(self, coords: dict, *args, **kwargs):
        """Evaluate the interpolation defined for the given coordinates

        Args:
            coords (dict): Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.trivariate.trivariate`
            **kwargs: List of keywords arguments provided to the interpolation
                method :py:meth:`pyinterp.trivariate.trivariate`

        Returns:
            The interpolated values
        """
        return trivariate.trivariate(self, *_coords(coords, self._dims), *args,
                                     **kwargs)

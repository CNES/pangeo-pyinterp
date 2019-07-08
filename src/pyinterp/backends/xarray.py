"""
XArray
------

Build interpolation objects from XArray Dataset instances
"""
from typing import Iterable, Optional, Tuple
import xarray as xr
from .. import cf
from .. import core
from .. import bicubic
from .. import bivariate
from .. import trivariate


class AxisIdentifier:
    def __init__(self, dataset: xr.Dataset):
        self.dataset = dataset

    def _axis(self, units: cf.AxisUnit) -> Optional[str]:
        """Returns the name of the dimension that defines an axis.

        Args:
            array : xarray.DataArray
                The array defining the regular grid in n dimensions.

        Returns
            str, optional:
                The name of the coordinate
        """
        for name, coord in self.dataset.coords.items():
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


def _lon_lat_from_dataset(dataset: xr.Dataset,
                          variable: str,
                          ndims: Optional[int] = 2) -> Tuple[str, str]:
    """
    Gets the name of the dimensions that define the longitudes and latitudes
    of the dataset.

    Args:
        dataset (xarray.Dataset): Provided dataset
        name (str): Variable to interpolate
        ndims (int, optional): Number of dimension expected for the variable

    Returns:
        tuple: longitude and latitude names

    Raises:
        ValueError if the provided dataset doen't define a
            longitude/latitude axis
        ValueError if the number of dimensions is different from the number of
            dimensions of the grid provided.
    """
    ident = AxisIdentifier(dataset)
    lon = ident.longitude()
    if lon is None:
        raise ValueError("The dataset doesn't define a longitude axis")
    lat = ident.latitude()
    if lat is None:
        raise ValueError("The dataset doesn't define a longitude axis")
    size = len(dataset.variables[variable].shape)
    if size != ndims:
        raise ValueError(
            "The number of dimensions of the variable "
            f"{variable} is incorrect. Expected {ndims}, found {size}.")
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


class Bivariate(bivariate.Bivariate):
    """Builds the Bivariate interpolator from the provided dataset.

    Args:
        dataset (xarray.Dataset): Provided dataset
        name (str): Variable to interpolate
    """

    def __init__(self, dataset: xr.Dataset, variable: str):
        self._dims = _lon_lat_from_dataset(dataset, variable)
        super(Bivariate, self).__init__(
            core.Axis(dataset.variables[self._dims[0]].values, is_circle=True),
            core.Axis(dataset.variables[self._dims[1]].values),
            dataset.variables[variable].transpose(*self._dims).values)

    def evaluate(self, coords: dict, *args, **kwargs):
        """Evaluate the interpolation defined for the given coordinates

        Args:
            coords (dict): Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.bivariate.Bivariate.evaluate`
            **kwargs: List of keywords arguments provided to the interpolation
                method :py:meth:`pyinterp.bivariate.Bivariate.evaluate`

        Returns:
            The interpolated values
        """
        return super(Bivariate, self).evaluate(*_coords(coords, self._dims),
                                               *args, **kwargs)

    def __getstate__(self) -> Tuple:
        return (self._dims, super(Bivariate, self).__getstate__())

    def __setstate__(self, state) -> None:
        self._dims, state = state
        super(Bivariate, self).__setstate__(state)


class Bicubic(bicubic.Bicubic):
    """Builds the Bicubic interpolator from the provided dataset.

    Args:
        dataset (xarray.Dataset): Provided dataset
        name (str): Variable to interpolate
    """

    def __init__(self, dataset: xr.Dataset, variable: str):
        self._dims = _lon_lat_from_dataset(dataset, variable)
        super(Bicubic, self).__init__(
            core.Axis(dataset.variables[self._dims[0]].values, is_circle=True),
            core.Axis(dataset.variables[self._dims[1]].values),
            dataset.variables[variable].transpose(*self._dims).values)

    def evaluate(self, coords: dict, *args, **kwargs):
        """Evaluate the interpolation defined for the given coordinates

        Args:
            coords (dict): Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic.Bicubic.evaluate`
            **kwargs: List of keyword arguments provided to the interpolation
                method :py:meth:`pyinterp.bicubic.Bicubic.evaluate`

        Returns:
            The interpolated values
        """
        return super(Bicubic, self).evaluate(*_coords(coords, self._dims),
                                             *args, **kwargs)

    def __getstate__(self) -> Tuple:
        return (self._dims, super(Bicubic, self).__getstate__())

    def __setstate__(self, state) -> None:
        self._dims, state = state
        super(Bicubic, self).__setstate__(state)


class Trivariate(trivariate.Trivariate):
    """Builds the Bicubic interpolator from the provided dataset.

    Args:
        dataset (xarray.Dataset): Provided dataset
        name (str): Variable to interpolate
    """

    def __init__(self, dataset: xr.Dataset, variable: str):
        x, y = _lon_lat_from_dataset(dataset, variable, ndims=3)
        z = (set(dataset.coords) - {x, y}).pop()
        self._dims = (x, y, z)
        super(Trivariate, self).__init__(
            core.Axis(dataset.variables[x].values, is_circle=True),
            core.Axis(dataset.variables[y].values),
            core.Axis(dataset.variables[z].values),
            dataset.variables[variable].transpose(x, y, z).values)

    def evaluate(self, coords: dict, *args, **kwargs):
        """Evaluate the interpolation defined for the given coordinates

        Args:
            coords (dict): Mapping from dimension names to the
                coordinates to interpolate. Coordinates must be array-like.
            *args: List of arguments provided to the interpolation
                method :py:meth:`pyinterp.trivariate.Trivariate.evaluate`
            **kwargs: List of keywords arguments provided to the interpolation
                method :py:meth:`pyinterp.trivariate.Trivariate.evaluate`

        Returns:
            The interpolated values
        """
        return super(Trivariate, self).evaluate(*_coords(coords, self._dims),
                                                *args, **kwargs)

    def __getstate__(self) -> Tuple:
        return (self._dims, super(Trivariate, self).__getstate__())

    def __setstate__(self, state) -> None:
        self._dims, state = state
        super(Trivariate, self).__setstate__(state)

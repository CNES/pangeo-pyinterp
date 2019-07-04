# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Interface with the library core
===============================
"""
from typing import List, Tuple, Optional
import numpy as np
import xarray as xr


def _core_suffix(x: np.ndarray):
    """Get the suffix of the class handling the numpy data type.

    Args:
        x (numpy.ndarray): array to process
    Returns:
        str: the class suffix
    """
    dtype = x.dtype.type
    if dtype == np.float64:
        return 'Float64'
    if dtype == np.float32:
        return 'Float32'
    if dtype == np.int64:
        return 'Int64'
    if dtype == np.uint64:
        return 'UInt64'
    if dtype == np.int32:
        return 'Int32'
    if dtype == np.uint32:
        return 'UInt32'
    if dtype == np.int16:
        return 'Int16'
    if dtype == np.uint16:
        return 'UInt16'
    if dtype == np.int8:
        return 'Int8'
    if dtype == np.uint8:
        return 'UInt8'
    raise ValueError("Unhandled dtype: " + str(dtype))


class AxisUnit(list):
    """Units management for axes"""

    @property
    def units(self) -> List:
        """Get the list of known units

        Returns:
            list: The known units
        """
        return self


class AxisLatitudeUnit(AxisUnit):
    """Units known to the axis defining the latitude"""

    def __init__(self):
        super(AxisLatitudeUnit, self).__init__()
        self.append("degrees_north")
        self.append("degree_north")
        self.append("degree_N")
        self.append("degrees_N")
        self.append("degreeN")
        self.append("degreesN")


class AxisLongitudeUnit(AxisUnit):
    """Units known to the axis defining the longitude"""

    def __init__(self):
        super(AxisLongitudeUnit, self).__init__()
        self.append("degrees_east")
        self.append("degree_east")
        self.append("degree_E")
        self.append("degrees_E")
        self.append("degreeE")
        self.append("degreesE")


class AxisTimeUnit(AxisUnit):
    """Units known to the axis defining the time"""

    def __init__(self):
        super(AxisTimeUnit, self).__init__()
        self.append("days")
        self.append("seconds")


class AxisIdentifier:
    def __init__(self, dataset: xr.Dataset):
        self.dataset = dataset

    def _axis(self, units: AxisUnit) -> Optional[str]:
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
        return self._axis(AxisLongitudeUnit())

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
        return self._axis(AxisLatitudeUnit())


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

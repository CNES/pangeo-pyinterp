"""
Geohash converters
------------------
"""
import numpy
import xarray

from .. import core
from ..core import geohash


def to_xarray(hashs: numpy.ndarray, data: numpy.ndarray) -> xarray.DataArray:
    """Get the XArray grid representing the GeoHash grid.

    Args:
        hashs: Geohash codes.
        data: The data associated with the codes provided.

    Returns:
        The XArray grid representing the GeoHash grid.
    """
    if hashs.shape != data.shape:
        raise ValueError(
            "hashs, data could not be broadcast together with shape "
            f"{hashs.shape}, f{data.shape}")
    if hashs.dtype.kind != 'S':
        raise TypeError("hashs must be a string array")
    lon, lat = geohash.decode(
        geohash.bounding_boxes(precision=hashs.dtype.itemsize))
    x_axis = core.Axis(numpy.unique(lon), is_circle=True)
    y_axis = core.Axis(numpy.unique(lat))

    dtype = data.dtype
    if numpy.issubdtype(dtype, numpy.dtype("object")):
        grid = numpy.empty((len(y_axis), len(x_axis)), dtype)
    else:
        grid = numpy.zeros((len(y_axis), len(x_axis)), dtype)

    lon, lat = geohash.decode(hashs)
    grid[y_axis.find_index(lat), x_axis.find_index(lon)] = data

    return xarray.DataArray(
        grid,
        dims=('lat', 'lon'),
        coords=dict(lon=xarray.DataArray(x_axis,
                                         dims=("lon", ),
                                         attrs=dict(units="degrees_north")),
                    lat=xarray.DataArray(y_axis,
                                         dims=("lat", ),
                                         attrs=dict(units="degrees_east"))))

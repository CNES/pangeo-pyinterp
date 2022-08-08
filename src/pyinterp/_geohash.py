"""
Geohash encoding and decoding
-----------------------------
"""
from typing import Optional, Tuple, Type

#
import numpy
import xarray

#
from . import geodetic
from .core import GeoHash as BaseGeoHash, geohash


class GeoHash(BaseGeoHash):
    """Handle GeoHash encoded in base 32.

    Geohashing is a geocoding method used to encode geographic coordinates
    (latitude and longitude) into a short string of digits and letters
    delineating an area on a map, which is called a cell, with varying
    resolutions. The more characters in the string, the more precise the
    location. The table below gives the correspondence between the number of
    characters, the size of the boxes of the grid at the equator and the total
    number of boxes.

    =========  ===============  ==========
    precision  lng/lat (km)     samples
    =========  ===============  ==========
    1          4950/4950        32
    2          618.75/1237.50   1024
    3          154.69/154.69    32768
    4          19.34/38.67      1048576
    5          4.83/4.83        33554432
    6          0.60/1.21        1073741824
    =========  ===============  ==========

    Geohashes use Base-32 alphabet encoding (characters can be ``0`` to ``9``
    and ``A`` to ``Z``, excl ``A``, ``I``, ``L`` and ``O``).

    The geohash is a compact way of representing a location, and is useful for
    storing a location in a database, or for indexing a location in a database.
    """

    @classmethod
    def grid(cls,
             box: Optional[geodetic.Box] = None,
             precision: int = 1) -> xarray.Dataset:
        """Return the GeoHash grid covering the provided box.

        Args:
            box: Bounding box.
            precision: Number of characters in the geohash. Default is 1.

        Returns:
            Grid of the geohash.
        """
        bbox = box or geodetic.Box.whole_earth()
        _, nlon, nlat = cls.grid_properties(bbox, precision)
        grid = geohash.bounding_boxes(bbox, precision=precision)
        grid = numpy.flip(grid.reshape((nlat, nlon)), axis=0)
        lon, _ = geohash.decode(grid[-1, :].ravel())
        _, lat = geohash.decode(grid[:, 0].ravel())

        return xarray.Dataset(
            dict(geohash=xarray.DataArray(
                grid,
                dims=('lat', 'lon'),
                coords=dict(
                    lon=xarray.DataArray(
                        lon, dims=('lon', ), attrs=dict(
                            units='degrees_north')),
                    lat=xarray.DataArray(
                        lat, dims=('lat', ), attrs=dict(units='degrees_east')),
                ))))

    @staticmethod
    def from_string(code: str, round: bool = False) -> 'GeoHash':
        """Create from its string representation.

        Args:
            code: Geohash code.
            round: If true, the coordinates of the point will be rounded to the
                accuracy defined by the GeoHash.

        Returns:
            GeoHash: Geohash.

        Throws:
            ValueError: If the code is not a valid geohash.
        """
        return GeoHash(*BaseGeoHash.from_string(code, round).reduce())

    def __repr__(self) -> str:
        lon, lat, precision = super().reduce()
        return f'{self.__class__.__name__}({lon}, {lat}, {precision})'

    def __reduce__(self) -> Tuple[Type, Tuple[float, float, int]]:
        return (self.__class__, super().reduce())

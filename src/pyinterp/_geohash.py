from typing import Optional, Tuple, Type

#
import numpy
import xarray

#
from . import geodetic
from .core import GeoHash as BaseGeoHash, geohash

class GeoHash(BaseGeoHash):
    """
    GeoHash 

    Args:
        longitude (float): Longitude of the point.
        latitude (float): Latitude of the point.
        precision (int): Number of characters in the geohash. Default is 12.

    Throws:
        ValueError: If the precision is not in the range [1, 12].
    """
    @classmethod
    def grid(cls, box: Optional[geodetic.Box] = None, precision: int = 1):
        """
        Return the GeoHash grid covering the provided box.

        Args:
            box (geodetic.Box, optional): Bounding box.
            precision (int): Number of characters in the geohash. Default is 1.

        Returns:
            xarray.Dataset: Grid of the geohash.
        """
        bbox = box or geodetic.Box.whole_earth()
        _, nlon, nlat = cls.grid_properties(bbox, precision)
        grid = geohash.bounding_boxes(bbox, precision=precision)
        grid = numpy.flip(grid.reshape((nlat, nlon)), axis=0)
        lon, _ = geohash.decode(grid[-1, :].ravel())
        _, lat = geohash.decode(grid[:, 0].ravel())

        return xarray.Dataset(
            dict(geohash=xarray.DataArray(
                geohash.int64.encode(lon, lat, precision=5).reshape(4, 8),
                dims=('lat', 'lon'),
                coords=dict(lon=xarray.DataArray(
                    lon, dims=("lon", ), attrs=dict(units="degrees_north")),
                            lat=xarray.DataArray(lat,
                                                 dims=("lat", ),
                                                 attrs=dict(
                                                     units="degrees_east"))))))

    @staticmethod
    def from_string(code: str, round: bool = False) -> "GeoHash":
        """
        Create from its string representation.

        Args:
            code (str): Geohash code.
            round (bool): If true, the coordinates of the point will be rounded
                to the accuracy defined by the GeoHash.

        Returns:
            GeoHash: Geohash.

        Throws:
            ValueError: If the code is not a valid geohash.
        """
        return GeoHash(*BaseGeoHash.from_string(code, round).reduce())

    def __repr__(self) -> str:
        lon, lat, precision = super().reduce()
        return f"{self.__class__.__name__}({lon}, {lat}, {precision})"

    def __reduce__(self) -> Tuple[Type, Tuple[float, float, int]]:
        return (self.__class__, super().reduce())
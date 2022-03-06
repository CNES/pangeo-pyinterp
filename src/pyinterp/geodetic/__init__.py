"""
Geographic coordinate system
----------------------------
"""
from typing import List, Optional, Tuple

from ..core import geodetic
from ..core.geodetic import (
    Crossover,
    Linestring,
    coordinate_distances,
    normalize_longitudes,
)

__all__ = [
    "Box",
    "coordinate_distances",
    "Coordinates",
    "Crossover",
    "Linestring",
    "normalize_longitudes",
    "Point",
    "Polygon",
    "System",
]


class System(geodetic.System):
    """World Geodetic System (WGS).

    Args:
        parameters: A tuple that defines:

            * the semi-major axis of ellipsoid, in meters.
            * flattening of the ellipsoid.

    .. note::
        If no arguments are provided, the constructor initializes a WGS-84
        ellipsoid.

    Examples:
        >>> import pyinterp
        >>> wgs84 = pyinterp.geodetic.System()
        >>> wgs84
        System(6378137.0, 0.0033528106647474805)
        >>> grs80 = pyinterp.geodetic.System((6378137, 1 / 298.257222101))
        >>> grs80
        System(6378137.0, 0.003352810681182319)
    """

    def __init__(self, parameters: Optional[Tuple[float, float]] = None):
        super(System, self).__init__(*(parameters or ()))

    def __repr__(self):
        return f"System({self.semi_major_axis}, {self.flattening})"


class Coordinates(geodetic.Coordinates):
    """World Geodetic Coordinates System.

    Args:
        system: WGS System. If this argument is not defined, the instance
            manages a WGS84 ellipsoid.
    """

    def __init__(self, system: Optional[System] = None):
        super().__init__(system)


class Point(geodetic.Point):
    """Handle a point in an equatorial spherical coordinate system in degrees.

    Args:
        lon: Longitude in degrees of the point.
        lat: Latitude in degrees of the point.
    """

    def __init__(self, lon: float = 0, lat: float = 0):
        super().__init__(lon, lat)


class Box(geodetic.Box):
    """Defines a box made of two describing points in a spherical coordinates
    system in degrees.

    Args:
        min_corner: the minimum corner point (lower left) of the box.
        max_corner: the maximum corner point (upper right) of the box.
    """

    def __init__(self,
                 min_corner: Optional[Point] = None,
                 max_corner: Optional[Point] = None):
        super().__init__(min_corner or geodetic.Point(), max_corner
                         or geodetic.Point())


class Polygon(geodetic.Polygon):
    """The polygon contains an outer ring and zero or more inner rings.
    Args:
        outer: outer ring.
        inners: list of inner rings.

    Raises:
        ValueError: if outer is not a list of
            :py:class:`pyinterp.geodetic.Point`.
        ValueError: if inners is not a list of list of
            :py:class:`pyinterp.geodetic.Point`.
    """

    def __init__(self,
                 outer: List[Point],
                 inners: Optional[List[List[Point]]] = None) -> None:
        super().__init__(outer, inners)  # type: ignore

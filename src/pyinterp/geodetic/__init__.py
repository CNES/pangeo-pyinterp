"""
Geographic coordinate system
----------------------------
"""
from typing import Optional, Tuple
from ..core import geodetic


class System(geodetic.System):
    """World Geodetic System (WGS).

    Args:
        parameters (tuple, optional): A tuple that defines:

            * the semi-major axis of ellipsoid, in meters
            * flattening of ellipsoid

    .. note::
        If no arguments are provided, the constructor initializes a WGS-84
        ellipsoid.
    """

    def __init__(self, parameters: Optional[Tuple[float, float]] = None):
        super(System, self).__init__() if parameters is None else super(
            System, self).__init__(*parameters)


class Coordinates(geodetic.Coordinates):
    """World Geodetic Coordinates System

    Args:
        system (pyinterp.geodetic.System, optional): WGS System. If this
            argument is not defined, the instance manages a WGS84 ellipsoid.
    """

    def __init__(self, system: Optional[System] = None):
        super(Coordinates, self).__init__(system)


class Point2D(geodetic.Point2D):
    """Handle a point in a equatorial spherical coordinates system in degrees.

    Args:
        lon (float, optional): Longitude in degrees of the point
        lat (float, optional): Latitude in degrees of the point
    """

    def __init__(self, lon: Optional[float] = 0, lat: Optional[float] = 0):
        super(Point2D, self).__init__(lon, lat)


class Box2D(geodetic.Box2D):
    """Defines a box made of two describing points in a spherical coordinates
    system in degrees.

    Args:
        min_corner (pyinterp.geodetic.Point2D, optional): the minimum corner
            point (lower left) of the box
        max_corner (pyinterp.geodetic.Point2D, optional): the maximum corner
            point (upper right) of the box
    """

    def __init__(self,
                 min_corner: Optional[Point2D] = None,
                 max_corner: Optional[Point2D] = None):
        super(Box2D, self).__init__(min_corner or geodetic.Point2D(),
                                    max_corner or geodetic.Point2D())

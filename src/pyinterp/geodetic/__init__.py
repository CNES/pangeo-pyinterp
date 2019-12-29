"""
Geographic coordinate system
----------------------------
"""
from typing import Optional, Tuple
from ..core import geodetic


class System(geodetic.System):
    """World Geodetic System (WGS).
    """
    def __init__(self, parameters: Optional[Tuple[float, float]] = None):
        """Constructor

        Args:
            parameters (tuple, optional): A tuple that defines:

                * the semi-major axis of ellipsoid, in meters
                * flattening of ellipsoid

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
        super(System, self).__init__() if parameters is None else super(
            System, self).__init__(*parameters)

    def __repr__(self):
        return f"System({self.semi_major_axis}, {self.flattening})"


class Coordinates(geodetic.Coordinates):
    """World Geodetic Coordinates System
    """
    def __init__(self, system: Optional[System] = None):
        """Constructor

        Args:
            system (pyinterp.geodetic.System, optional): WGS System. If this
                argument is not defined, the instance manages a WGS84
                ellipsoid.
        """
        super(Coordinates, self).__init__(system)


class Point2D(geodetic.Point2D):
    """Handle a point in a equatorial spherical coordinates system in degrees.
    """
    def __init__(self, lon: float = 0, lat: float = 0):
        """Initialize a new 2D point

        Args:
            lon (float, optional): Longitude in degrees of the point
            lat (float, optional): Latitude in degrees of the point
        """
        super(Point2D, self).__init__(lon, lat)


class Box2D(geodetic.Box2D):
    """Defines a box made of two describing points in a spherical coordinates
    system in degrees.
    """
    def __init__(self,
                 min_corner: Optional[Point2D] = None,
                 max_corner: Optional[Point2D] = None):
        """Initialize a new 2D box

        Args:
            min_corner (pyinterp.geodetic.Point2D, optional): the minimum
                corner point (lower left) of the box
            max_corner (pyinterp.geodetic.Point2D, optional): the maximum
                corner point (upper right) of the box
        """
        super(Box2D, self).__init__(min_corner or geodetic.Point2D(),
                                    max_corner or geodetic.Point2D())

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

        Default constructor

    Args:
        system (pyinterp.geodetic.System, optional): WGS System. If this
            argument is not defined, the instance manages a WGS84 ellipsoid.
    """

    def __init__(self, system: Optional[System] = None):
        return super(Coordinates, self).__init__(system)

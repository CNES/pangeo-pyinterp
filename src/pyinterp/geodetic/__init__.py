"""
Geographic coordinate system
----------------------------
"""
from typing import List, Optional, Tuple

import numpy

from .. import interface
from ..core import geodetic
from ..core.geodetic import (
    Crossover,
    LineString,
    calculate_crossover,
    coordinate_distances,
)

__all__ = [
    'Box',
    'calculate_crossover',
    'coordinate_distances',
    'Coordinates',
    'Crossover',
    'LineString',
    'MultiPolygon',
    'normalize_longitudes',
    'Point',
    'Polygon',
    'RTree',
    'Spheroid',
]


class Spheroid(geodetic.Spheroid):
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
        >>> wgs84 = pyinterp.geodetic.Spheroid()
        >>> wgs84
        Spheroid(6378137.0, 0.0033528106647474805)
        >>> grs80 = pyinterp.geodetic.Spheroid((6378137, 1 / 298.257222101))
        >>> grs80
        Spheroid(6378137.0, 0.003352810681182319)
    """

    def __init__(self, parameters: Optional[Tuple[float, float]] = None):
        super().__init__(*(parameters or ()))

    def __repr__(self):
        return f'Spheroid({self.semi_major_axis}, {self.flattening})'


class Coordinates(geodetic.Coordinates):
    """World Geodetic Coordinates System.

    Args:
        spheroid: WGS System. If this argument is not defined, the instance
            manages a WGS84 ellipsoid.
    """

    def __init__(self, spheroid: Optional[Spheroid] = None):
        super().__init__(spheroid)


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


class MultiPolygon(geodetic.MultiPolygon):
    """The multi-polygon contains a list of polygons.

    Args:
        polygons: list of polygons. If this argument is not defined, the
            instance manages an empty list of polygons.

    Raises:
        ValueError: if polygons is not a list of
            :py:class:`pyinterp.geodetic.Polygon`.
    """

    def __init__(self, polygons: Optional[List[Polygon]] = None) -> None:
        args = (polygons, ) if polygons is not None else ()
        super().__init__(*args)


def normalize_longitudes(lon: numpy.ndarray,
                         min_lon: float = -180.0) -> numpy.ndarray:
    """Normalizes longitudes to the range ``[min_lon, min_lon + 360)``.

    Args:
        lon: Longitudes in degrees.
        min_lon: Minimum longitude. Defaults to ``-180.0``.

    Returns:
        Longitudes normalized to the range ``[min_lon, min_lon + 360)``.
    """
    if lon.flags.writeable:
        geodetic.normalize_longitudes(lon, min_lon)
        return lon
    return geodetic.normalize_longitudes(lon, min_lon)  # type: ignore


class RTree(geodetic.RTree):
    """A spatial index based on the R-tree data structure.

    Args:
        spheroid: WGS of the coordinate system used to calculate distance.
            If this argument is not defined, the instance manages a WGS84
            ellipsoid.
    """

    def __init__(self, spheroid: Optional[Spheroid] = None) -> None:
        super().__init__(spheroid)

    def inverse_distance_weighting(
            self,
            lon: numpy.ndarray,
            lat: numpy.ndarray,
            radius: Optional[float] = None,
            k: int = 9,
            p: int = 2,
            within: bool = True,
            num_threads: int = 0) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Interpolation of the value at the requested position by inverse
        distance weighting method.

        Args:
            lon: Longitudes in degrees.
            lat: Latitudes in degrees.
            radius: The maximum radius of the search (m). Defaults The maximum
                distance between two points.
            k: The number of nearest neighbors to be used for calculating the
                interpolated value. Defaults to ``9``.
            p: The power parameters. Defaults to ``2``.
            within: If true, the method ensures that the neighbors found are
                located around the point of interest. In other words, this
                parameter ensures that the calculated values will not be
                extrapolated. Defaults to ``true``.
            num_threads: The number of threads to use for the computation. If 0
                all CPUs are used. If 1 is given, no parallel computing code is
                used at all, which is useful for debugging. Defaults to ``0``.
        Returns:
            The interpolated value and the number of neighbors used in
            the calculation.
        """
        return super().inverse_distance_weighting(lon, lat, radius, k, p,
                                                  within, num_threads)

    def radial_basis_function(
        self,
        lon: numpy.ndarray,
        lat: numpy.ndarray,
        radius: Optional[float] = None,
        k: int = 9,
        rbf: Optional[str] = None,
        epsilon: Optional[float] = None,
        smooth: float = 0,
        within: bool = True,
        num_threads: int = 0,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Interpolation of the value at the requested position by radial basis
        function interpolation.

        Args:
            lon: Longitudes in degrees.
            lat: Latitudes in degrees.
            radius: The maximum radius of the search (m). Defaults The maximum
                distance between two points.
            k: The number of nearest neighbors to be used for calculating the
                interpolated value. Defaults to ``9``.
            rbf: The radial basis function, based on the radius, :math:`r`
                given by the distance between points. This parameter can take
                one of the following values:

                * ``cubic``: :math:`\\varphi(r) = r^3`
                * ``gaussian``: :math:`\\varphi(r) = e^{-(\\dfrac{r}
                  {\\varepsilon})^2}`
                * ``inverse_multiquadric``: :math:`\\varphi(r) = \\dfrac{1}
                  {\\sqrt{1+(\\dfrac{r}{\\varepsilon})^2}}`
                * ``linear``: :math:`\\varphi(r) = r`
                * ``multiquadric``: :math:`\\varphi(r) = \\sqrt{1+(
                  \\dfrac{r}{\\varepsilon})^2}`
                * ``thin_plate``: :math:`\\varphi(r) = r^2 \\ln(r)`

                Default to ``multiquadric``
            epsilon: adjustable constant for gaussian or multiquadrics
                functions. Default to the average distance between nodes.
            smooth: values greater than zero increase the smoothness of the
                approximation. Default to 0 (interpolation).
            within: If true, the method ensures that the neighbors found are
                located around the point of interest. In other words, this
                parameter ensures that the calculated values will not be
                extrapolated. Defaults to ``true``.
            num_threads: The number of threads to use for the computation. If 0
                all CPUs are used. If 1 is given, no parallel computing code is
                used at all, which is useful for debugging. Defaults to ``0``.
        Returns:
            The interpolated value and the number of neighbors used in the
            calculation.
        """
        return super().radial_basis_function(
            lon, lat, radius, k,
            interface._core_radial_basis_function(rbf, epsilon), epsilon,
            smooth, within, num_threads)

    def window_function(
        self,
        lon: numpy.ndarray,
        lat: numpy.ndarray,
        radius: float,
        k: int = 9,
        wf: Optional[str] = None,
        arg: Optional[float] = None,
        within: bool = True,
        num_threads: int = 0,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Interpolation of the value at the requested position by window
        function.

        The interpolated value will be equal to the expression:

        .. math::

            \\frac{\\sum_{i=1}^{k} \\omega(d_i,r)x_i}
            {\\sum_{i=1}^{k} \\omega(d_i,r)}

        where :math:`d_i` is the distance between the point of interest and
        the :math:`i`-th neighbor, :math:`r` is the radius of the search,
        :math:`x_i` is the value of the :math:`i`-th neighbor, and
        :math:`\\omega(d_i,r)` is weight calculated by the window function
        describe above.

        Args:
            lon: Longitudes in degrees.
            lat: Latitudes in degrees.
            radius: The maximum radius of the search (m).
            k: The number of nearest neighbors to be used for calculating the
                interpolated value. Defaults to ``9``.
            wf: The window function, based on the distance the distance between
                points (:math:`d`) and the radius (:math:`r`). This parameter
                can take one of the following values:

                * ``blackman``: :math:`w(d) = 0.42659 - 0.49656 \\cos(
                  \\frac{\\pi (d + r)}{r}) + 0.076849 \\cos(
                  \\frac{2 \\pi (d + r)}{r})`
                * ``blackman_harris``: :math:`w(d) = 0.35875 - 0.48829
                  \\cos(\\frac{\\pi (d + r)}{r}) + 0.14128
                  \\cos(\\frac{2 \\pi (d + r)}{r}) - 0.01168
                  \\cos(\\frac{3 \\pi (d + r)}{r})`
                * ``boxcar``: :math:`w(d) = 1`
                * ``flat_top``: :math:`w(d) = 0.21557895 -
                  0.41663158 \\cos(\\frac{\\pi (d + r)}{r}) +
                  0.277263158 \\cos(\\frac{2 \\pi (d + r)}{r}) -
                  0.083578947 \\cos(\\frac{3 \\pi (d + r)}{r}) +
                  0.006947368 \\cos(\\frac{4 \\pi (d + r)}{r})`
                * ``lanczos``: :math:`w(d) = \\left\\{\\begin{array}{ll}
                  sinc(\\frac{d}{r}) \\times sinc(\\frac{d}{arg \\times r}),
                  & d \\le arg \\times r \\\\ 0,
                  & d \\gt arg \\times r \\end{array} \\right\\}`
                * ``gaussian``: :math:`w(d) = e^{ -\\frac{1}{2}\\left(
                  \\frac{d}{\\sigma}\\right)^2 }`
                * ``hamming``: :math:`w(d) = 0.53836 - 0.46164
                  \\cos(\\frac{\\pi (d + r)}{r})`
                * ``nuttall``: :math:`w(d) = 0.3635819 - 0.4891775
                  \\cos(\\frac{\\pi (d + r)}{r}) + 0.1365995
                  \\cos(\\frac{2 \\pi (d + r)}{r})`
                * ``parzen``: :math:`w(d) = \\left\\{ \\begin{array}{ll} 1 - 6
                  \\left(\\frac{2*d}{2*r}\\right)^2
                  \\left(1 - \\frac{2*d}{2*r}\\right),
                  & d \\le \\frac{2r + arg}{4} \\\\
                  2\\left(1 - \\frac{2*d}{2*r}\\right)^3
                  & \\frac{2r + arg}{2} \\le d \\lt \\frac{2r +arg}{4}
                  \\end{array} \\right\\}`
                * ``parzen_swot``: :math:`w(d) = \\left\\{\\begin{array}{ll}
                  1 - 6\\left(\\frac{2 * d}{2 * r}\\right)^2
                  + 6\\left(1 - \\frac{2 * d}{2 * r}\\right), &
                  d \\le \\frac{2r}{4} \\\\
                  2\\left(1 - \\frac{2 * d}{2 * r}\\right)^3 &
                  \\frac{2r}{2} \\ge d \\gt \\frac{2r}{4} \\end{array}
                  \\right\\}`
            arg: The optional argument of the window function. Defaults to
                ``1`` for ``lanczos``, to ``0`` for ``parzen`` and for all
                other functions is ``None``.
            within: If true, the method ensures that the neighbors found are
                located around the point of interest. In other words, this
                parameter ensures that the calculated values will not be
                extrapolated. Defaults to ``true``.
            num_threads: The number of threads to use for the computation. If 0
                all CPUs are used. If 1 is given, no parallel computing code is
                used at all, which is useful for debugging. Defaults to ``0``.
        Returns:
            The interpolated value and the number of neighbors used in the
            calculation.
        """
        return super().window_function(
            lon, lat, radius, k, interface._core_window_function(wf, arg), arg,
            within, num_threads)

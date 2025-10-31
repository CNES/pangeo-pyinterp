"""Geographic coordinate system."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..typing import NDArray1DFloat64

from .. import interface
from ..core import geodetic
from ..core.geodetic import (
    Crossover,
    LineString,
    calculate_crossover,
    calculate_crossover_list,
    coordinate_distances,
)

__all__ = [
    'Box',
    'Coordinates',
    'Crossover',
    'LineString',
    'MultiPolygon',
    'Point',
    'Polygon',
    'RTree',
    'Spheroid',
    'calculate_crossover',
    'calculate_crossover_list',
    'coordinate_distances',
    'normalize_longitudes',
]


class Spheroid(geodetic.Spheroid):
    """Represent a World Geodetic System (WGS).

    Define an ellipsoid model for geodetic calculations.

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

    def __init__(self, parameters: tuple[float, float] | None = None) -> None:
        """Initialize a Spheroid instance."""
        super().__init__(*(parameters or ()))

    def __repr__(self) -> str:
        """Return a string representation of the Spheroid instance."""
        return f'Spheroid({self.semi_major_axis}, {self.flattening})'


class Coordinates(geodetic.Coordinates):
    """Represent a World Geodetic Coordinates System.

    Manage geodetic coordinates using a specified spheroid model.

    Args:
        spheroid: WGS System. If this argument is not defined, the instance
            manages a WGS84 ellipsoid.

    """

    def __init__(self, spheroid: Spheroid | None = None) -> None:
        """Initialize a Coordinates instance."""
        super().__init__(spheroid)


class Point(geodetic.Point):
    """Handle a point in an equatorial spherical coordinate system.

    Represent a geographic point using longitude and latitude in degrees.

    Args:
        lon: Longitude in degrees of the point.
        lat: Latitude in degrees of the point.

    """

    def __init__(self, lon: float = 0, lat: float = 0) -> None:
        """Initialize a Point instance."""
        super().__init__(lon, lat)


class Box(geodetic.Box):
    """Define a box made of two describing points in spherical coordinates.

    Represent a rectangular region using minimum and maximum corner points in
    degrees. The Box class supports both standard rectangular regions and
    dateline-crossing regions.

    Args:
        min_corner: the minimum corner point (lower left) of the box.
        max_corner: the maximum corner point (upper right) of the box.

    Note:
        **Handling the International Date Line:**

        When creating a box that crosses the International Date Line (180°/-180°
        longitude), the Box class automatically detects this situation when
        ``max_corner.lon < min_corner.lon`` and normalizes the coordinates
        internally.

        For example, a box from 170°E to -170°W (crossing the dateline) should
        be specified with:

        * ``min_corner = Point(170, -10)``
        * ``max_corner = Point(-170, 10)``

        The Box will automatically handle this as a dateline-crossing region
        spanning from 170°E eastward through 180° to -170°W, rather than
        incorrectly wrapping westward from 170°E to -170°W.

        For queries, longitude values are automatically normalized to match the
        box's coordinate system, so you can use standard [-180, 180] coordinates
        when testing point containment.

    Examples:
        Create a standard box that doesn't cross the dateline:

        >>> import pyinterp.geodetic
        >>> box = pyinterp.geodetic.Box(
        ...     pyinterp.geodetic.Point(-10, -5),
        ...     pyinterp.geodetic.Point(10, 5))
        >>> box.covered_by(pyinterp.geodetic.Point(0, 0))
        True
        >>> box.covered_by(pyinterp.geodetic.Point(20, 0))
        False

        Create a box crossing the International Date Line:

        >>> dateline_box = pyinterp.geodetic.Box(
        ...     pyinterp.geodetic.Point(170, -10),
        ...     pyinterp.geodetic.Point(-170, 10))
        >>> # Points on both sides of the dateline
        >>> dateline_box.covered_by(pyinterp.geodetic.Point(175, 0))
        True
        >>> dateline_box.covered_by(pyinterp.geodetic.Point(-175, 0))
        True
        >>> # Point in the gap (outside the box)
        >>> dateline_box.covered_by(pyinterp.geodetic.Point(0, 0))
        False

    See Also:
        :ref:`example_dateline_box`: Example demonstrating dateline handling
        with the Box class.

    """

    def __init__(self,
                 min_corner: Point | None = None,
                 max_corner: Point | None = None) -> None:
        """Initialize a Box instance."""
        super().__init__(min_corner or geodetic.Point(), max_corner
                         or geodetic.Point())


class Polygon(geodetic.Polygon):
    """Represent a polygon with an outer ring and optional inner rings.

    Define a polygon containing one outer ring and zero or more inner rings
    (holes).

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
                 outer: list[Point],
                 inners: list[list[Point]] | None = None) -> None:
        """Initialize a Polygon instance."""
        super().__init__(outer, inners)


class MultiPolygon(geodetic.MultiPolygon):
    """Represent a collection of polygons.

    Define a multi-polygon containing a list of polygons.

    Args:
        polygons: list of polygons. If this argument is not defined, the
            instance manages an empty list of polygons.

    Raises:
        ValueError: if polygons is not a list of
            :py:class:`pyinterp.geodetic.Polygon`.

    """

    def __init__(self, polygons: list[Polygon] | None = None) -> None:
        """Initialize a MultiPolygon instance."""
        args = (polygons, ) if polygons is not None else ()
        super().__init__(*args)


def normalize_longitudes(lon: NDArray1DFloat64,
                         min_lon: float = -180.0) -> NDArray1DFloat64:
    """Normalize longitudes to the range ``[min_lon, min_lon + 360)``.

    Adjust longitude values to fall within the specified range.

    Args:
        lon: Longitudes in degrees.
        min_lon: Minimum longitude. Defaults to ``-180.0``.

    Returns:
        Longitudes normalized to the range ``[min_lon, min_lon + 360)``.

    """
    if lon.flags.writeable:
        geodetic.normalize_longitudes(lon, min_lon)
        return lon
    return geodetic.normalize_longitudes(  # type: ignore[return-value]
        lon, min_lon)


class RTree(geodetic.RTree):
    """Provide a spatial index based on the R-tree data structure.

    Create a spatial indexing structure for efficient geographic queries.

    Args:
        spheroid: WGS of the coordinate system used to calculate distance.
            If this argument is not defined, the instance manages a WGS84
            ellipsoid.

    """

    def __init__(self, spheroid: Spheroid | None = None) -> None:
        """Initialize a RTree instance."""
        super().__init__(spheroid)

    def inverse_distance_weighting(
            self,
            lon: NDArray1DFloat64,
            lat: NDArray1DFloat64,
            radius: float | None = None,
            k: int = 9,
            p: int = 2,
            within: bool = True,
            num_threads: int = 0) -> tuple[NDArray1DFloat64, NDArray1DFloat64]:
        """Interpolate values using inverse distance weighting method.

        Calculate interpolated values at requested positions using the inverse
        distance weighting approach.

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

    def radial_basis_function(  # type: ignore[override]
        self,
        lon: NDArray1DFloat64,
        lat: NDArray1DFloat64,
        radius: float | None = None,
        k: int = 9,
        rbf: str | None = None,
        epsilon: float | None = None,
        smooth: float = 0,
        within: bool = True,
        num_threads: int = 0,
    ) -> tuple[NDArray1DFloat64, NDArray1DFloat64]:
        r"""Interpolate values using radial basis function interpolation.

        Calculate interpolated values at requested positions using radial basis
        functions.

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

                * ``cubic``: :math:`\varphi(r) = r^3`
                * ``gaussian``: :math:`\varphi(r) = e^{-(\dfrac{r}
                  {\varepsilon})^2}`
                * ``inverse_multiquadric``: :math:`\varphi(r) = \dfrac{1}
                  {\sqrt{1+(\dfrac{r}{\varepsilon})^2}}`
                * ``linear``: :math:`\varphi(r) = r`
                * ``multiquadric``: :math:`\varphi(r) = \sqrt{1+(
                  \dfrac{r}{\varepsilon})^2}`
                * ``thin_plate``: :math:`\varphi(r) = r^2 \ln(r)`

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

    def window_function(  # type: ignore[override]
        self,
        lon: NDArray1DFloat64,
        lat: NDArray1DFloat64,
        radius: float,
        k: int = 9,
        wf: str | None = None,
        arg: float | None = None,
        within: bool = True,
        num_threads: int = 0,
    ) -> tuple[NDArray1DFloat64, NDArray1DFloat64]:
        r"""Interpolate values using a window function.

        Calculate interpolated values at requested positions using a specified
        window function.

        The interpolated value will be equal to the expression:

        .. math::

            \frac{\sum_{i=1}^{k} \omega(d_i,r)x_i}
            {\sum_{i=1}^{k} \omega(d_i,r)}

        where :math:`d_i` is the distance between the point of interest and
        the :math:`i`-th neighbor, :math:`r` is the radius of the search,
        :math:`x_i` is the value of the :math:`i`-th neighbor, and
        :math:`\omega(d_i,r)` is weight calculated by the window function
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

                * ``blackman``: :math:`w(d) = 0.42659 - 0.49656 \cos(
                  \frac{\pi (d + r)}{r}) + 0.076849 \cos(
                  \frac{2 \pi (d + r)}{r})`
                * ``blackman_harris``: :math:`w(d) = 0.35875 - 0.48829
                  \cos(\frac{\pi (d + r)}{r}) + 0.14128
                  \cos(\frac{2 \pi (d + r)}{r}) - 0.01168
                  \cos(\frac{3 \pi (d + r)}{r})`
                * ``boxcar``: :math:`w(d) = 1`
                * ``flat_top``: :math:`w(d) = 0.21557895 -
                  0.41663158 \cos(\frac{\pi (d + r)}{r}) +
                  0.277263158 \cos(\frac{2 \pi (d + r)}{r}) -
                  0.083578947 \cos(\frac{3 \pi (d + r)}{r}) +
                  0.006947368 \cos(\frac{4 \pi (d + r)}{r})`
                * ``lanczos``: :math:`w(d) = \left\{\begin{array}{ll}
                  sinc(\frac{d}{r}) \times sinc(\frac{d}{arg \times r}),
                  & d \le arg \times r \\ 0,
                  & d \gt arg \times r \end{array} \right\}`
                * ``gaussian``: :math:`w(d) = e^{ -\frac{1}{2}\left(
                  \frac{d}{\sigma}\right)^2 }`
                * ``hamming``: :math:`w(d) = 0.53836 - 0.46164
                  \cos(\frac{\pi (d + r)}{r})`
                * ``nuttall``: :math:`w(d) = 0.3635819 - 0.4891775
                  \cos(\frac{\pi (d + r)}{r}) + 0.1365995
                  \cos(\frac{2 \pi (d + r)}{r})`
                * ``parzen``: :math:`w(d) = \left\{ \begin{array}{ll} 1 - 6
                  \left(\frac{2*d}{2*r}\right)^2
                  \left(1 - \frac{2*d}{2*r}\right),
                  & d \le \frac{2r + arg}{4} \\
                  2\left(1 - \frac{2*d}{2*r}\right)^3
                  & \frac{2r + arg}{2} \le d \lt \frac{2r +arg}{4}
                  \end{array} \right\}`
                * ``parzen_swot``: :math:`w(d) = \left\{\begin{array}{ll}
                  1 - 6\left(\frac{2 * d}{2 * r}\right)^2
                  + 6\left(1 - \frac{2 * d}{2 * r}\right), &
                  d \le \frac{2r}{4} \\
                  2\left(1 - \frac{2 * d}{2 * r}\right)^3 &
                  \frac{2r}{2} \ge d \gt \frac{2r}{4} \end{array}
                  \right\}`
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

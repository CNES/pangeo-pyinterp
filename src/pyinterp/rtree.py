# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
RTree spatial index
-------------------
"""
from typing import Optional, Tuple

import numpy

from . import core, geodetic, interface


class RTree:
    """R*Tree spatial index for geodetic scalar values.

    Args:
        system: WGS of the coordinate system used to transform equatorial
            spherical positions (longitudes, latitudes, altitude) into ECEF
            coordinates. If not set the geodetic system used is WGS-84.
            Default to ``None``.
        dtype: Data type of the instance to create.
        ndims: The number of dimensions of the tree. This dimension must be
            at least equal to 3 to store the ECEF coordinates of the points.
            Default to ``3``.
    """

    def __init__(self,
                 system: Optional[geodetic.Spheroid] = None,
                 dtype: Optional[numpy.dtype] = None,
                 ndims: int = 3):
        """Initialize a new R*Tree."""
        dtype = dtype or numpy.dtype('float64')
        if ndims < 3:
            raise ValueError('ndims must be >= 3')
        if dtype == numpy.dtype('float64'):
            self._instance = getattr(core, f'RTree{ndims}DFloat64')(system)
        elif dtype == numpy.dtype('float32'):
            self._instance = getattr(core, f'RTree{ndims}DFloat32')(system)
        else:
            raise ValueError(f'dtype {dtype} not handled by the object')
        self.dtype = dtype

    def bounds(
            self
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Returns the box able to contain all values stored in the container.

        Returns:
            A tuple that contains the coordinates of the minimum and maximum
            corners of the box able to contain all values stored in the
            container or an empty tuple if there are no values in the container.
        """
        return self._instance.bounds()

    def clear(self) -> None:
        """Removes all values stored in the container."""
        return self._instance.clear()

    def __len__(self):
        """Returns the number of values stored in the tree."""
        return self._instance.__len__()

    def __bool__(self):
        """Returns true if the tree is not empty."""
        return self._instance.__bool__()

    def packing(self, coordinates: numpy.ndarray,
                values: numpy.ndarray) -> None:
        """The tree is created using packing algorithm (The old data is erased
        before construction.)

        Args:
            coordinates: a matrix ``(n, ndims)`` where ``n`` is the number of
                observations and ``ndims`` is the number of coordinates in
                order: longitude and latitude in degrees, altitude in meters and
                then the other coordinates defined in Euclidean space if
                ``dims`` > 3. If the shape of the matrix is ``(n, ndims)`` then
                the method considers the altitude constant and equal to zero.
            values: An array of size ``(n)`` containing the values associated
                with the coordinates provided.
        """
        self._instance.packing(coordinates, values)

    def insert(self, coordinates: numpy.ndarray,
               values: numpy.ndarray) -> None:
        """Insert new data into the search tree.

        Args:
            coordinates: a matrix ``(n, ndims)`` where ``n`` is the number of
                observations and ``ndims`` is the number of coordinates in
                order: longitude and latitude in degrees, altitude in meters and
                then the other coordinates defined in Euclidean space if
                ``dims`` > 3. If the shape of the matrix is ``(n, ndims)`` then
                the method considers the altitude constant and equal to zero.
            values: An array of size ``(n)`` containing the values associated
                with the coordinates provided.
        """
        self._instance.insert(coordinates, values)

    def value(self,
              coordinates: numpy.ndarray,
              radius: Optional[float] = None,
              k: int = 4,
              within: bool = True,
              num_threads: int = 0) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the coordinates and values for the K-nearest neighbors of a
        given point.

        Args:
            coordinates: a matrix ``(n, ndims)`` where ``n`` is the number of
                observations and ``ndims`` is the number of coordinates in
                order: longitude and latitude in degrees, altitude in meters.
                If ndims is equal to 2 then the altitude is considered to be
                constant and equal to zero.
            radius (optional): The maximum distance in meters to search for
                neighbors. If not set, the search is performed on all the
                neighbors.
            k: The number of nearest neighbors to return.
            within: if true, the method returns the k nearest neighbors if the
                point is within by its neighbors.
            num_threads: The number of threads to use for the computation. If 0
                all CPUs are used. If 1 is given, no parallel computing code is
                used at all, which is useful for debugging. Defaults to ``0``.

        Returns:
            A tuple containing the coordinates and values of the K-nearest
            neighbors of the given point.

        .. note::
            The matrix containing the coordinates of the neighbors is a matrix
            of dimension ``(k, n)`` where ``n`` is equal to 2 if the provided
            coordinates matrix defines only longitude and latitude, and 3 if the
            defines longitude, latitude and altitude.
        """
        return self._instance.value(coordinates, radius, k, within,
                                    num_threads)

    def query(self,
              coordinates: numpy.ndarray,
              k: int = 4,
              within: bool = True,
              num_threads: int = 0) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Search for the nearest K nearest neighbors of a given point.

        Args:
            coordinates: a matrix ``(n, ndims)`` where ``n`` is the number of
                observations and ``ndims`` is the number of coordinates in
                order: longitude and latitude in degrees, altitude in meters.
                If ndims is equal to 2 then the altitude is considered to be
                constant and equal to zero.
            k: The number of nearest neighbors to be searched. Defaults to
                ``4``.
            within: If true, the method ensures that the neighbors found are
                located within the point of interest. Defaults to ``false``.
            num_threads: The number of threads to use for the computation. If 0
                all CPUs are used. If 1 is given, no parallel computing code is
                used at all, which is useful for debugging. Defaults to ``0``.
        Returns:
            A tuple containing a matrix describing for each provided position,
            the distance, in meters, between the provided position and the found
            neighbors and a matrix containing the value of the different
            neighbors found for all provided positions.
            If no neighbors are found, the distance and the value are set to
            ``-1``.
        """
        return self._instance.query(coordinates, k, within, num_threads)

    def inverse_distance_weighting(
            self,
            coordinates: numpy.ndarray,
            radius: Optional[float] = None,
            k: int = 9,
            p: int = 2,
            within: bool = True,
            num_threads: int = 0) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Interpolation of the value at the requested position by inverse
        distance weighting method.

        Args:
            coordinates: a matrix ``(n, ndims)`` where ``n`` is the number of
                observations and ``ndims`` is the number of coordinates in
                order: longitude and latitude in degrees, altitude in meters.
                If ndims is equal to 2 then the altitude is considered to be
                constant and equal to zero.
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
        return self._instance.inverse_distance_weighting(
            coordinates, radius, k, p, within, num_threads)

    def radial_basis_function(
            self,
            coordinates: numpy.ndarray,
            radius: Optional[float] = None,
            k: int = 9,
            rbf: Optional[str] = None,
            epsilon: Optional[float] = None,
            smooth: float = 0,
            within: bool = True,
            num_threads: int = 0) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Interpolation of the value at the requested position by radial basis
        function interpolation.

        Args:
            coordinates: a matrix ``(n, ndims)`` where ``n`` is the number of
                observations and ``ndims`` is the number of coordinates in
                order: longitude and latitude in degrees, altitude in meters.
                If ndims is equal to 2 then the altitude is considered to be
                constant and equal to zero.
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
        return self._instance.radial_basis_function(
            coordinates, radius, k,
            interface._core_radial_basis_function(rbf, epsilon), epsilon,
            smooth, within, num_threads)

    def window_function(
            self,
            coordinates: numpy.ndarray,
            radius: float,
            k: int = 9,
            wf: Optional[str] = None,
            arg: Optional[float] = None,
            within: bool = True,
            num_threads: int = 0) -> Tuple[numpy.ndarray, numpy.ndarray]:
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
            coordinates: a matrix ``(n, ndims)`` where ``n`` is the number of
                observations and ``ndims`` is the number of coordinates in
                order: longitude and latitude in degrees, altitude in meters.
                If ndims is equal to 2 then the altitude is considered to be
                constant and equal to zero.
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
        return self._instance.window_function(
            coordinates, radius, k, interface._core_window_function(wf, arg),
            arg, within, num_threads)

    def __getstate__(self) -> Tuple:
        """Return the state of the object for pickling purposes.

        Returns:
            The state of the object for pickling purposes.
        """
        return (self.dtype, self._instance.__getstate__())

    def __setstate__(self, state: Tuple):
        """Set the state of the object from pickling.

        Args:
            state: The state of the object for pickling purposes.
        """
        if len(state) != 2:
            raise ValueError('invalid state')
        _class = RTree(None, state[0])
        self.dtype = _class.dtype
        _class._instance.__setstate__(state[1])
        self._instance = _class._instance

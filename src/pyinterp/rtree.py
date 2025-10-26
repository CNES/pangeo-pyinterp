# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""RTree spatial index."""
from __future__ import annotations

from typing import TYPE_CHECKING, overload
import warnings

import numpy

from . import core, geodetic, interface

if TYPE_CHECKING:
    from .typing import (
        NDArray1DFloat32,
        NDArray1DFloat64,
        NDArray2DFloat32,
        NDArray2DFloat64,
        NDArray3DFloat32,
        NDArray3DFloat64,
    )


class RTree:
    """R*Tree spatial index for geodetic scalar values.

    Args:
        system: WGS of the coordinate system used to transform equatorial
            spherical positions (longitudes, latitudes, altitude) into ECEF
            coordinates. If not set the geodetic system used is WGS-84.
            Default to ``None``.
        dtype: Data type of the instance to create.
        ecef: If true, the coordinates are provided in the ECEF system,
            otherwise the coordinates are provided in the geodetic system.
            Default to ``False``.

    Raises:
        ValueError: if the data type is not handled by the object, or if the
            a geodetic system is provided and the coordinates system is ECEF
            (ecef keyword is set to True).

    """

    def __init__(self,
                 system: geodetic.Spheroid | None = None,
                 dtype: numpy.dtype | None = None,
                 ecef: bool = False) -> None:
        """Initialize a new R*Tree."""
        self._instance: core.RTree3DFloat32 | core.RTree3DFloat64
        dtype = dtype or numpy.dtype('float64')
        if dtype == numpy.dtype('float64'):
            self._instance = core.RTree3DFloat64(system, ecef)
        elif dtype == numpy.dtype('float32'):
            self._instance = core.RTree3DFloat32(system, ecef)
        else:
            raise ValueError(f'dtype {dtype} not handled by the object')
        self.dtype = dtype

    def bounds(
            self
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Get the bounding box containing all stored values.

        Returns:
            A tuple containing the coordinates of the minimum and maximum
            corners of the box able to contain all values stored in the
            container, or an empty tuple if the container is empty.

        """
        return self._instance.bounds()

    def clear(self) -> None:
        """Remove all values stored in the container."""
        return self._instance.clear()

    def __len__(self) -> int:
        """Get the number of values stored in the tree.

        Returns:
            The number of values in the tree.

        """
        return self._instance.__len__()

    def __bool__(self) -> bool:
        """Check if the tree is not empty.

        Returns:
            True if the tree contains values, False otherwise.

        """
        return self._instance.__bool__()

    @overload
    def packing(self, coordinates: NDArray2DFloat32,
                values: NDArray1DFloat32) -> None:
        ...

    @overload
    def packing(self, coordinates: NDArray3DFloat32,
                values: NDArray1DFloat32) -> None:
        ...

    @overload
    def packing(self, coordinates: NDArray2DFloat64,
                values: NDArray1DFloat64) -> None:
        ...

    @overload
    def packing(self, coordinates: NDArray3DFloat64,
                values: NDArray1DFloat64) -> None:
        ...

    def packing(
        self,
        coordinates: NDArray2DFloat32 | NDArray3DFloat32 | NDArray2DFloat64
        | NDArray3DFloat64,
        values: NDArray1DFloat32 | NDArray1DFloat64,
    ) -> None:
        """Create the tree using packing algorithm.

        The old data is erased before construction.

        Args:
            coordinates: a matrix of shape ``(n, 3)``, where ``n`` is the number
                of observations and 3 represents the coordinates in theorder:
                x, y, and z.
                If the matrix shape is ``(n, 2)``, the z-coordinate is assumed
                to be zero.
                The coordinates (x, y, z) are in the Cartesian coordinate system
                (ECEF) if the instance is configured to use this system (ecef
                keyword set to True during construction). Otherwise, the
                coordinates are in the geodetic system (longitude, latitude, and
                altitude) in degrees, degrees, and meters, respectively.
            values: An array of size ``(n)`` containing the values associated
                with the coordinates provided.

        """
        self._instance.packing(coordinates, values)  # type: ignore[arg-type]

    def insert(self, coordinates: numpy.ndarray,
               values: numpy.ndarray) -> None:
        """Insert new data into the search tree.

        Args:
            coordinates: a matrix of shape ``(n, 3)``, where ``n`` is the number
                of observations and 3 represents the coordinates in theorder:
                x, y, and z.
                If the matrix shape is ``(n, 2)``, the z-coordinate is assumed
                to be zero.
                The coordinates (x, y, z) are in the Cartesian coordinate system
                (ECEF) if the instance is configured to use this system (ecef
                keyword set to True during construction). Otherwise, the
                coordinates are in the geodetic system (longitude, latitude, and
                altitude) in degrees, degrees, and meters, respectively.
            values: An array of size ``(n)`` containing the values associated
                with the coordinates provided.

        """
        self._instance.insert(coordinates, values)

    def value(self,
              coordinates: numpy.ndarray,
              radius: float | None = None,
              k: int = 4,
              within: bool = True,
              num_threads: int = 0) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Get the coordinates and values for the K-nearest neighbors.

        Args:
            coordinates: a matrix of shape ``(n, 3)``, where ``n`` is the number
                of observations and 3 represents the coordinates in theorder:
                x, y, and z.
                If the matrix shape is ``(n, 2)``, the z-coordinate is assumed
                to be zero.
                The coordinates (x, y, z) are in the Cartesian coordinate system
                (ECEF) if the instance is configured to use this system (ecef
                keyword set to True during construction). Otherwise, the
                coordinates are in the geodetic system (longitude, latitude, and
                altitude) in degrees, degrees, and meters, respectively.
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
            coordinates matrix defines only x and y, and 3 if the defines x, y,
            and z.

        """
        return self._instance.value(coordinates, radius, k, within,
                                    num_threads)

    def query(self,
              coordinates: numpy.ndarray,
              k: int = 4,
              within: bool = True,
              num_threads: int = 0) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Search for the nearest K nearest neighbors of a given point.

        Args:
            coordinates: a matrix of shape ``(n, 3)``, where ``n`` is the number
                of observations and 3 represents the coordinates in theorder:
                x, y, and z.
                If the matrix shape is ``(n, 2)``, the z-coordinate is assumed
                to be zero.
                The coordinates (x, y, z) are in the Cartesian coordinate system
                (ECEF) if the instance is configured to use this system (ecef
                keyword set to True during construction). Otherwise, the
                coordinates are in the geodetic system (longitude, latitude, and
                altitude) in degrees, degrees, and meters, respectively.
            k: The number of nearest neighbors to be searched. Defaults to
                ``4``.
            within: If true, the method ensures that the neighbors found are
                located within the point of interest. Defaults to ``false``.
            num_threads: The number of threads to use for the computation. If 0
                all CPUs are used. If 1 is given, no parallel computing code is
                used at all, which is useful for debugging. Defaults to ``0``.

        Returns:
            A tuple containing a matrix describing for each provided position,
            the distance between the provided position and the found neighbors
            (in meters if the RTree handles LLA coordinates, otherwise in
            Cartesian units) and a matrix containing the value of the different
            neighbors found for all provided positions.
            If no neighbors are found, the distance and the value are set to
            ``-1``.

        """
        return self._instance.query(coordinates, k, within, num_threads)

    def inverse_distance_weighting(
            self,
            coordinates: numpy.ndarray,
            radius: float | None = None,
            k: int = 9,
            p: int = 2,
            within: bool = True,
            num_threads: int = 0) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Interpolate values using inverse distance weighting method.

        Args:
            coordinates: a matrix of shape ``(n, 3)``, where ``n`` is the number
                of observations and 3 represents the coordinates in theorder:
                x, y, and z.
                If the matrix shape is ``(n, 2)``, the z-coordinate is assumed
                to be zero.
                The coordinates (x, y, z) are in the Cartesian coordinate system
                (ECEF) if the instance is configured to use this system (ecef
                keyword set to True during construction). Otherwise, the
                coordinates are in the geodetic system (longitude, latitude, and
                altitude) in degrees, degrees, and meters, respectively.
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
            radius: float | None = None,
            k: int = 9,
            rbf: str | None = None,
            epsilon: float | None = None,
            smooth: float = 0,
            within: bool = True,
            num_threads: int = 0) -> tuple[numpy.ndarray, numpy.ndarray]:
        r"""Interpolate values using radial basis function interpolation.

        Args:
            coordinates: a matrix of shape ``(n, 3)``, where ``n`` is the number
                of observations and 3 represents the coordinates in theorder:
                x, y, and z.
                If the matrix shape is ``(n, 2)``, the z-coordinate is assumed
                to be zero.
                The coordinates (x, y, z) are in the Cartesian coordinate system
                (ECEF) if the instance is configured to use this system (ecef
                keyword set to True during construction). Otherwise, the
                coordinates are in the geodetic system (longitude, latitude, and
                altitude) in degrees, degrees, and meters, respectively.
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
            radius: float | None = None,
            k: int = 9,
            wf: str | None = None,
            arg: float | None = None,
            within: bool = True,
            num_threads: int = 0) -> tuple[numpy.ndarray, numpy.ndarray]:
        r"""Interpolate values using a window function.

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
            coordinates: a matrix of shape ``(n, 3)``, where ``n`` is the number
                of observations and 3 represents the coordinates in theorder:
                x, y, and z.
                If the matrix shape is ``(n, 2)``, the z-coordinate is assumed
                to be zero.
                The coordinates (x, y, z) are in the Cartesian coordinate system
                (ECEF) if the instance is configured to use this system (ecef
                keyword set to True during construction). Otherwise, the
                coordinates are in the geodetic system (longitude, latitude, and
                altitude) in degrees, degrees, and meters, respectively.
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

    def universal_kriging(
            self,
            coordinates: numpy.ndarray,
            radius: float | None = None,
            k: int = 9,
            covariance: str | None = None,
            sigma: float = 1.0,
            alpha: float = 1_000_000.0,
            within: bool = True,
            num_threads: int = 0) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Interpolate the values of a point using universal kriging.

        See the :meth:`kriging` method for the description of the parameters.

        .. deprecated:: 2025.9.0

            universal_kriging method is deprecated, use :meth:`kriging` method
            instead.
        """
        warnings.warn(
            'universal_kriging method is deprecated, '
            'use kriging method instead',
            DeprecationWarning,
            stacklevel=2)
        return self.kriging(
            coordinates,
            radius=radius,
            k=k,
            covariance=covariance,
            sigma=sigma,
            alpha=alpha,
            within=within,
            num_threads=num_threads,
        )

    def kriging(self,
                coordinates: numpy.ndarray,
                *,
                radius: float | None = None,
                k: int = 9,
                covariance: str | None = None,
                drift_function: str | None = None,
                sigma: float = 1.0,
                alpha: float = 1_000_000.0,
                nugget: float = 0.0,
                within: bool = True,
                num_threads: int = 0) -> tuple[numpy.ndarray, numpy.ndarray]:
        r"""Interpolate the values of a point using kriging.

        Args:
            coordinates: a matrix of shape ``(n, 3)``, where ``n`` is the number
                of observations and 3 represents the coordinates in theorder:
                x, y, and z.
                If the matrix shape is ``(n, 2)``, the z-coordinate is assumed
                to be zero.
                The coordinates (x, y, z) are in the Cartesian coordinate system
                (ECEF) if the instance is configured to use this system (ecef
                keyword set to True during construction). Otherwise, the
                coordinates are in the geodetic system (longitude, latitude, and
                altitude) in degrees, degrees, and meters, respectively.
            radius: The maximum radius of the search (m).
            k: The number of nearest neighbors to be used for calculating the
                interpolated value. Defaults to ``9``.
            covariance: The covariance function, based on the distance between
                points. This parameter can take one of the following values:

                * ``matern_12``: :math:`\\sigma^2\\exp\\left(-\\frac{d}{\\rho}
                  \\right)`
                * ``matern_32``: :math:`\\sigma^2\\left(1+\\frac{\\sqrt{3}d}{
                  \\rho}\\right)\\exp\\left(-\\frac{\\sqrt{3}d}{\\rho}
                  \\right)`
                * ``matern_52``: :math:`\\sigma^2\\left(1+\\frac{\\sqrt{5}d}{
                  \\rho}+\\frac{5d^2}{3\\rho^2}\\right) \\exp\\left(-\\frac{
                  \\sqrt{5}d}{\\rho} \\right)`
                * ``cauchy``: :math:`\\sigma^2 \\left(1 + \\frac{d}{\\rho}
                  \\right)^{-1}`
                * ``gaussian``: :math:`\\sigma^2 \\exp \\left(-\\frac{d^2}{
                  \\rho^2} \\right)`
                * ``spherical``: :math:`\\sigma^2 \\left(1 - \\frac{3d}{2r}
                  + \\frac{3d^3}{2r^3} \\right) \\left(\\frac{d}{r} \\le 1
                  \\right)`
                * ``linear``: :math:`\\sigma^2 \\left(1 - \\frac{d}{r}
                  \\right) \\left(\\frac{d}{r} \\le 1 \\right)`
            drift_function: The drift (trend) function to be used for universal
                kriging. This parameter can take one of the following values:

                * ``linear``: :math:`m(x,y,z) = \\beta_0 + \\beta_1 x +
                  \\beta_2 y + \\beta_3 z`
                * ``quadratic``: :math:`m(x,y,z) = \\beta_0 + \\beta_1 x +
                  \\beta_2 y + \\beta_3 z + \\beta_4 x^2 + \\beta_5 y^2 +
                  \\beta_6 z^2 + \\beta_7 xy + \\beta_8 xz + \\beta_9 yz`

                Defaults to ``None`` (simple kriging with known mean 0).
            sigma: The sill (magnitude) parameter :math:`\\sigma` of the
                covariance function. Determines the overall scale (maximum
                covariance).
            alpha: The range parameter :math:`\\rho`. Determines how quickly
                the covariance decays with distance. Units must match the
                distance units used internally (geodetic/ECEF -> meters,
                pure Cartesian -> user units).
            nugget: Nugget effect (added to the covariance matrix diagonal).
                Accounts for measurement error or unresolved microscale
                variability. Must be :math:`\\ge 0`.
            within: If true, the method ensures that the neighbors found are
                located around the point of interest (prevents extrapolation).
            num_threads: Number of threads to use. ``0`` uses all available,
                ``1`` disables parallelism (useful for debugging).

        Returns:
            The interpolated value and the number of neighbors used in the
            calculation.

        .. note::
            * If ``drift_function`` is ``None``, simple kriging with known mean
              0 is applied.
            * If ``drift_function`` is provided, universal kriging augments the
              system with the corresponding trend basis functions.
            * ``alpha`` corresponds to the range parameter :math:`\\rho`
              controlling spatial correlation extent.

        """
        return self._instance.kriging(
            coordinates, radius, k,
            interface._core_covariance_function(covariance),
            interface._core_drift_function(drift_function), sigma, alpha,
            nugget, within, num_threads)

    def __getstate__(self) -> tuple:
        """Get the state of the object for pickling.

        Returns:
            The state of the object for pickling purposes.

        """
        return (self.dtype, self._instance.__getstate__())

    def __setstate__(self, state: tuple) -> None:
        """Set the state of the object from pickling.

        Args:
            state: The state of the object for pickling purposes.

        """
        if len(state) != 2:  # noqa: PLR2004
            raise ValueError('invalid state')
        _class = RTree(None, state[0])
        self.dtype = _class.dtype
        _class._instance.__setstate__(state[1])
        self._instance = _class._instance

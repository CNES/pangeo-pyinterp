# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Trivariate interpolation
========================
"""
import numpy as np

from .. import core, grid, interface


def trivariate(grid3d: grid.Grid3D,
               x: np.ndarray,
               y: np.ndarray,
               z: np.ndarray,
               interpolator: str = "bilinear",
               z_method: str = "linear",
               bounds_error: bool = False,
               num_threads: int = 0,
               **kwargs) -> np.ndarray:
    """Interpolate the values provided on the defined trivariate function.

    Args:
        grid3d (pyinterp.grid.Grid3D): Function on a uniform 3-dimensional
            grid to be interpolated.
        x (numpy.ndarray): X-values.
        y (numpy.ndarray): Y-values.
        z (numpy.ndarray): Z-values.
        interpolator (str, optional): The interpolation method to be performed
            on the surface defined by the Y and Y axes. Supported are
            ``bilinear`` and ``nearest``, and ``inverse_distance_weighting``.
            Default to ``bilinear``.
        z_method (str, optional): The interpolation method to be performed
            on the Z axis. Supported are ``linear``and ``nearest``. Default
            to ``linear``.
        bounds_error (bool, optional): If True, when interpolated values
            are requested outside of the domain of the input axes (x,y), a
            :py:class:`ValueError` is raised. If False, then the value is set
            to NaN. Default to ``False``.
        num_threads (int, optional): The number of threads to use for the
            computation. If 0 all CPUs are used. If 1 is given, no parallel
            computing code is used at all, which is useful for debugging.
            Defaults to ``0``.
        p (int, optional): The power to be used by the interpolator
            inverse_distance_weighting. Default to ``2``.
    Returns:
        numpy.ndarray: Values interpolated.
    """
    instance = grid3d._instance
    function = interface._core_function("trivariate", instance)
    return getattr(core, function)(instance,
                                   np.asarray(x),
                                   np.asarray(y),
                                   np.asarray(z),
                                   grid._core_variate_interpolator(
                                       grid3d, interpolator, **kwargs),
                                   z_method=z_method,
                                   bounds_error=bounds_error,
                                   num_threads=num_threads)

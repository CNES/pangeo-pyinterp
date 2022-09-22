# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Quadrivariate interpolation
===========================
"""
import numpy

from .. import core, grid, interface


def quadrivariate(grid4d: grid.Grid4D,
                  x: numpy.ndarray,
                  y: numpy.ndarray,
                  z: numpy.ndarray,
                  u: numpy.ndarray,
                  interpolator: str = 'bilinear',
                  z_method: str = 'linear',
                  u_method: str = 'linear',
                  bounds_error: bool = False,
                  num_threads: int = 0,
                  **kwargs) -> numpy.ndarray:
    """Interpolate the values provided on the defined quadrivariate function.

    Args:
        grid4d: Function on a uniform 4-dimensional grid to be interpolated.
        x: X-values.
        y: Y-values.
        z: Z-values.
        u: U-values.
        interpolator: The interpolation method to be performed on the surface
            defined by the Y and Y axes. Supported are ``bilinear`` and
            ``nearest``, and ``inverse_distance_weighting``. Default to
            ``bilinear``.
        z_method: The interpolation method to be performed on the Z axis.
            Supported are ``linear`` and ``nearest``. Default to ``linear``.
        u_method: The interpolation method to be performed on the U axis.
            Supported are ``linear`` and ``nearest``. Default to ``linear``.
        bounds_error: If True, when interpolated values are requested outside
            of the domain of the input axes (x,y), a :py:class:`ValueError` is
            raised. If False, then the value is set to NaN. Default to
            ``False``.
        num_threads: The number of threads to use for the computation. If 0 all
            CPUs are used. If 1 is given, no parallel computing code is used at
            all, which is useful for debugging. Defaults to ``0``.
        p: The power to be used by the interpolator inverse_distance_weighting.
            Default to ``2``.
    Returns:
        Values interpolated.
    """
    instance = grid4d._instance
    function = interface._core_function('quadrivariate', instance)
    return getattr(core, function)(instance,
                                   numpy.asarray(x),
                                   numpy.asarray(y),
                                   numpy.asarray(z),
                                   numpy.asarray(u),
                                   grid._core_variate_interpolator(
                                       grid4d, interpolator, **kwargs),
                                   z_method=z_method,
                                   u_method=u_method,
                                   bounds_error=bounds_error,
                                   num_threads=num_threads)

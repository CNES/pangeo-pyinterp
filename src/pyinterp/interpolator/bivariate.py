# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Bivariate interpolation
=======================
"""
import numpy

from .. import core, grid, interface


def bivariate(grid2d: grid.Grid2D,
              x: numpy.ndarray,
              y: numpy.ndarray,
              interpolator: str = 'bilinear',
              bounds_error: bool = False,
              num_threads: int = 0,
              **kwargs) -> numpy.ndarray:
    """Interpolate the values provided on the defined bivariate function.

    Args:
        grid2d: Function on a uniform 2-dimensional grid to be interpolated.
        x: X-values.
        y: Y-values.
        interpolator: The method of interpolation to perform. Supported are
            ``bilinear``, ``nearest``, and ``inverse_distance_weighting``.
            Default to ``bilinear``.
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
    instance = grid2d._instance
    function = interface._core_function('bivariate', instance)
    return getattr(core,
                   function)(instance, numpy.asarray(x), numpy.asarray(y),
                             grid._core_variate_interpolator(
                                 grid2d, interpolator, **kwargs), bounds_error,
                             num_threads)

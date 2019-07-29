# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Bivariate interpolation
=======================
"""
from typing import Optional
import numpy as np
from . import core
from . import grid
from . import interface


def bivariate(grid2d: grid.Grid2D,
              x: np.ndarray,
              y: np.ndarray,
              interpolator: Optional[str] = "bilinear",
              bounds_error: Optional[bool] = False,
              num_threads: Optional[int] = 0,
              **kwargs) -> np.ndarray:
    """Interpolate the values provided on the defined bivariate function.

    Args:
        grid2d (pyinterp.grid.Grid2D): Function on a uniform 2-dimensional
            grid to be interpolated.
        x (numpy.ndarray): X-values
        y (numpy.ndarray): Y-values
        interpolator (str, optional): The method of interpolation to
            perform. Supported are ``bilinear``, ``nearest``, and
            ``inverse_distance_weighting``. Default to ``bilinear``.
        bounds_error (bool, optional): If True, when interpolated values
            are requested outside of the domain of the input axes (x,y), a
            :py:class:`ValueError` is raised. If False, then value is set
            to NaN. Default to ``False``
        num_threads (int, optional): The number of threads to use for the
            computation. If 0 all CPUs are used. If 1 is given, no parallel
            computing code is used at all, which is useful for debugging.
            Defaults to ``0``.
        p (int, optional): The power to be used by the interpolator
            inverse_distance_weighting. Default to ``2``.
    Return:
        numpy.ndarray: Values interpolated
    """
    instance = grid2d._instance
    function = f"bivariate_{interface._core_function_suffix(instance)}"
    return getattr(core, function)(instance, np.asarray(x), np.asarray(y),
                                   grid._core_variate_interpolator(
                                       grid2d, interpolator, **kwargs),
                                   bounds_error, num_threads)

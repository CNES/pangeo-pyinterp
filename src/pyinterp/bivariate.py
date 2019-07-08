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
from . import GridInterpolator
from . import core


class Bivariate(GridInterpolator):
    """Interpolation of bivariate functions

    Args:
        x (pyinterp.core.Axis): X-Axis
        y (pyinterp.core.Axis): Y-Axis
        array (numpy.ndarray): Bivariate function
    """
    _CLASS = "Bivariate"
    _INTEROLATOR = "2D"

    def __init__(self, x: core.Axis, y: core.Axis, values: np.ndarray):
        super(Bivariate, self).__init__(x, y, values)

    def evaluate(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 interpolator: Optional[str] = "bilinear",
                 num_threads: Optional[int] = 0,
                 **kwargs) -> np.ndarray:
        """Interpolate the values provided on the defined bivariate function.

        Args:
            x (numpy.ndarray): X-values
            y (numpy.ndarray): Y-values
            interpolator (str, optional): The method of interpolation to
                perform. Supported are ``bilinear``, ``nearest``, and
                ``inverse_distance_weighting``. Default to ``bilinear``.
            num_threads (int, optional): The number of threads to use for the
                computation. If 0 all CPUs are used. If 1 is given, no parallel
                computing code is used at all, which is useful for debugging.
                Defaults to ``0``.
            p (int, optional): The power to be used by the interpolator
                inverse_distance_weighting. Default to ``2``.
        Return:
            numpy.ndarray: Values interpolated
        """
        return self._instance.evaluate(
            np.asarray(x), np.asarray(y),
            self._n_variate_interpolator(interpolator, **kwargs), num_threads)

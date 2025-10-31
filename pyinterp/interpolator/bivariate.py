# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Bivariate interpolation."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy

from .. import core, grid, interface

if TYPE_CHECKING:
    from ..typing import NDArray1D


def bivariate(grid2d: grid.Grid2D,
              x: NDArray1D,
              y: NDArray1D,
              interpolator: str = 'bilinear',
              bounds_error: bool = False,
              num_threads: int = 0,
              **kwargs: int) -> NDArray1D:
    """Interpolate the values provided on the defined bivariate function.

    Args:
        grid2d: Function on a uniform 2-dimensional grid to be interpolated.
        x: X-values.
        y: Y-values.
        interpolator: The method of interpolation to perform. Supported are
            ``bilinear``, ``nearest``, and ``inverse_distance_weighting``.
            Defaults to ``bilinear``.
        bounds_error: If True, when interpolated values are requested outside
            of the domain of the input axes (x,y), a :py:class:`ValueError` is
            raised. If False, the value is set to NaN. Defaults to ``False``.
        num_threads: The number of threads to use for the computation. If 0,
            all CPUs are used. If 1 is given, no parallel computing code is
            used (useful for debugging). Defaults to ``0``.
        **kwargs: Additional keyword arguments. Currently only ``p`` is
            recognized: the power parameter used by the
            ``inverse_distance_weighting`` interpolator. Default for ``p`` is
            2.

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

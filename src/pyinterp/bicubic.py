# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Bicubic interpolation
=====================
"""
from typing import Optional
import numpy as np
from . import core
from . import grid
from . import interface


def bicubic(grid2d: grid.Grid2D,
            x: np.ndarray,
            y: np.ndarray,
            nx: Optional[int] = 3,
            ny: Optional[int] = 3,
            fitting_model: Optional[str] = "c_spline",
            boundary: Optional[str] = "undef",
            bounds_error: Optional[bool] = False,
            num_threads: Optional[int] = 0) -> np.ndarray:
    """Extension of cubic interpolation for interpolating data points on a
    two-dimensional regular grid. The interpolated surface is smoother than
    corresponding surfaces obtained by bilinear interpolation or
    nearest-neighbor interpolation.

    Args:
        grid2d (pyinterp.grid.Grid2D): Function on a uniform 2-dimensional
            grid to be interpolated.
        x (numpy.ndarray): X-values
        y (numpy.ndarray): Y-values
        nx (int, optional): The number of X coordinate values required to
            perform the interpolation. Defaults to ``3``.
        ny (int, optional): The number of Y coordinate values required to
            perform the interpolation. Defaults to ``3``.
        fitting_model (str, optional): Type of interpolation to be
            performed. Supported are ``linear``, ``polynomial``,
            ``c_spline``, ``c_spline_periodic``, ``akima``,
            ``akima_periodic`` and ``steffen``. Default to
            ``c_spline``.
        boundary (str, optional): A flag indicating how to handle
            boundaries of the frame.

            * ``expand``: Expand the boundary as a constant.
            * ``wrap``: circular boundary conditions.
            * ``sym``: Symmetrical boundary conditions.
            * ``undef``: Boundary violation is not defined.

            Default ``undef``
        bounds_error (bool, optional): If True, when interpolated values
            are requested outside of the domain of the input axes (x,y), a
            :py:class:`ValueError` is raised. If False, then value is set
            to NaN. Default to ``False``

        num_threads (int, optional): The number of threads to use for the
            computation. If 0 all CPUs are used. If 1 is given, no parallel
            computing code is used at all, which is useful for debugging.
            Defaults to ``0``.
    Return:
        numpy.ndarray: Values interpolated
    """
    if fitting_model not in [
            'c_spline', 'c_spline_periodic', 'akima', 'akima_periodic',
            'steffen'
    ]:
        raise ValueError(f"fitting model {fitting_model!r} is not defined")

    fitting_model = "".join(item.capitalize()
                            for item in fitting_model.split("_"))

    if boundary not in ['expand', 'wrap', 'sym', 'undef']:
        raise ValueError(f"boundary {boundary!r} is not defined")

    boundary = boundary.capitalize()

    instance = grid2d._instance
    function = f"bicubic_{interface._core_function_suffix(instance)}"
    return getattr(core,
                   function)(instance, np.asarray(x), np.asarray(y), nx, ny,
                             getattr(core.FittingModel, fitting_model),
                             getattr(core.Axis.Boundary, boundary),
                             bounds_error, num_threads)

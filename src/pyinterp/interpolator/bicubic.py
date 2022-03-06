# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Bicubic interpolation
=====================
"""
from typing import Optional, Union

import numpy as np

from .. import core, grid, interface


def bicubic(mesh: Union[grid.Grid2D, grid.Grid3D, grid.Grid4D],
            x: np.ndarray,
            y: np.ndarray,
            z: Optional[np.ndarray] = None,
            u: Optional[np.ndarray] = None,
            nx: Optional[int] = 3,
            ny: Optional[int] = 3,
            fitting_model: str = "c_spline",
            boundary: str = "undef",
            bounds_error: bool = False,
            num_threads: int = 0) -> np.ndarray:
    """Bicubic gridded interpolator.

    Args:
        mesh: Function on a uniform grid to be interpolated. If the grid is a
            ND grid, the bicubic interpolation is performed spatially along the
            X and Y axes of the ND grid and a linear interpolation are performed
            along the other axes between the values obtained by the bicubic
            interpolation.

            .. warning::

                The GSL functions for calculating bicubic functions require
                that the axes defined in the grids are strictly increasing.

        x: X-values.
        y: Y-values.
        z: None for a :py:class:`2D Grid <pyinterp.grid.Grid2D>` otherwise
            Z-values.
        u: None for a :py:class:`2D Grid <pyinterp.grid.Grid2D>`,
            :py:class:`3D Grid <pyinterp.grid.Grid3D>` otherwise U-values.
        nx: The number of X-coordinate values required to perform the
            interpolation. Defaults to ``3``.
        ny: The number of Y-coordinate values required to perform the
            interpolation. Defaults to ``3``.
        fitting_model: Type of interpolation to be performed.
            Supported are ``linear``, ``bicubic``, ``polynomial``, ``c_spline``,
            ``c_spline_periodic``, ``akima``, ``akima_periodic`` and
            ``steffen``. Default to ``bicubic``.
        boundary: A flag indicating how to handle boundaries of the frame.

            * ``expand``: Expand the boundary as a constant.
            * ``wrap``: circular boundary conditions.
            * ``sym``: Symmetrical boundary conditions.
            * ``undef``: Boundary violation is not defined.

            Default ``undef``.
        bounds_error: If True, when interpolated values are requested outside
            of the domain of the input axes (x,y), a :py:class:`ValueError` is
            raised. If False, then the value is set to NaN. Default to
            ``False``.
        num_threads: The number of threads to use for the computation. If 0 all
            CPUs are used. If 1 is given, no parallel computing code is used at
            all, which is useful for debugging. Defaults to ``0``.
    Returns:
        Values interpolated.
    """
    if not mesh.x.is_ascending():
        raise ValueError('X-axis is not increasing')
    if not mesh.y.is_ascending():
        raise ValueError('Y-axis is not increasing')
    if fitting_model not in [
            'akima_periodic', 'akima', 'bicubic', 'c_spline_periodic',
            'c_spline', 'linear', 'polynomial', 'steffen'
    ]:
        raise ValueError(f"fitting model {fitting_model!r} is not defined")

    if boundary not in ['expand', 'wrap', 'sym', 'undef']:
        raise ValueError(f"boundary {boundary!r} is not defined")

    instance = mesh._instance
    function = interface._core_function(
        "bicubic" if fitting_model == "bicubic" else "spline", instance)
    args = [
        instance,
        np.asarray(x),
        np.asarray(y), nx, ny, fitting_model, boundary, bounds_error,
        num_threads
    ]
    if isinstance(mesh, (grid.Grid3D, grid.Grid4D)):
        if z is None:
            raise ValueError(
                f"You must specify the Z-values for a {mesh._DIMENSIONS}D "
                "grid.")
        args.insert(3, np.asarray(z))
    if isinstance(mesh, grid.Grid4D):
        if u is None:
            raise ValueError("You must specify the U-values for a 4D grid.")
        args.insert(4, np.asarray(u))
    return getattr(core, function)(*args)

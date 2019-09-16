# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Regular grids
=============
"""
import numpy as np
from . import core
from . import interface


class Grid2D:
    """Cartesian Grid 2D

    Args:
        x (pyinterp.core.Axis): X-Axis
        y (pyinterp.core.Axis): Y-Axis
        array (numpy.ndarray): Discrete representation of a continuous
            function on a uniform 2-dimensional grid.
    """
    _DIMENSIONS = 2

    def __init__(self, *args):
        _class = f"Grid{self._DIMENSIONS}D" + interface._core_class_suffix(
            args[-1])
        self._instance = getattr(core, _class)(*args)

    @property
    def x(self) -> core.Axis:
        """
        Gets the X-Axis handled by this instance

        Returns:
            pyinterp.core.Axis: X-Axis
        """
        return self._instance.x

    @property
    def y(self) -> core.Axis:
        """
        Gets the Y-Axis handled by this instance

        Returns:
            pyinterp.core.Axis: Y-Axis
        """
        return self._instance.y

    @property
    def array(self) -> np.ndarray:
        """
        Gets the values handled by this instance

        Returns:
            numpy.ndarray: values
        """
        return self._instance.array


class Grid3D(Grid2D):
    """Cartesian Grid 3D

    Args:
        x (pyinterp.core.Axis): X-Axis
        y (pyinterp.core.Axis): Y-Axis
        z (pyinterp.core.Axis): Z-Axis
        array (numpy.ndarray): Discrete representation of a continuous
            function on a uniform 3-dimensional grid.
    """
    _DIMENSIONS = 3

    @property
    def z(self) -> core.Axis:
        """
        Gets the Z-Axis handled by this instance

        Returns:
            pyinterp.core.Axis: Z-Axis
        """
        return self._instance.z


def _core_variate_interpolator(instance: object, interpolator: str, **kwargs):
    """Obtain the interpolator from the string provided."""
    if isinstance(instance, Grid2D):
        dimensions = instance._DIMENSIONS
    else:
        raise TypeError("instance is not an object handling a grid.")

    if interpolator == "bilinear":
        return getattr(core, f"Bilinear{dimensions}D")(**kwargs)
    if interpolator == "nearest":
        return getattr(core, f"Nearest{dimensions}D")(**kwargs)
    if interpolator == "inverse_distance_weighting":
        return getattr(core,
                       f"InverseDistanceWeighting{dimensions}D")(**kwargs)

    raise ValueError(f"interpolator {interpolator!r} is not defined")

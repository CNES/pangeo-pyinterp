# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Regular grids
=============
"""
from typing import Optional
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
        increasing_axes ({'inplace', 'copy'}, optional): Optional string
            indicating how to ensure that the grid axes are increasing. If axes
            are decreasing, the axes and grid provided will be flipped in place
            or copied before being flipped. By default, the decreasing axes are
            not modified.
    """
    _DIMENSIONS = 2

    def __init__(self, *args, increasing_axes: Optional[str] = None):
        _class = f"Grid{self._DIMENSIONS}D" + interface._core_class_suffix(
            args[-1])
        if increasing_axes is not None:
            if increasing_axes not in ['inplace', 'copy']:
                raise ValueError("increasing_axes "
                                 f"{increasing_axes!r} is not defined")
            inplace = increasing_axes == 'inplace'
            args = list(args)
            for idx, item in enumerate(args):
                if isinstance(item, core.Axis) and not item.is_ascending():
                    args[idx] = item.flip(inplace=inplace)
                    args[-1] = np.flip(args[-1], axis=idx)
        self._instance = getattr(core, _class)(*args)

    def __repr__(self):
        result = [
            "<%s.%s>" % (self.__class__.__module__, self.__class__.__name__)
        ]
        result.append("Axis:")
        for item in dir(self):
            attr = getattr(self, item)
            if isinstance(attr, core.Axis):
                result.append("  %s: %s" % (item, attr))
        result.append("Data:")
        result += ["  %s" % line for line in str(self.array).split("\n")]
        return "\n".join(result)

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
        increasing_axes (bool, optional): Ensure that the axes of the grid are
            increasing. If this is not the case, the axes and grid provided
            will be flipped. Default to False.
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

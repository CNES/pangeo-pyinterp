# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Common classes
--------------
"""
from typing import Tuple
import numpy as np
from . import core
from . import interface


class GridInterpolator:
    """Abstract class of interpolation of numerical grids.

    Args:
        *args (tuple): Constructor's arguments.

    .. warning::

        This class should not be instantiated directly.
    """
    _CLASS = None
    _INTEROLATOR = None

    def __init__(self, *args):
        self._class = self._CLASS + interface._core_suffix(args[-1])
        self._instance = getattr(core, self._class)(*args)

    @classmethod
    def _n_variate_interpolator(cls, interpolator: str, **kwargs):
        if interpolator == "bilinear":
            return getattr(core, "Bilinear" + cls._INTEROLATOR)(**kwargs)
        elif interpolator == "nearest":
            return getattr(core, "Nearest" + cls._INTEROLATOR)(**kwargs)
        elif interpolator == "inverse_distance_weighting":
            return getattr(core, "InverseDistanceWeighting" +
                           cls._INTEROLATOR)(**kwargs)

        raise ValueError(f"interpolator {interpolator!r} is not defined")

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

    def __getstate__(self) -> Tuple:
        return (self._class, self._instance.__getstate__())

    def __setstate__(self, state) -> None:
        self._class = state[0]
        self._instance = getattr(getattr(core, self._class),
                                 "_setstate")(state[1])

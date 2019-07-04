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
import xarray as xr
from . import core
from . import interface


class GridInterpolator:
    """"""
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
    def x(self):
        """
        Gets the X-Axis handled by this instance

        Returns:
            pyinterp.core.Axis: X-Axis
        """
        return self._instance.x

    @property
    def y(self):
        """
        Gets the Y-Axis handled by this instance

        Returns:
            pyinterp.core.Axis: Y-Axis
        """
        return self._instance.y

    @property
    def array(self):
        """
        Gets the values handled by this instance

        Returns:
            numpy.ndarray: values
        """
        return self._instance.array

    def __getstate__(self):
        return (self._class, self._instance.__getstate__())

    def __setstate__(self, state):
        self._class = state[0]
        self._instance = getattr(getattr(core, self._class),
                                 "_setstate")(state[1])


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
                 **kwargs):
        """Interpolate the values provided on the defined bivariate function.

        Args:
            x (numpy.ndarray): X-values
            y (numpy.ndarray): Y-values
            interpolator (str, optional): The method of interpolation to
                perform. Supported are ``bilinear`` and ``nearest``, and
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


def from_dataset(dataset: xr.Dataset, variable: str):
    """Builds the interpolator from the provided dataset.

    Args:
        dataset (xarray.Dataset): Provided dataset
        name (str): Variable to interpolate

    Returns:
        Bivariate: the interpolator
    """
    lon, lat = interface._lon_lat_from_dataset(dataset, variable)
    return Bivariate(core.Axis(dataset.variables[lon].values, is_circle=True),
                     core.Axis(dataset.variables[lat].values),
                     dataset.variables[variable].transpose(lon, lat).values)

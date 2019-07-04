# Copyright (c) 2019 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
Trivariate interpolation
========================
"""
from typing import Optional
import numpy as np
import xarray as xr
from . import core
from . import interface
from . import bivariate


class Trivariate(bivariate.Bivariate):
    """Interpolation of trivariate functions

    Args:
        x (pyinterp.core.Axis): X-Axis
        y (pyinterp.core.Axis): Y-Axis
        z (pyinterp.core.Axis): Z-Axis
        array (numpy.ndarray): Trivariate function
    """
    _CLASS = "Trivariate"
    _INTEROLATOR = "3D"

    def __init__(self, x: core.Axis, y: core.Axis, z: core.Axis,
                 values: np.ndarray):
        bivariate.GridInterpolator.__init__(self, x, y, z, values)

    @property
    def z(self) -> core.Axis:
        """
        Gets the Z-Axis handled by this instance

        Returns:
            pyinterp.core.Axis: Z-Axis
        """
        return self._instance.z

    def evaluate(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 interpolator: Optional[str] = "bilinear",
                 num_threads: Optional[int] = 0,
                 **kwargs) -> np.ndarray:
        """Interpolate the values provided on the defined trivariate function.

        Args:
            x (numpy.ndarray): X-values
            y (numpy.ndarray): Y-values
            z (numpy.ndarray): Z-values
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
            np.asarray(x), np.asarray(y), np.asarray(z),
            self._n_variate_interpolator(interpolator, **kwargs), num_threads)


def from_dataset(dataset: xr.Dataset, variable: str) -> Trivariate:
    """Builds the interpolator from the provided dataset.

    Args:
        dataset (xarray.Dataset): Provided dataset
        name (str): Variable to interpolate

    Returns:
        Trivariate: the interpolator
    """
    lon, lat = interface._lon_lat_from_dataset(dataset, variable, 3)
    size = len(dataset.variables[variable].shape)
    if size != 3:
        raise ValueError("The number of dimensions of the variable "
                         f"{variable} is incorrect. Expected 3, found {size}.")
    z = (set(dataset.coords) - {lon, lat}).pop()
    return Trivariate(
        core.Axis(dataset.variables[lon].values, is_circle=True),
        core.Axis(dataset.variables[lat].values),
        core.Axis(dataset.variables[z].values),
        dataset.variables[variable].transpose(lon, lat, z).values)

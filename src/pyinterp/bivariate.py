"""
Bivariate interpolation
=======================
"""
import numpy as np
import xarray as xr
from . import core
from . import interface


class Bivariate:
    """Bivariate interpolation"""

    def __init__(self, lon: np.ndarray, lat: np.ndarray, values: np.ndarray):
        _class = getattr(core, "Bivariate" + interface._core_suffix(values))
        self._interp = _class(core.Axis(lon, is_circle=True), core.Axis(lat),
                              values)

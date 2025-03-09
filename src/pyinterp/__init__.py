# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""
pyinterp
--------
"""

from . import geodetic, geohash, version
from ._geohash import GeoHash
from .binning import Binning1D, Binning2D
from .core import (
    Axis,
    AxisInt64,
    RadialBasisFunction,
    TemporalAxis,
    WindowFunction,
    dateutils,
    interpolate1d,
)
from .grid import Grid2D, Grid3D, Grid4D
from .histogram2d import Histogram2D
from .interpolator.bicubic import bicubic
from .interpolator.bivariate import bivariate
from .interpolator.quadrivariate import quadrivariate
from .interpolator.trivariate import trivariate
from .orbit import (
    EquatorCoordinates,
    Orbit,
    Pass,
    Swath,
    calculate_orbit,
    calculate_pass,
    calculate_swath,
)
from .rtree import RTree
from .statistics import DescriptiveStatistics, StreamingHistogram

__version__ = version.release()
__date__ = version.date()
del version

__all__ = [
    'Axis',
    'AxisInt64',
    'Binning1D',
    'Binning2D',
    'DescriptiveStatistics',
    'EquatorCoordinates',
    'GeoHash',
    'Grid2D',
    'Grid3D',
    'Grid4D',
    'Histogram2D',
    'Orbit',
    'Pass',
    'RTree',
    'RadialBasisFunction',
    'StreamingHistogram',
    'Swath',
    'TemporalAxis',
    'WindowFunction',
    '__date__',
    '__version__',
    'bicubic',
    'bivariate',
    'calculate_orbit',
    'calculate_pass',
    'calculate_swath',
    'dateutils',
    'geodetic',
    'geohash',
    'interpolate1d',
    'quadrivariate',
    'trivariate',
]

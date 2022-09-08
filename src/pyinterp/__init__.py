# Copyright (c) 2022 CNES
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
    '__date__',
    '__version__',
    'Axis',
    'AxisInt64',
    'bicubic',
    'Binning1D',
    'Binning2D',
    'bivariate',
    'calculate_orbit',
    'calculate_pass',
    'calculate_swath',
    'dateutils',
    'DescriptiveStatistics',
    'EquatorCoordinates',
    'geodetic',
    'geohash',
    'GeoHash',
    'Grid2D',
    'Grid3D',
    'Grid4D',
    'Histogram2D',
    'Orbit',
    'Pass',
    'quadrivariate',
    'RadialBasisFunction',
    'RTree',
    'StreamingHistogram',
    'Swath',
    'TemporalAxis',
    'trivariate',
    'WindowFunction',
]

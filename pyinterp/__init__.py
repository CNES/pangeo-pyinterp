# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Pyinterp - Interpolation and geospatial operations for Python.

This package provides efficient interpolation methods for gridded data,
geospatial operations, and statistical analysis tools.
"""

import copyreg
import sys
from typing import Any


try:
    from ._version import version as __version__
except ImportError:
    # Package is not installed, use a development version
    __version__ = "0.0.0.dev0"

from . import core, fill, geohash
from .core import (
    Axis,
    Binning1D,
    Binning1DFloat32,
    Binning1DFloat64,
    Binning2D,
    Binning2DFloat32,
    Binning2DFloat64,
    DescriptiveStatistics,
    Grid,
    GridHolder,
    Histogram2D,
    Histogram2DFloat32,
    Histogram2DFloat64,
    RTree3D,
    RTree3DFloat32,
    RTree3DFloat64,
    TemporalAxis,
    config,
    dateutils,
    period,
)
from .regular_grid_interpolator import (
    bivariate,
    quadrivariate,
    trivariate,
    univariate,
    univariate_derivative,
)
from .rtree import (
    inverse_distance_weighting,
    kriging,
    query,
    radial_basis_function,
    window_function,
)


# Create Grid type aliases for runtime
# These mirror the type stubs in core/__init__.pyi
Grid1D = Grid
Grid2D = Grid
Grid3D = Grid
Grid4D = Grid

# Also add them to the core module namespace for core.Grid1D access
core.Grid1D = Grid1D  # type: ignore[assignment,misc]
core.Grid2D = Grid2D  # type: ignore[assignment,misc]
core.Grid3D = Grid3D  # type: ignore[assignment,misc]
core.Grid4D = Grid4D  # type: ignore[assignment,misc]

# Set up geometry module with flexible import patterns.
#
# Instead of having a separate geometry/ directory, we directly expose
# core.geometry and register submodules in sys.modules to enable:
# - from pyinterp.geometry import cartesian
# - from pyinterp.geometry.geographic import Point
# - from pyinterp.geometry.geographic.algorithms import area.
sys.modules.update(
    (
        (
            f"{__name__}.geometry",
            core.geometry,
        ),
        (
            f"{__name__}.geometry.cartesian",
            core.geometry.cartesian,
        ),
        (
            f"{__name__}.geometry.geographic",
            core.geometry.geographic,
        ),
        (
            f"{__name__}.geometry.satellite",
            core.geometry.satellite,
        ),
        (
            f"{__name__}.geometry.cartesian.algorithms",
            core.geometry.cartesian.algorithms,
        ),
        (
            f"{__name__}.geometry.geographic.algorithms",
            core.geometry.geographic.algorithms,
        ),
    )
)


__all__ = [
    "Axis",
    "Binning1D",
    "Binning1DFloat32",
    "Binning1DFloat64",
    "Binning2D",
    "Binning2DFloat32",
    "Binning2DFloat64",
    "DescriptiveStatistics",
    "Grid",
    "Grid1D",
    "Grid2D",
    "Grid3D",
    "Grid4D",
    "GridHolder",
    "Histogram2D",
    "Histogram2DFloat32",
    "Histogram2DFloat64",
    "RTree3D",
    "RTree3DFloat32",
    "RTree3DFloat64",
    "TemporalAxis",
    "bivariate",
    "config",
    "dateutils",
    "fill",
    "geohash",
    "geometry",
    "inverse_distance_weighting",
    "kriging",
    "period",
    "quadrivariate",
    "query",
    "radial_basis_function",
    "trivariate",
    "univariate",
    "univariate_derivative",
    "window_function",
]


def _unpickle_grid(state: tuple[Any, ...]) -> core.GridHolder:
    """Unpickle a Grid from state tuple (axes..., array)."""
    return core.Grid(*state)


def _reduce_grid(grid: core.GridHolder) -> tuple[Any, ...]:
    """Pickle reducer for Grid objects."""
    return (_unpickle_grid, (grid.__getstate__(),))


# Register the pickle reducer for GridHolder (the actual C++ class)
copyreg.pickle(core.GridHolder, _reduce_grid)

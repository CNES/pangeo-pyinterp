:tocdepth: 2

.. currentmodule:: pyinterp

API Documentation
#################

This page presents the high-level public Python API grouped by theme. For low-level
(C++ binding) symbols see :doc:`core_api`.

Coordinates & Axes
==================
Fundamental 1D axis helpers used by grid and interpolator objects.

.. autosummary::
  :toctree: generated/

  Axis
  AxisInt64
  TemporalAxis

CF Metadata Helpers
===================
Identify axis meaning from CF-compliant unit strings.

.. autosummary::
  :toctree: generated/

  cf.AxisLatitudeUnit
  cf.AxisLongitudeUnit
  cf.AxisTimeUnit
  cf.AxisUnit

Spatial Index
=============
Spatial search structures.

.. autosummary::
  :toctree: generated/

  RTree

Geohash
=======
Encode geographic coordinates into compact base32 strings and work with the
resulting discrete spatial grid.

.. autosummary::
  :toctree: generated/

  GeoHash
  geohash.bounding_boxes
  geohash.decode
  geohash.encode
  geohash.int64.decode
  geohash.int64.encode
  geohash.int64.neighbors
  geohash.transform
  geohash.where

Geodetic Utilities
==================
Geodetic primitives, distance helpers and spherical geometry utilities.

.. autosummary::
  :toctree: generated/

  geodetic.Box
  geodetic.coordinate_distances
  geodetic.Coordinates
  geodetic.Crossover
  geodetic.LineString
  geodetic.MultiPolygon
  geodetic.normalize_longitudes
  geodetic.Point
  geodetic.Polygon
  geodetic.RTree
  geodetic.Spheroid

Binning & Histograms
====================
One and two dimensional statistical accumulation utilities.

.. autosummary::
  :toctree: generated/

  Binning1D
  Binning2D
  Histogram2D

Cartesian Grids
===============
Regular rectilinear grid containers used with interpolators.

.. autosummary::
  :toctree: generated/

  grid.Grid2D
  grid.Grid3D
  grid.Grid4D

Filling Undefined Values
========================
Functions for filling missing values in grids.

.. autosummary::
  :toctree: generated/

  fill.fft_inpaint
  fill.gauss_seidel
  fill.loess
  fill.multi_grid

Interpolators
=============
Scalar field interpolation functions over Cartesian grids.

.. autosummary::
  :toctree: generated/

  interpolate1d
  bicubic
  bivariate
  trivariate
  quadrivariate

Univariate Statistics
=====================
Streaming descriptive statistics of 1D sample streams.

.. autosummary::
  :toctree: generated/

  DescriptiveStatistics
  StreamingHistogram

Orbit Interpolation
===================
Work with repeating satellite orbits and derived passes/swaths.

.. autosummary::
  :toctree: generated/

  calculate_orbit
  calculate_pass
  calculate_swath
  EquatorCoordinates
  Orbit
  Pass
  Swath

Xarray Backends
===============
Helpers for constructing interpolators directly from ``xarray.DataArray`` objects.

.. autosummary::
  :toctree: generated/

  backends.xarray.Grid2D
  backends.xarray.Grid3D
  backends.xarray.Grid4D
  backends.xarray.RegularGridInterpolator

Type Hints
==========
Type aliases for numpy array shapes used throughout the library.

.. automodule:: pyinterp.typing

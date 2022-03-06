:tocdepth: 2

.. currentmodule:: pyinterp

API Documentation
#################

Geographic indexers
===================

.. autosummary::
  :toctree: generated/

  Axis
  RTree
  TemporalAxis

geohash
-------

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

Binning
=======

.. autosummary::
  :toctree: generated/

  Binning1D
  Binning2D
  Histogram2D

Cartesian Grids
===============

.. autosummary::
  :toctree: generated/

  grid.Grid2D
  grid.Grid3D
  grid.Grid4D

Climate and Forecast
====================

Axis identification from CF attribute values.

.. autosummary::
  :toctree: generated/

  cf.AxisLatitudeUnit
  cf.AxisLongitudeUnit
  cf.AxisTimeUnit
  cf.AxisUnit

Geodetic systems
================

.. autosummary::
  :toctree: generated/

  geodetic.Box
  geodetic.Coordinates
  geodetic.Crossover
  geodetic.Linestring
  geodetic.Point
  geodetic.Polygon
  geodetic.System
  geodetic.coordinate_distances
  geodetic.normalize_longitudes

.. _cartesian_interpolators:

Cartesian interpolators
=======================

.. autosummary::
  :toctree: generated/

  bicubic
  bivariate
  trivariate
  quadrivariate

Fill undefined values
=====================

.. autosummary::
  :toctree: generated/

  fill.loess
  fill.gauss_seidel

Univariate statistics
=====================

.. autosummary::
  :toctree: generated/

  DescriptiveStatistics
  StreamingHistogram

Xarray
======

Construction of Cartesian interpolators from xarray.DataArray

.. autosummary::
  :toctree: generated/

  backends.xarray.Grid2D
  backends.xarray.Grid3D
  backends.xarray.Grid4D
  backends.xarray.RegularGridInterpolator

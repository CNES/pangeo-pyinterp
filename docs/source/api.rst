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

  geohash.area
  geohash.bounding_box
  geohash.bounding_boxes
  geohash.decode
  geohash.encode
  geohash.error
  geohash.grid_properties
  geohash.index
  geohash.lock
  geohash.neighbors
  geohash.storage
  geohash.where

Binning
=======

.. autosummary::
  :toctree: generated/

  Binning2D

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
  geodetic.Point
  geodetic.Polygon
  geodetic.System

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

Xarray
======

Construction of Cartesian interpolators from xarray.DataArray

.. autosummary::
  :toctree: generated/

  backends.xarray.Grid2D
  backends.xarray.Grid3D
  backends.xarray.Grid4D
  backends.xarray.RegularGridInterpolator
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

  geodetic.Box2D
  geodetic.Coordinates
  geodetic.Point2D
  geodetic.System

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

Xarray
======

Construction of Cartesian interpolators from xarray.DataArray

.. autosummary::
  :toctree: generated/

  backends.xarray.Grid2D
  backends.xarray.Grid3D
  backends.xarray.Grid4D
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
Spatial search structures and unstructured data interpolation.

.. autosummary::
  :toctree: generated/

  RTree3D
  RTree3DFloat32
  RTree3DFloat64
  inverse_distance_weighting
  kriging
  query
  radial_basis_function
  window_function

Geohash
=======
Encode geographic coordinates into compact base32 strings and work with the
resulting discrete spatial grid.

.. autosummary::
  :toctree: generated/

  geohash.GeoHash
  geohash.area
  geohash.bounding_boxes
  geohash.decode
  geohash.encode
  geohash.transform
  geohash.to_xarray

Binning & Histograms
====================
One and two dimensional statistical accumulation utilities.

.. autosummary::
  :toctree: generated/

  Binning1D
  Binning1DFloat32
  Binning1DFloat64
  Binning2D
  Binning2DFloat32
  Binning2DFloat64
  Histogram2D
  Histogram2DFloat32
  Histogram2DFloat64

Filling Undefined Values
========================
Functions for filling missing values in grids.

.. autosummary::
  :toctree: generated/

  fill.fft_inpaint
  fill.gauss_seidel
  fill.loess
  fill.multigrid

Interpolators
=============
Scalar field interpolation functions over Cartesian grids.

.. autosummary::
  :toctree: generated/

  univariate
  bivariate
  trivariate
  quadrivariate

Univariate Statistics
=====================
Streaming descriptive statistics of 1D sample streams.

.. autosummary::
  :toctree: generated/

  DescriptiveStatistics

Orbit Interpolation
===================
Work with repeating satellite orbits and derived passes/swaths.

.. autosummary::
  :toctree: generated/

  orbit.calculate_orbit
  orbit.calculate_pass
  orbit.calculate_swath
  orbit.EquatorCoordinates
  orbit.Orbit
  orbit.Pass
  orbit.Swath

Xarray Backends
===============
Helpers for constructing interpolators directly from ``xarray.DataArray`` objects.

.. autosummary::
  :toctree: generated/

  backends.xarray.Grid2D
  backends.xarray.Grid3D
  backends.xarray.Grid4D
  backends.xarray.RegularGridInterpolator

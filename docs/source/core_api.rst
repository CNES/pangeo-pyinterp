:tocdepth: 2

.. currentmodule:: pyinterp

Core API Documentation
######################

Low-level classes & functions implemented in the compiled extension module (``pyinterp.core``).
Most users should start with the high-level :doc:`api` overview.

NumPy Datetime Utilities
========================
Vectorized helpers for working with ``numpy.datetime64`` values.

.. autosummary::
  :toctree: generated/

  core.dateutils.date
  core.dateutils.datetime
  core.dateutils.timedelta_since_january
  core.dateutils.isocalendar
  core.dateutils.time
  core.dateutils.weekday

Bicubic & Spline Routines
=========================
Internal bicubic spline interpolation kernels.

.. autosummary::
  :toctree: generated/

  core.bicubic_float32
  core.bicubic_float64
  core.spline_float32
  core.spline_float64

Binning
=======
Core accumulation structures by dtype.

.. autosummary::
  :toctree: generated/

  core.Binning1DFloat64
  core.Binning1DFloat32
  core.Binning2DFloat64
  core.Binning2DFloat32
  core.Histogram2DFloat64
  core.Histogram2DFloat32

Bivariate Interpolators
=======================
Object oriented 2D interpolator classes.

.. autosummary::
  :toctree: generated/

  core.Bilinear2D
  core.BivariateInterpolator2D
  core.InverseDistanceWeighting2D
  core.Nearest2D

Bivariate Kernel Functions
==========================
Vectorized functional 2D interpolation routines.

.. autosummary::
  :toctree: generated/

  core.bivariate_int8
  core.bivariate_float32
  core.bivariate_float64

Cartesian Grids
===============
Typed grid containers.

.. autosummary::
  :toctree: generated/

  core.Grid2DInt8
  core.Grid2DFloat32
  core.Grid2DFloat64
  core.Grid3DInt8
  core.Grid3DFloat32
  core.Grid3DFloat64
  core.Grid4DInt8
  core.Grid4DFloat32
  core.Grid4DFloat64

Univariate Descriptive Statistics
=================================
Streaming statistics.

.. autosummary::
  :toctree: generated/

  core.DescriptiveStatisticsFloat32
  core.DescriptiveStatisticsFloat64
  core.StreamingHistogramFloat64
  core.StreamingHistogramFloat32

Geodetic System
===============
Geodetic primitives and spatial operations.

.. autosummary::
  :toctree: generated/

  core.geodetic.Box
  core.geodetic.Coordinates
  core.geodetic.Crossover
  core.geodetic.LineString
  core.geodetic.MultiPolygon
  core.geodetic.Point
  core.geodetic.Polygon
  core.geodetic.RTree
  core.geodetic.Spheroid

Geohash (Integer)
=================
Integer encoded geohash helpers.

.. autosummary::
  :toctree: generated/

  core.geohash.int64.decode
  core.geohash.int64.encode
  core.geohash.int64.neighbors

Geohash (Bytes)
===============
Byte encoded geohash helpers and transformations.

.. autosummary::
  :toctree: generated/

  core.GeoHash
  core.geohash.area
  core.geohash.bounding_boxes
  core.geohash.decode
  core.geohash.encode
  core.geohash.int64
  core.geohash.transform
  core.geohash.where

Temporal Cartesian Grids
========================
Grids extended with a time dimension.

.. autosummary::
  :toctree: generated/

  core.TemporalGrid3DFloat32
  core.TemporalGrid3DFloat64
  core.TemporalGrid4DFloat32
  core.TemporalGrid4DFloat64

4D Interpolation Kernels
========================
Functional 4D interpolation (space + time) routines.

.. autosummary::
  :toctree: generated/

  core.quadrivariate_float32
  core.quadrivariate_float64

R*Tree
======
Spatial index & associated interpolation kernels.

.. autosummary::
  :toctree: generated/

  core.RadialBasisFunction
  core.WindowFunction
  core.RTree3DFloat32
  core.RTree3DFloat64

Replace Undefined Values
========================
Gap filling routines and supporting types.

.. autosummary::
  :toctree: generated/

  core.fill.FirstGuess
  core.fill.ValueType
  core.fill.fft_inpaint_float32
  core.fill.fft_inpaint_float64
  core.fill.gauss_seidel_float32
  core.fill.gauss_seidel_float64
  core.fill.loess_float32
  core.fill.loess_float64
  core.fill.multigrid_float32
  core.fill.multigrid_float64

3D Interpolators
================
Object oriented 3D interpolator classes.

.. autosummary::
  :toctree: generated/

  core.Bilinear3D
  core.BivariateInterpolator3D
  core.InverseDistanceWeighting3D
  core.Nearest3D

Temporal 3D Interpolators
=========================
Object oriented 3D + time interpolator classes.

.. autosummary::
  :toctree: generated/

  core.TemporalBilinear3D
  core.TemporalBivariateInterpolator3D
  core.TemporalInverseDistanceWeighting3D
  core.TemporalNearest3D

3D Interpolation Kernels
========================
Vectorized functional 3D interpolation routines.

.. autosummary::
  :toctree: generated/

  core.trivariate_float32
  core.trivariate_float64

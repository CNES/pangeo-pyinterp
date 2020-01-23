.. currentmodule:: pyinterp

Core API Documentation
######################

Functions and classes implemented in the C++ module

Axis
----

.. autosummary::
  :toctree: generated/

  core.Axis
  core.AxisBoundary
  core.TemporalAxis

Bicubic interpolation
---------------------

.. autosummary::
  :toctree: generated/

  core.FittingModel
  core.bicubic_float32
  core.bicubic_float64

Binning
-------

.. autosummary::
  :toctree: generated/

  core.Binning2DFloat64
  core.Binning2DFloat32

Bivariate interpolators
-----------------------

.. autosummary::
  :toctree: generated/

  core.Bilinear2D
  core.BivariateInterpolator2D
  core.InverseDistanceWeighting2D
  core.Nearest2D

Bivariate interpolations
------------------------

.. autosummary::
  :toctree: generated/

  core.bivariate_float32
  core.bivariate_float64

Cartesian Grids
---------------

.. autosummary::
  :toctree: generated/

  core.Grid2DFloat32
  core.Grid2DFloat64
  core.Grid3DFloat32
  core.Grid3DFloat64
  core.Grid4DFloat32
  core.Grid4DFloat64

Geodetic System
---------------

.. autosummary::
  :toctree: generated/

  core.geodetic.Box2D
  core.geodetic.Coordinates
  core.geodetic.Point2D
  core.geodetic.System

Temporal Cartesian Grids
------------------------

.. autosummary::
  :toctree: generated/

  core.TemporalGrid3DFloat32
  core.TemporalGrid3DFloat64
  core.TemporalGrid4DFloat32
  core.TemporalGrid4DFloat64

4D interpolation
----------------

.. autosummary::
  :toctree: generated/

  core.quadrivariate_float32
  core.quadrivariate_float64

R*Tree
------
.. autosummary::
  :toctree: generated/

  core.RadialBasisFunction
  core.RTree3DFloat32
  core.RTree3DFloat64

Replace undefined values
------------------------

.. autosummary::
  :toctree: generated/

  core.fill.ValueType
  core.fill.FirstGuess
  core.fill.loess_float64
  core.fill.loess_float32
  core.fill.gauss_seidel_float64
  core.fill.gauss_seidel_float32

3D interpolators
----------------

.. autosummary::
  :toctree: generated/

  core.Bilinear3D
  core.BivariateInterpolator3D
  core.InverseDistanceWeighting3D
  core.Nearest3D

Temporal 3D interpolators
-------------------------

.. autosummary::
  :toctree: generated/

  core.TemporalBilinear3D
  core.TemporalBivariateInterpolator3D
  core.TemporalInverseDistanceWeighting3D
  core.TemporalNearest3D

3D interpolations
-----------------

.. autosummary::
  :toctree: generated/

  core.trivariate_float32
  core.trivariate_float64


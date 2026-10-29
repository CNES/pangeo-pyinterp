Configuration Objects
=====================
Configuration builders for the interpolation and filling functions. Each
object is created with its defaults and refined with chained ``with_*``
methods; pass it as the ``method`` (regular grids) or ``config`` (R-trees,
gap filling) argument instead of the equivalent keyword arguments, e.g.
``windowed.Bivariate.bicubic().with_half_window_size_x(10)``.

Windowed Interpolation on Regular Grids
---------------------------------------
Configurations of the windowed methods (splines, polynomials, Akima, ...) of
:doc:`regular_grid`.

.. currentmodule:: pyinterp.core.config.windowed

.. autosummary::
   :toctree: _generated/

   Univariate
   Bivariate
   Trivariate
   Quadrivariate
   AxisConfig
   BoundaryConfig

Geometric Interpolation on Regular Grids
----------------------------------------
Configurations of the geometric methods (bilinear, nearest, inverse distance
weighting) of :doc:`regular_grid`.

.. currentmodule:: pyinterp.core.config.geometric

.. autosummary::
   :toctree: _generated/

   Bivariate
   Trivariate
   Quadrivariate

Unstructured Grids
------------------
Configurations of the R-tree interpolators of :doc:`unstructured_grid` and
:doc:`optimal_interpolation`.

.. currentmodule:: pyinterp.core.config.rtree

.. autosummary::
   :toctree: _generated/

   InverseDistanceWeighting
   RadialBasisFunction
   Kriging
   InterpolationWindow
   OptimalInterpolation
   Query
   BoundaryCheck
   CovarianceFunction
   DriftFunction
   RBFKernel
   WindowKernel

Reduced Gaussian Grids
----------------------
Configuration of the interpolation of :doc:`reduced_gaussian`; the functions
of :mod:`pyinterp.reduced_gaussian` build it from their keyword arguments.

.. currentmodule:: pyinterp.core.config.reduced_gaussian

.. autosummary::
   :toctree: _generated/

   ReducedGaussian

Gap Filling
-----------
Configurations of the algorithms of :doc:`fill`.

.. currentmodule:: pyinterp.core.config.fill

.. autosummary::
   :toctree: _generated/

   FFTInpaint
   GaussSeidel
   Loess
   Multigrid
   FirstGuess
   LoessValueType

Optimal Interpolation
=====================

Optimal Interpolation (OI / BLUE) for scattered 4D observations, with an
anisotropic covariance kernel and a per-observation error variance. The
estimator is backed by a 4D R-tree that stores each observation together with
its measurement-error variance.

.. currentmodule:: pyinterp

Estimator
---------

.. autosummary::
   :toctree: _generated/

   OptimalInterpolation
   OIResult

Spatial Index (4D R-Tree)
-------------------------

The 4D R-tree is the indexing primitive feeding the estimator. Unlike
:py:func:`RTree3D() <pyinterp.RTree3D>`, it does not provide inverse-distance
weighting, kriging, RBF or window-function methods; it exposes k-nearest
neighbour queries in 4D space-time and the built-in ``optimal_interpolation``
backend.

.. autosummary::
   :toctree: _generated/

   RTree4D
   RTree4DFloat32
   RTree4DFloat64

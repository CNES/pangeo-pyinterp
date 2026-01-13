R*Tree Spatial Index
====================

.. currentmodule:: pyinterp.core

Spatial index structures and interpolation configuration for unstructured
point data. The R*Tree provides efficient nearest-neighbor queries and
supports various interpolation methods.

Configuration Classes
---------------------

Fluent builder classes for configuring interpolation methods. Each class
provides ``with_*`` methods for setting parameters.

.. autosummary::

   config.rtree.Query
   config.rtree.InverseDistanceWeighting
   config.rtree.RadialBasisFunction
   config.rtree.InterpolationWindow
   config.rtree.Kriging

Kernel Enumerations
-------------------

Kernel functions used by interpolation methods.

.. autosummary::

   config.rtree.RBFKernel
   config.rtree.WindowKernel

Spatial Index Classes
---------------------

R*Tree implementations by data type.

.. autosummary::

   RTree3DFloat32
   RTree3DFloat64
   RTree3D

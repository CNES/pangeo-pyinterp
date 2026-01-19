Interpolation on Unstructured Grids
===================================

Spatial index trees.

.. currentmodule:: pyinterp

Spatial Index (R-Tree)
----------------------

.. autosummary::
   :toctree: _generated/

   RTree3D
   RTree3DFloat32
   RTree3DFloat64

.. currentmodule:: pyinterp.rtree

Algorithms
----------
Interpolation algorithms for unstructured data.

.. autosummary::
   :toctree: _generated/

   inverse_distance_weighting
   radial_basis_function
   kriging
   window_function
   query

Core RTree
----------

Base Class and Type Hints
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``RTree3DHolder`` is the generic base class used in type hints for R-Tree
spatial indexes with support for different data types (float32 and float64). It
is returned by the factory functions :py:func:`RTree3D() <pyinterp.RTree3D>`,
:py:func:`RTree3DFloat32() <pyinterp.RTree3DFloat32>`, and
:py:func:`RTree3DFloat64() <pyinterp.RTree3DFloat64>`.

.. currentmodule:: pyinterp.core

.. py:class:: RTree3DHolder(Generic[_FloatDType])

   Generic holder for a 3D R-Tree spatial index supporting different floating-point types.

   :param spheroid: Optional spheroid for geographic coordinate transformations.
   :type spheroid: pyinterp.core.geometry.geographic.Spheroid | None

   .. py:method:: bounds() -> tuple[NDArray, NDArray] | None

      Get the bounding box of all inserted points.

   .. py:method:: clear() -> None

      Clear the R-Tree by removing all stored points.

   .. py:method:: empty() -> bool

      Check if the R-Tree is empty.

   .. py:method:: insert(coordinates: NDArray[TwoDims], values: NDArray[OneDim]) -> None

      Insert points and associated values into the R-Tree.

   .. py:method:: inverse_distance_weighting(coordinates: NDArray[TwoDims], config: rtree.InverseDistanceWeighting | None = None) -> tuple[NDArray[OneDim], NDArray[UInt32]]

      Perform Inverse Distance Weighting interpolation on the R-Tree.

   .. py:method:: kriging(coordinates: NDArray[TwoDims], config: rtree.Kriging | None = None) -> tuple[NDArray[OneDim], NDArray[UInt32]]

      Perform Kriging interpolation on the R-Tree.

   .. py:method:: packing(coordinates: NDArray[TwoDims], values: NDArray[OneDim]) -> None

      Pack points and values into the R-Tree for optimized queries.

   .. py:method:: query(coordinates: NDArray[TwoDims], config: rtree.Query | None = None) -> tuple[NDArray[TwoDims], NDArray[TwoDims]]

      Query the R-Tree for nearest neighbors.

   .. py:method:: radial_basis_function(coordinates: NDArray[TwoDims], config: rtree.RadialBasisFunction | None = None) -> tuple[NDArray[OneDim], NDArray[UInt32]]

      Perform Radial Basis Function interpolation on the R-Tree.

   .. py:method:: size() -> int

      Get the number of points stored in the R-Tree.

   .. py:method:: window_function(coordinates: NDArray[TwoDims], config: rtree.InterpolationWindow | None = None) -> tuple[NDArray[OneDim], NDArray[UInt32]]

      Perform window function interpolation on the R-Tree.

   .. py:attribute:: spheroid
      :type: pyinterp.core.geometry.geographic.Spheroid | None

      The spheroid associated with this R-Tree instance.

Geographic R-Tree Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specialized R-Tree implementation for geographic (spherical/ellipsoidal) coordinates.

.. currentmodule:: pyinterp.core.geometry.geographic

.. autosummary::
   :toctree: _generated/

   RTree

.. note::
   The :py:class:`pyinterp.core.geometry.geographic.RTree` class provides geographic-aware
   spatial indexing on spherical/ellipsoidal coordinates (longitude/latitude).
   This is the same class alias used throughout the library's high-level API
   as :py:class:`pyinterp.geometry.geographic.RTree`.

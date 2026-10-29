Interpolation on Unstructured Grids
===================================

Interpolation of scattered points. Build a spatial index with
:py:func:`RTree3D() <pyinterp.RTree3D>`, load the observations with its
``packing`` or ``insert`` methods, then interpolate with the functions of
``pyinterp.rtree`` listed below (inverse distance weighting, radial basis
functions, kriging or window functions).

.. currentmodule:: pyinterp

Spatial Index (R-Tree)
----------------------

.. autosummary::
   :toctree: _generated/

   RTree3D
   RTree3DFloat32
   RTree3DFloat64

.. module:: pyinterp.rtree

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
   :type spheroid: pyinterp.geometry.geographic.Spheroid | None

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
      :type: pyinterp.geometry.geographic.Spheroid | None

      The spheroid associated with this R-Tree instance.

Geographic R-Tree Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specialized R-Tree implementation for geographic (spherical/ellipsoidal)
coordinates: :py:class:`pyinterp.geometry.geographic.RTree`, documented with
the :doc:`geometric primitives <geometry>`.

Choosing between the two indexes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both accept longitude/latitude points. They differ by the distance they
measure and by their speed:

* :py:func:`RTree3D(spheroid) <pyinterp.RTree3D>`, created with a spheroid
  (without one, the coordinates are taken as Cartesian), converts the points
  to ECEF and searches them with straight-line (chord) distances. It is the
  fastest index, 10 to 20 times faster to query than the geographic one (the
  gap grows with the size of the tree), and also supports altitudes, any
  spheroid and single precision.
* ``RTree3D(spheroid, distance="geodesic")`` keeps the ECEF search but ranks
  the neighbors by geodesic distance on the spheroid, and returns geodesic
  distances: the same results as the geographic index, 5 to 10 times faster.
  The points must have no altitude. Radial basis functions and kriging keep
  straight-line distances between the neighbors in their kernels.
* :py:class:`geographic.RTree <pyinterp.geometry.geographic.RTree>`
  uses geodesic distances on the WGS84 ellipsoid, and evaluates them to
  search its nodes, which makes it the slowest.

The chord is shorter than the geodesic by about 0.001% at 100 km, 0.1% at
1,000 km and 2.6% at 5,000 km: at short range, the default ``RTree3D`` finds
the same neighbors as the geodesic searches, except for near-ties. Use
``distance="geodesic"`` when the distances, or the weights derived from them,
must be geodesic over long ranges.

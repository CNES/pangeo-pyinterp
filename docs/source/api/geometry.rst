Geometric Primitives
====================
Geometric primitives and spatial algorithms.

.. currentmodule:: pyinterp.geometry

This module provides a comprehensive set of geometric primitives (points,
lines, polygons) and spatial algorithms. It is built on the Boost.Geometry
library and organized into two namespaces based on the coordinate system:

* **Geographic**: For coordinates on the Earth's surface (Long/Lat).
  Calculations account for the Earth's curvature (ellipsoid/sphere).
* **Cartesian**: For planar coordinates (X/Y). Calculations use Euclidean
  geometry.

.. note::
   The API is symmetrical: both namespaces provide the same classes
   (e.g., ``Point``, ``Polygon``) and algorithms (e.g., ``intersection``,
   ``distance``), but their internal implementation differs to suit the
   coordinate system.

Geographic System
-----------------
Geographic (spherical/ellipsoidal) geometry.

.. currentmodule:: pyinterp.geometry.geographic

Primitives
^^^^^^^^^^
Data structures for representing spatial features defined by Longitude and Latitude.

.. autosummary::
   :toctree: _generated/

   Point
   Segment
   LineString
   Ring
   Box
   Polygon
   MultiPoint
   MultiLineString
   MultiPolygon

Coordinate Reference Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Classes for managing coordinate conversions and defining the Earth's shape.

.. autosummary::
   :toctree: _generated/

   Coordinates
   Spheroid

Algorithms
^^^^^^^^^^
Spatial operations and predicates.

.. autosummary::
   :toctree: _generated/

   algorithms.area
   algorithms.azimuth
   algorithms.centroid
   algorithms.closest_points
   algorithms.convert_to_cartesian
   algorithms.convex_hull
   algorithms.covered_by
   algorithms.crosses
   algorithms.densify
   algorithms.difference
   algorithms.disjoint
   algorithms.distance
   algorithms.envelope
   algorithms.equals
   algorithms.intersection
   algorithms.intersects
   algorithms.is_empty
   algorithms.is_simple
   algorithms.is_valid
   algorithms.length
   algorithms.perimeter
   algorithms.simplify
   algorithms.touches
   algorithms.union
   algorithms.within

Distance Strategies
^^^^^^^^^^^^^^^^^^^
Strategies for computing geodesic distances on the Earth's surface.

.. autosummary::
   :toctree: _generated/

   algorithms.Strategy
   algorithms.ANDOYER
   algorithms.KARNEY
   algorithms.THOMAS
   algorithms.VINCENTY

Cartesian System
----------------
Cartesian (planar) geometry.

.. currentmodule:: pyinterp.geometry.cartesian

Primitives
^^^^^^^^^^
Data structures for representing spatial features defined by X and Y coordinates.

.. autosummary::
   :toctree: _generated/

   Point
   Segment
   LineString
   Ring
   Box
   Polygon
   MultiPoint
   MultiLineString
   MultiPolygon

Algorithms
^^^^^^^^^^
Spatial operations and predicates in Euclidean space.

.. autosummary::
   :toctree: _generated/

   algorithms.area
   algorithms.azimuth
   algorithms.buffer
   algorithms.centroid
   algorithms.closest_points
   algorithms.convert_to_geographic
   algorithms.convex_hull
   algorithms.covered_by
   algorithms.crosses
   algorithms.densify
   algorithms.difference
   algorithms.disjoint
   algorithms.distance
   algorithms.envelope
   algorithms.equals
   algorithms.intersection
   algorithms.intersects
   algorithms.is_empty
   algorithms.is_simple
   algorithms.is_valid
   algorithms.length
   algorithms.perimeter
   algorithms.simplify
   algorithms.touches
   algorithms.union
   algorithms.within

Buffer Strategies
^^^^^^^^^^^^^^^^^
Configuration for the buffering algorithm (Cartesian only).

.. autosummary::
   :toctree: _generated/

   algorithms.DistanceAsymmetric
   algorithms.DistanceSymmetric
   algorithms.EndFlat
   algorithms.EndRound
   algorithms.JoinMiter
   algorithms.JoinRound
   algorithms.PointCircle
   algorithms.PointSquare
   algorithms.SideStraight

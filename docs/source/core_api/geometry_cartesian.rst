Cartesian Geometry
==================

.. currentmodule:: pyinterp.core.geometry.cartesian

Geometric data structures and algorithms for Cartesian (planar X/Y) coordinate
systems. Calculations use Euclidean geometry.

Data Structures
---------------

Geometric primitives for representing spatial features in a Cartesian plane.

.. autosummary::

   Point
   Segment
   LineString
   Box
   Polygon
   MultiLineString
   MultiPolygon
   MultiPoint

Buffer Strategies
-----------------

Configuration options for the buffer algorithm, controlling how corners,
endpoints, and distances are handled.

.. autosummary::

   algorithms.DistanceAsymmetric
   algorithms.DistanceSymmetric
   algorithms.EndFlat
   algorithms.EndRound
   algorithms.JoinMiter
   algorithms.JoinRound
   algorithms.PointCircle
   algorithms.PointSquare
   algorithms.SideStraight

Algorithms
----------

Spatial operations and predicates for Cartesian geometries.

.. autosummary::

   algorithms.area
   algorithms.azimuth
   algorithms.buffer
   algorithms.centroid
   algorithms.clear
   algorithms.closest_points
   algorithms.convert_to_geographic
   algorithms.convex_hull
   algorithms.correct
   algorithms.covered_by
   algorithms.crosses
   algorithms.densify
   algorithms.difference
   algorithms.disjoint
   algorithms.distance
   algorithms.envelope
   algorithms.equals
   algorithms.for_each_point_covered_by
   algorithms.for_each_point_distance
   algorithms.for_each_point_within
   algorithms.from_geojson
   algorithms.from_wkt
   algorithms.intersection
   algorithms.intersects
   algorithms.is_empty
   algorithms.is_simple
   algorithms.is_valid
   algorithms.length
   algorithms.line_interpolate
   algorithms.num_geometries
   algorithms.num_interior_rings
   algorithms.num_points
   algorithms.num_segments
   algorithms.overlaps
   algorithms.perimeter
   algorithms.relate
   algorithms.relation
   algorithms.reverse
   algorithms.simplify
   algorithms.to_geojson
   algorithms.to_wkt
   algorithms.touches
   algorithms.union
   algorithms.unique
   algorithms.within

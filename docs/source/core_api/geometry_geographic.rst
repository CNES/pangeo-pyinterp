Geographic Geometry
===================

.. currentmodule:: pyinterp.core.geometry.geographic

Geometric data structures and algorithms for geographic (latitude/longitude)
coordinate systems. Calculations account for Earth's curvature using spherical
or ellipsoidal models.

Data Structures
---------------

Geometric primitives for representing spatial features on Earth's surface.

.. autosummary::

   Point
   Segment
   LineString
   Box
   Polygon
   MultiLineString
   MultiPolygon
   MultiPoint

Distance Strategies
-------------------

Algorithms for computing geodesic distances. Different strategies offer
trade-offs between accuracy and computational speed.

.. autosummary::

   algorithms.Strategy
   algorithms.ANDOYER
   algorithms.KARNEY
   algorithms.THOMAS
   algorithms.VINCENTY

Algorithms
----------

Spatial operations and predicates for geographic geometries.

.. autosummary::

   algorithms.area
   algorithms.azimuth
   algorithms.centroid
   algorithms.clear
   algorithms.closest_points
   algorithms.convert_to_cartesian
   algorithms.convex_hull
   algorithms.correct
   algorithms.covered_by
   algorithms.crosses
   algorithms.curvilinear_distance
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

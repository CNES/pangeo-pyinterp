Geometry
========

.. currentmodule:: pyinterp

Geometric data structures and algorithms for both geographic (spherical/ellipsoidal)
and Cartesian (planar) coordinate systems. The geometry module provides a comprehensive
set of spatial primitives and operations built on the Boost.Geometry library.

The module is organized into two coordinate system namespaces:

- **Geographic**: For latitude/longitude coordinates on Earth's surface
- **Cartesian**: For planar X/Y coordinates

Each namespace provides the same set of geometric primitives (Point, LineString,
Polygon, etc.) and algorithms (intersection, union, distance, etc.), but with
calculations appropriate for the coordinate system.

.. toctree::
   :maxdepth: 2
   :hidden:

   geometry_geographic
   geometry_cartesian

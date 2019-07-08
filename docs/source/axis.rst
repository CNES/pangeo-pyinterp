Geographic indexers
-------------------

N-Dimensional Grids
===================

N-dimensional grid is a grid defined by a matrix, in a 2D space, by a cube in a
3D space, etc. Each dimension of the grid is associated with a vector
corresponding to its coordinates or axes. Axes are used to locate a pixel in
the grid from the coordinates of a point. These axes are either:

* regular: latitudes are defined by a vector of 181 values spaced a degree from
  -90 to 90 degrees;
* irregular: latitudes are represented by a vector of 109 values irregularly
  spaced from -90 to 89.940374 degrees.

These objects are manipulated by the class :py:class:`pyinterp.core.Axis` which
will choose, according to the definition of the axis, the best implementation.
This object will allow you to find the two indexes framing a given value. This
operating mode allows better performance when searching for a regular axis (a
simple calculation allows you to immediately find the index of a point) while
in the case of an irregular axis, the search will be performed using a binary
search.

Finally, this object is able to define a circular axis from a vector in order
to correctly locate a value on the circle. This is the type of axis that will
be used when handling longitudes.

Unstructured Grids
==================

In the case of unstructured grids, the index used is an :py:class:`R*Tree
<pyinterp.rtree.RTree>`. These trees have better performance than the *KDTree*
generally found in Python library implementations.

The tree used here is the implementation provided by the `C++ Boost library
<https://www.boost.org/doc/libs/1_70_0/libs/geometry/doc/html/geometry/reference/spatial_indexes/boost__geometry__index__rtree.html>`_.
An adaptation has been introduced to effectively address spherical equatorial
coordinates. Although the Boost library allows these coordinates to be
manipulated natively, but the performance is lower than in the case of a
Cartesian space. Thus, we have chosen to implement a conversion of Longitude
Latitude Altitude (LLA) coordinates into Earth-Centered, Earth-Fixed (ECEF)
coordinates in a transparent way for the user to ensure that we are able to
preserve good performance. The disadvantage of this implementation is that it
requires a little more memory, as one more element must be used to index the
value of the Cartesian space.

The management of the LLA/ECEF coordinate conversion is managed to use the
`Olson, D.K. <https://ieeexplore.ieee.org/document/481290>`_ algorithm. It has
excellent performance with an accuracy of 1e-8 meters for altitude.


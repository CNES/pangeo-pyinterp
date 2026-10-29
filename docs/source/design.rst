Design Notes
############

This page explains how pyinterp locates points in grids and in scattered
data, and how it handles geographic coordinates.

Axes
====

An N-dimensional grid is a matrix in a 2D space, a cube in a 3D space, and so
on. Each dimension of the grid is associated with a vector of coordinates, its
axis, which is used to locate the cell containing a point. An axis is either:

* *regular*: for example, a vector of 181 latitudes spaced one degree apart
  from -90 to 90 degrees;
* *irregular*: for example, a vector of 109 latitudes irregularly spaced from
  -90 to 89.940374 degrees.

The class :class:`pyinterp.Axis` chooses the best implementation from the
values of the axis, and finds the two indexes framing a given value. On a
regular axis, the index of a point is computed directly; on an irregular axis,
it is found by a binary search.

An axis can also be periodic, to locate values on a circle: this is how
longitudes are handled, so that a grid stored in [0, 360) can be queried in
[-180, 180) or across the antimeridian.

Temporal Axes
=============

The class :class:`pyinterp.TemporalAxis` handles axes of dates, stored as
64-bit integers, the encoding of the `NumPy datetime64
<https://numpy.org/doc/stable/reference/arrays.datetime.html>`_ type. The
dates are processed with integer arithmetic, so that no information is lost
during the calculations. Temporal axes are used by the spatiotemporal grids to
interpolate in time.

Scattered Data
==============

Scattered data are indexed in an R*-tree, which performs better than the
k-d trees found in most Python libraries. The tree is the implementation of
the `Boost C++ libraries
<https://www.boost.org/doc/libs/release/libs/geometry/doc/html/geometry/reference/spatial_indexes/boost__geometry__index__rtree.html>`_.

Boost can index spherical equatorial coordinates natively, but more slowly
than Cartesian coordinates. :class:`pyinterp.RTree3D` therefore converts the
longitudes, latitudes and altitudes (LLA) into Earth-Centered, Earth-Fixed
(ECEF) coordinates, transparently for the user. This keeps the performance of
the Cartesian space, at the cost of some memory, as one more coordinate is
stored for each point.

The LLA to ECEF conversions
(:meth:`~pyinterp.geometry.geographic.Coordinates.lla_to_ecef` and
:meth:`~pyinterp.geometry.geographic.Coordinates.ecef_to_lla`) use the
algorithm of `Vermeille, H. <https://doi.org/10.1007/s00190-002-0273-6>`_,
which is fast and accurate to 1e-8 meters in altitude.

Geohash
=======

Geohashing encodes geographic coordinates (latitude and longitude) into a
short string of digits and letters that delineates an area on a map, called a
cell. The more characters in the string, the smaller the cell and the more
precise the location.

Geohashes use a base-32 alphabet: the digits ``0`` to ``9`` and the letters
except ``a``, ``i``, ``l`` and ``o``. A geohash is a compact way to store or
index a location in a database; see :mod:`pyinterp.geohash`.

Geometry
========

:mod:`pyinterp.geometry` provides bindings to the concepts and algorithms of
`Boost.Geometry
<https://www.boost.org/doc/libs/release/libs/geometry/doc/html/index.html>`_
in Cartesian and geographic coordinates: points, linestrings, polygons, rings,
segments and boxes, with areas, perimeters, distances, azimuths, centroids,
closest points, buffers and envelopes. In geographic coordinates, the
calculations on the spheroid can use the Andoyer, Karney, Thomas or Vincenty
strategies.

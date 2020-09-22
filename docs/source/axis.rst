Geographic indexers
-------------------

N-Dimensional Grids
===================

N-dimensional grid is a grid defined by a matrix, in a 2D space, by a cube in a
3D space, etc. Each dimension of the grid is associated with a vector
corresponding to its coordinates or axes. Axes used to locate a pixel in the
grid from the coordinates of a point. These axes are either:

* **regular**: a vector of 181 latitudes spaced a degree from -90 to 90
  degrees;
* **irregular**: a vector of 109 latitudes irregularly spaced from -90 to
  89.940374 degrees.

These objects are manipulated by the class :py:class:`pyinterp.Axis`, which will
choose, according to Axis definition, the best implementation. This object will
allow you to find the two indexes framing a given value. This operating mode
allows better performance when searching for a regular axis (a simple
calculation will enable you to see the index of a point immediately). In
contrast, in the case of an irregular axis, the search will be performed using a
binary search.

Finally, this class can define a circular axis from a vector to correctly
locate a value on the circle. This type of Axis will is used handling
longitudes.

Temporal Axes
=============

The :py:class:`pyinterp.TemporalAxis` class handles temporal axes, i.e., axes
defined by 64-bit integer vectors, which is the encoding used by `numpy
<https://docs.scipy.org/doc/numpy/reference/arrays.datetime.html>`_ to control
dates. This class allows handling dates without loss of information when the
precision of the times is the nanosecond. These objects are used by
spatiotemporal grids to perform temporal interpolations.

Unstructured Grids
==================

In the case of unstructured grids, the index used is a :py:class:`R*Tree
<pyinterp.RTree>`. These trees have better performance than the *KDTree*
generally found in Python library implementations.

The tree used here is the implementation provided by the `C++ Boost library
<https://www.boost.org/doc/libs/1_70_0/libs/geometry/doc/html/geometry/reference/spatial_indexes/boost__geometry__index__rtree.html>`_.
An adaptation has introduced to address spherical equatorial coordinates
effectively. Although the Boost library allows these coordinates to manipulated
natively, the performance is lower than in the case of Cartesian space. Thus, we
have chosen to implement a conversion of Longitude Latitude Altitude (``LLA``)
coordinates into Earth-Centered, Earth-Fixed (``ECEF``) coordinates
transparently for the user to ensure that we can preserve excellent performance.
The disadvantage of this implementation is that it requires a little more
memory, as one more element gets used to index the value of the Cartesian space.

The management of the ``LLA``/``ECEF`` coordinate conversion is managed to use
the `Olson, D.K. <https://ieeexplore.ieee.org/document/481290>`_ algorithm. It
has excellent performance with an accuracy of 1e-8 meters for altitude.

Geohash
=======

Geohashing is a geocoding method used to encode geographic coordinates
(latitude and longitude) into a short string of digits and letters delineating
an area on a map, which is called a cell, with varying resolutions. The more
characters in the string, the more precise the location.

Geohashes use Base-32 alphabet encoding (characters can be ``0`` to ``9`` and
``A`` to ``Z``, excl ``A``, ``I``, ``L`` and ``O``).

The figure below illustrates the cells and associated geohash codes for a
two-character resolution.

This method is used to build a :py:mod:`geographic index
<pyinterp.geohash.index>`, possibly stored on disk, for the purpose of indexing
data.

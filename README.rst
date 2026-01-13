###############
pangeo-pyinterp
###############

|Build Status| |Downloads| |Platforms|
|Latest Release Date| |License| |Binder|

Python library for optimized geo-referenced interpolation.

About
=====

The motivation of this project is to provide tools for interpolating
geo-referenced data used in the field of geosciences. Other libraries cover this
problem, but written entirely in Python, the performance of these projects was
not quite sufficient for our needs. That is why this project started.

With this library, you can interpolate `2D
<https://cnes.github.io/pangeo-pyinterp/generated/pyinterp.grid.Grid2D.html#pyinterp.grid.Grid2D>`_,
`3D
<https://cnes.github.io/pangeo-pyinterp/generated/pyinterp.grid.Grid3D.html#pyinterp.grid.Grid3D>`_,
or `4D
<https://cnes.github.io/pangeo-pyinterp/generated/pyinterp.grid.Grid4D.html#pyinterp.grid.Grid4D>`_
fields using ``n-variate`` and ``bicubic`` `interpolators
<https://cnes.github.io/pangeo-pyinterp/api.html#cartesian-interpolators>`_
and `unstructured grids
<https://cnes.github.io/pangeo-pyinterp/generated/pyinterp.RTree.html>`_.
You can also apply for a data `binning
<https://cnes.github.io/pangeo-pyinterp/api.html#binning>`_ on the
bivariate area by simple or linear binning.

The library core is written in C++ using the `Boost C++ Libraries
<https://www.boost.org/>`_, `Eigen3 <http://eigen.tuxfamily.org/>`_ and
`nanobind <https://github.com/wjakob/nanobind/>`_ libraries.

This software also uses `CMake <https://cmake.org/>`_ to configure the project
and `Googletest <https://github.com/google/googletest>`_ to perform unit testing
of the library kernel.

Fill undefined values
=====================

The undefined values in the grids do not allow interpolation of values located
in the neighborhood. This behavior is a concern when you need to interpolate
values near the mask of some fields. The library provides utilities to fill the
undefined values:

* `fft_inpaint <https://cnes.github.io/pangeo-pyinterp/generated/pyinterp.fill.fft_inpaint.html>`_
  to fill all undefined values in a grid using the FFT Inpainting method.
* `loess <https://cnes.github.io/pangeo-pyinterp/generated/pyinterp.fill.loess.html>`_
  to fill the undefined values on the boundary between the defined/undefined
  values using local regression.
* `gauss_seidel <https://cnes.github.io/pangeo-pyinterp/generated/pyinterp.fill.gauss_seidel.html>`_
  to fill all undefined values in a grid using the Gauss-Seidel method by
  relaxation.
* `multi_grid <https://cnes.github.io/pangeo-pyinterp/generated/pyinterp.fill.multi_grid.html>`_
  to fill all undefined values in a grid using the Multi-Grid method.

Geographic indexers
===================

N-Dimensional Grids
-------------------

N-dimensional grid is a grid defined by a matrix, in a 2D space, by a cube in a
3D space, etc. Each dimension of the grid is associated with a vector
corresponding to its coordinates or axes. Axes used to locate a pixel in the
grid from the coordinates of a point. These axes are either:

* *regular*: a vector of 181 latitudes spaced a degree from -90 to 90 degrees;
* *irregular*: a vector of 109 latitudes irregularly spaced from -90 to
  89.940374 degrees.

These objects are manipulated by the class `pyinterp.Axis
<https://cnes.github.io/pangeo-pyinterp/generated/pyinterp.Axis.html>`_,
which will choose, according to Axis definition, the best implementation. This
object will allow you to find the two indexes framing a given value. This
operating mode allows better performance when searching for a regular axis (a
simple calculation will enable you to see the index of a point immediately). In
contrast, in the case of an irregular axis, the search will be performed using a
binary search.

Finally, this class can define a circular axis from a vector to correctly
locate a value on the circle. This type of Axis will is used for handling
longitudes.

Temporal Axes
-------------

The `pyinterp.TemporalAxis
<https://cnes.github.io/pangeo-pyinterp/generated/pyinterp.TemporalAxis.html>`_
class handles temporal axes, i.e., axes defined by 64-bit integer vectors, which
is the encoding used by `numpy
<https://docs.scipy.org/doc/numpy/reference/arrays.datetime.html>`_ to control
dates. This class allows you to process dates using integer arithmetic to ensure
that no information is lost during calculations. These objects are used by
spatiotemporal grids to perform temporal interpolations.

Unstructured Grids
------------------

In the case of unstructured grids, the index used is a *R\*Tree*. These trees
have better performance than the *KDTree* generally found in Python library
implementations.

The tree used here is the implementation provided by the `C++ Boost library
<https://www.boost.org/doc/libs/1_70_0/libs/geometry/doc/html/geometry/reference/spatial_indexes/boost__geometry__index__rtree.html>`_.

An adaptation has been introduced to address spherical equatorial coordinates
effectively. Although the Boost library allows these coordinates to manipulate
natively, the performance is lower than in the case of Cartesian space. Thus, we
have chosen to implement a conversion of Longitude Latitude Altitude (*LLA*)
coordinates into Earth-Centered, Earth-Fixed (*ECEF*) coordinates transparently
for the user to ensure that we can preserve excellent performance. The
disadvantage of this implementation is that it requires fairly more memory, as
one more element gets used to index the value of the Cartesian space.

The management of the `LLA
<https://cnes.github.io/pangeo-pyinterp/generated/pyinterp.geodetic.Coordinates.ecef_to_lla.html>`_/`ECEF
<https://cnes.github.io/pangeo-pyinterp/generated/pyinterp.geodetic.Coordinates.lla_to_ecef.html>`_
coordinate conversion is managed to use the `Vermeille, H.
<https://doi.org/10.1007/s00190-002-0273-6>`_ algorithm. It has an excellent
performance with the accuracy of 1e-8 meters for altitude.

Geometry
--------

This library provides Python bindings to `Boost Geometry
<https://www.boost.org/doc/libs/release/libs/geometry/doc/html/index.html>`_
concepts and algorithms for both geographic and Cartesian coordinate spaces.
The geometry module offers a comprehensive set of geometric primitives including
points, linestrings, polygons, rings, segments, and bounding boxes. For each
coordinate space (``cartesian`` and ``geographic``), dedicated algorithms are
provided for common geometric operations such as:

* Computing areas and perimeters of geometries
* Calculating distances and azimuths between points
* Determining centroids and closest points
* Performing buffer and envelope operations
* Converting between coordinate spaces

In the geographic space, additional features include support for multiple
geodetic strategies (Andoyer, Karney, Thomas, Vincenty) for precise calculations
on a spheroid, and coordinate transformations (LLA/ECEF, geographic to Cartesian
conversions).

Geohash
-------

Geohashing is a geocoding method used to encode geographic coordinates
(latitude and longitude) into a short string of digits and letters delineating
an area on a map, which is called a cell, with varying resolutions. The more
characters in the string, the more precise the location.

Geohashes use Base-32 alphabet encoding (characters can be ``0`` to ``9`` and
``A`` to ``Z``, excl ``A``, ``I``, ``L`` and ``O``).

The geohash is a compact way of representing a location, and is useful for
storing a location in a database, or for indexing a location in a database.


.. |Build Status| image:: https://github.com/CNES/pangeo-pyinterp/actions/workflows/ci.yaml/badge.svg
    :target: https://github.com/CNES/pangeo-pyinterp/actions
.. |Downloads| image:: https://img.shields.io/conda/dn/conda-forge/pyinterp.svg?service=github
    :target: image:: https://anaconda.org/conda-forge/pyinterp
.. |Platforms| image:: https://img.shields.io/conda/pn/conda-forge/pyinterp.svg?service=github
    :target: https://anaconda.org/conda-forge/pyinterp
.. |Latest Release Date| image:: https://anaconda.org/conda-forge/pyinterp/badges/latest_release_date.svg?service=github
    :target: https://github.com/CNES/pangeo-pyinterp/commits/master
.. |License| image:: https://anaconda.org/conda-forge/pyinterp/badges/license.svg?service=github
    :target: https://opensource.org/licenses/BSD-3-Clause
.. |Binder| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/CNES/pangeo-pyinterp/master?urlpath=lab/tree/notebooks/auto_examples/

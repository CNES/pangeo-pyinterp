[![Build Status](https://dev.azure.com/fbriol/pangeo-pyinterp/_apis/build/status/CNES.pangeo-pyinterp?branchName=master)](https://dev.azure.com/fbriol/pangeo-pyinterp/_build/latest?definitionId=2&branchName=master)
[![conda](https://anaconda.org/conda-forge/pyinterp/badges/installer/conda.svg?service=github)](https://www.anaconda.com/distribution/)
[![platforms](https://anaconda.org/conda-forge/pyinterp/badges/platforms.svg?service=github)](https://anaconda.org/conda-forge/pyinterp)
[![latest-release-date](https://anaconda.org/conda-forge/pyinterp/badges/latest_release_date.svg?service=github)](https://github.com/CNES/pangeo-pyinterp/commits/master)
[![license](https://anaconda.org/conda-forge/pyinterp/badges/license.svg?service=github)](https://opensource.org/licenses/BSD-3-Clause)
[![Binder](https://binder.pangeo.io/badge_logo.svg)](https://binder.pangeo.io/v2/gh/CNES/pangeo-pyinterp/master?filepath=notebooks)


# pangeo-pyinterp
Python library for optimized geo-referenced interpolation.

## About
The motivation of this project is to provide tools for interpolating
geo-referenced data used in the field of geosciences. There are other libraries
that cover this problem, but written entirely in Python, the performance of
these projects was not quite sufficient for our needs. That is why this project
was created.

This version can interpolate 2D fields using `bivariate` and `bicubic`
interpolators, 3D fields using `trivariate` interpolators and `unstructured
grid`. You can also apply a data `binning` on bivariate field by simple or linear
binning.

The library core is written in C++ using the [Boost C++
Libararies](https://www.boost.org/), [Eigen3](http://eigen.tuxfamily.org/),
[GNU Scientific Library](https://www.gnu.org/software/gsl/) and
[pybind11](https://github.com/pybind/pybind11/) libraries.

This software also uses [CMake](https://cmake.org/) to configure the project
and [Googletest](https://github.com/google/googletest) to perform unit testing
of the library kernel.

## Fill undefined values

The undefined values in the grids do not allow interpolation of values located
in the neighborhood. This behavior is a concern when you need to interpolate
values near the mask of some fields. The library provides utilities to fill the
undefined values:

* `loess` to fill the undefined values on the boundary between the defined/undefined
  values using local regression.
* `gauss_seidel` to fill all undefined values in a grid using the Gauss-Seidel
  method by relaxation.

## Geographic indexers

### N-Dimensional Grids

N-dimensional grid is a grid defined by a matrix, in a 2D space, by a cube in a
3D space, etc. Each dimension of the grid is associated with a vector
corresponding to its coordinates or axes. Axes are used to locate a pixel in
the grid from the coordinates of a point. These axes are either:

* *regular*: latitudes are defined by a vector of 181 values spaced a degree
  from -90 to 90 degrees;
* *irregular*: latitudes are represented by a vector of 109 values
  irregularly spaced from -90 to 89.940374 degrees.

These objects are manipulated by the class `pyinterp.core.Axis` which
will choose, according to the definition of the axis, the best implementation.
This object will allow you to find the two indexes framing a given value. This
operating mode allows better performance when searching for a regular axis (a
simple calculation allows you to immediately find the index of a point) while
in the case of an irregular axis, the search will be performed using a binary
search.

Finally, this object is able to define a circular axis from a vector in order
to correctly locate a value on the circle. This is the type of axis that will
be used when handling longitudes.

### Unstructured Grids

In the case of unstructured grids, the index used is a *R\*Tree*. These trees
have better performance than the *KDTree* generally found in Python library
implementations.

The tree used here is the implementation provided by the [C++ Boost
library](https://www.boost.org/doc/libs/1_70_0/libs/geometry/doc/html/geometry/reference/spatial_indexes/boost__geometry__index__rtree.html).

An adaptation has been introduced to effectively address spherical equatorial
coordinates. Although the Boost library allows these coordinates to be
manipulated natively, but the performance is lower than in the case of a
Cartesian space. Thus, we have chosen to implement a conversion of Longitude
Latitude Altitude (*LLA*) coordinates into Earth-Centered, Earth-Fixed (*ECEF*)
coordinates in a transparent way for the user to ensure that we are able to
preserve good performance. The disadvantage of this implementation is that it
requires a little more memory, as one more element must be used to index the
value of the Cartesian space.

The management of the *LLA*/*ECEF* coordinate conversion is managed to use the
[Olson, D.K.](https://ieeexplore.ieee.org/document/481290) algorithm. It has
excellent performance with an accuracy of 1e-8 meters for altitude.


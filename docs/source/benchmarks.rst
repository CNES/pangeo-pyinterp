Benchmarks
##########

This page compares pyinterp with the tools commonly used for the same tasks:
`SciPy <https://scipy.org/>`_, `xarray <https://xarray.dev/>`_ and `xESMF
<https://xesmf.readthedocs.io/>`_. The libraries do not all compute the same
thing in the same conditions, so the differences that matter are described
before the timings.

Longitudes Wrap Around
======================

pyinterp handles periodic axes: a global grid stored in [0, 360) can be
queried in [-180, 180), beyond 360, and in its last cell, between the last
longitude and the first one plus 360. ``xarray.DataArray.interp`` and
``scipy.interpolate.RegularGridInterpolator`` do not: they return NaN for
these points, or wrong values when extrapolation is enabled. With a grid of
0.25 degree storing :math:`\cos(\lambda)`:

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Longitude
     - -10
     - 0.1
     - 359.9
     - 370
   * - Exact value
     - 0.984808
     - 0.999998
     - 0.999998
     - 0.984808
   * - ``xarray.DataArray.interp``
     - NaN
     - 0.999996
     - NaN
     - NaN
   * - ``xarray.DataArray.interp``, extrapolating
     - 1.000381
     - 0.999996
     - 1.000008
     - 1.001161
   * - ``scipy.interpolate.RegularGridInterpolator``
     - NaN
     - 0.999996
     - NaN
     - NaN
   * - pyinterp
     - 0.984808
     - 0.999996
     - 0.999996
     - 0.984808

With these libraries, the grid must be padded with a copy of its first
columns, or the coordinates brought back into the range of the grid. Only
``scipy.ndimage.map_coordinates`` wraps an axis (``mode="grid-wrap"``), on
regular grids whose coordinates are converted to indexes by the user. In the
timings below, scipy and xarray are given points inside the longitudes of the
grid, which favors them.

Setup
=====

The timings were measured on an Apple M3 Max (10 performance and 4
efficiency cores) under macOS, with Python 3.13, NumPy 2.5.4, SciPy 1.18.1,
xarray 2026.9.0, xESMF 0.9.2 and the development version of pyinterp
following 2026.10.0. Each time is the best of at least three runs. The source
is a global grid of 0.25 degree (1440 × 720 values). pyinterp is timed on one
thread and on all the cores (``num_threads=0``, the default); the other
libraries run on one thread, except the k-d tree of SciPy, timed with all its
workers too.

To run the comparison on your machine, from a clone of the repository and
with the libraries to compare installed:

.. code-block:: bash

   python -m benchmarks.compare

Bilinear Interpolation at Scattered Points
==========================================

One million points drawn at random over the globe, such as the measurements
of a satellite track:

.. list-table::
   :header-rows: 1
   :widths: 55 15 15 15

   * - Library
     - Time
     - Points per second
     - Speed-up
   * - ``scipy.interpolate.RegularGridInterpolator``
     - 108.5 ms
     - 9.2 M
     - 1.0
   * - ``xarray.DataArray.interp``
     - 116.2 ms
     - 8.6 M
     - 0.9
   * - ``scipy.ndimage.map_coordinates``, ``order=1``
     - 24.5 ms
     - 40.7 M
     - 4.4
   * - pyinterp, one thread
     - 21.1 ms
     - 47.5 M
     - 5.2
   * - pyinterp, all cores
     - 3.2 ms
     - 314.0 M
     - 34.1

On one thread, pyinterp is as fast as ``map_coordinates``, which works on
indexes rather than coordinates, and five times faster than
``RegularGridInterpolator``, on which ``xarray.DataArray.interp`` relies.

Regridding
==========

The global grid is interpolated onto a grid of 0.1 degree, from 80° S to
80° N (5.76 million points), with the bilinear method:

.. list-table::
   :header-rows: 1
   :widths: 55 15 15 15

   * - Library
     - Time
     - Points per second
     - Speed-up
   * - xESMF, applying the weights
     - 1534.3 ms
     - 3.8 M
     - 1.0
   * - ``scipy.interpolate.RegularGridInterpolator``
     - 143.9 ms
     - 40.0 M
     - 10.7
   * - ``xarray.DataArray.interp``
     - 34.4 ms
     - 167.2 M
     - 44.5
   * - ``da.pyinterp.interp``, one thread
     - 9.6 ms
     - 597.3 M
     - 159.1
   * - ``da.pyinterp.interp``, all cores
     - 3.9 ms
     - 1476.2 M
     - 393.2

xESMF first computes its weights, which took 49.9 s here; they can be saved
and reused for other fields on the same grids. xarray and pyinterp exploit
the structure of the target grid: each coordinate is located once on its axis
instead of once per point. The results of pyinterp and xESMF differ by at
most :math:`1.2 \times 10^{-5}`.

Conservative Regridding
=======================

The global grid of 0.25 degree is coarsened to a global grid of 1 degree:
each target cell receives the mean of the source values weighted by the area
of their overlap with the cell, which preserves the integral of the field.

.. list-table::
   :header-rows: 1
   :widths: 55 15 15 15

   * - Library
     - Time
     - Source cells per second
     - Speed-up
   * - xESMF, applying the weights
     - 123.7 ms
     - 8.4 M
     - 1.0
   * - ``da.pyinterp.regrid``, one thread
     - 1.2 ms
     - 868.4 M
     - 103.6
   * - ``da.pyinterp.regrid``, all cores
     - 0.7 ms
     - 1553.2 M
     - 185.3

xESMF first computes its weights, which took 11.7 s here. pyinterp needs no
weights to compute: the overlap of two longitude/latitude boxes is the product
of the overlaps of their sides, so the regridding applies two sparse 1D
operators along the longitudes and the latitudes.

The two libraries do not draw the cells in the same way. pyinterp bounds them
by meridians and parallels, while ESMF draws their edges as great circles,
which bulge towards the poles: the results differ by up to
:math:`2 \times 10^{-5}` of the amplitude of a smooth field at mid-latitudes
with the 1 degree grid, and :math:`4 \times 10^{-4}` with a grid of 5 degrees
(see :doc:`api/regrid`).

xESMF also regrids curvilinear and unstructured source grids, with the
bilinear and conservative methods, which pyinterp does not handle: its grid
interpolators and its conservative regridding need rectilinear grids, and its
R*-trees can only treat the nodes of the other grids as scattered points.

Cubic Interpolation
===================

One million points drawn at random, with a cubic method:

.. list-table::
   :header-rows: 1
   :widths: 55 15 15 15

   * - Library
     - Time
     - Points per second
     - Speed-up
   * - ``scipy.ndimage.map_coordinates``, ``order=3``
     - 56.6 ms
     - 17.7 M
     - 1.0
   * - pyinterp bicubic, 4 × 4 window, one thread
     - 232.4 ms
     - 4.3 M
     - 0.2
   * - pyinterp bicubic, 4 × 4 window, all cores
     - 26.7 ms
     - 37.5 M
     - 2.1
   * - pyinterp bicubic, 6 × 6 window, one thread
     - 436.2 ms
     - 2.3 M
     - 0.1
   * - pyinterp bicubic, 6 × 6 window, all cores
     - 45.4 ms
     - 22.0 M
     - 1.2

On one thread, ``map_coordinates`` is four times faster. The two methods
differ, though: ``map_coordinates`` evaluates a B-spline whose coefficients
are computed once for the whole grid (11 ms here, not included above), while
pyinterp fits a spline on a window around each point (``half_window_size_x``
and ``half_window_size_y``, 3 by default). The global B-spline cannot handle
undefined values: a single NaN in the grid turns all its coefficients, and
then all the interpolated values, into NaN. The window of pyinterp only
affects the points whose window holds an undefined value, and the gaps can be
filled beforehand with :mod:`pyinterp.fill`.

Scattered Observations
======================

One million observations drawn at random over the globe are interpolated at
one million other points, by inverse distance weighting of their 11 nearest
neighbors. SciPy has no geodetic tree: its k-d tree indexes the
Earth-centered (ECEF) coordinates of the observations, and the weighting is
written with NumPy. The comparison is made twice.

Given the longitudes and latitudes of the points to query, pyinterp converts
them to ECEF coordinates itself (``RTree3D(spheroid=...)``), while the time
of SciPy includes their conversion by NumPy, on one thread:

.. list-table::
   :header-rows: 1
   :widths: 55 15 15 15

   * - Library
     - Time
     - Points per second
     - Speed-up
   * - ``scipy.spatial.cKDTree`` and NumPy, one thread
     - 2016.2 ms
     - 0.5 M
     - 1.0
   * - ``scipy.spatial.cKDTree`` and NumPy, all cores
     - 278.4 ms
     - 3.6 M
     - 7.2
   * - pyinterp ``RTree3D``, geodetic, one thread
     - 896.8 ms
     - 1.1 M
     - 2.2
   * - pyinterp ``RTree3D``, geodetic, all cores
     - 104.8 ms
     - 9.5 M
     - 19.2

Given the ECEF coordinates of the points, pyinterp works in Cartesian mode
(``RTree3D()``), exactly as the k-d tree of SciPy:

.. list-table::
   :header-rows: 1
   :widths: 55 15 15 15

   * - Library
     - Time
     - Points per second
     - Speed-up
   * - ``scipy.spatial.cKDTree`` and NumPy, one thread
     - 2084.1 ms
     - 0.5 M
     - 1.0
   * - ``scipy.spatial.cKDTree`` and NumPy, all cores
     - 250.3 ms
     - 4.0 M
     - 8.3
   * - pyinterp ``RTree3D``, Cartesian, one thread
     - 903.1 ms
     - 1.1 M
     - 2.3
   * - pyinterp ``RTree3D``, Cartesian, all cores
     - 96.2 ms
     - 10.4 M
     - 21.7

pyinterp is more than twice as fast as SciPy, on one thread as on all cores,
with the same results to :math:`10^{-15}`; from one run to another, the
timings vary by about 10 %. Its R*-tree is searched depth first, closest
nodes first, and the points are processed along a space-filling curve, so
that consecutive searches visit the same nodes. Working on longitudes and
latitudes costs pyinterp less than 10 % of the time of the Cartesian mode. pyinterp also offers radial basis
functions, kriging and window functions on the same tree.

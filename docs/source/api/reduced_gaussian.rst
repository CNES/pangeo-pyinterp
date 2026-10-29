Reduced Gaussian Grids
======================
Interpolation of the reduced Gaussian grids of the ECMWF analyses and
forecasts (``reduced_gg`` in GRIB, the ``N320``, ``N640``, ``O1280`` grids): a
fixed set of parallels, the Gaussian latitudes, each holding equally spaced
points whose number decreases towards the poles. The grids are interpolated
in 2D (a value per point), in 3D (a column of model levels per point) and in
time, from numeric inputs only: the latitudes of the parallels, their number
of points and the values. Reading the files, GRIB or otherwise, is left to the
caller; :ref:`example_reduced_gaussian` gives a complete recipe with
`ecCodes <https://confluence.ecmwf.int/display/ECC>`_.

Mesh, zone, grid and series
---------------------------

* A :class:`~pyinterp.reduced_gaussian.Mesh` describes the geometry and
  never holds values. Its points are numbered parallel after parallel from
  the north, and on each parallel eastwards from longitude 0, as a global
  GRIB field stores them.
* A :class:`~pyinterp.reduced_gaussian.Zone` is the subset of the points a
  regional grid holds: sorted, disjoint ranges ``[start, stop)`` of the flat
  indices of the mesh. It is computed from the mesh alone, **before any value
  is read**, from a box (:meth:`~pyinterp.reduced_gaussian.Zone.from_box`) or
  from the points to interpolate
  (:meth:`~pyinterp.reduced_gaussian.Zone.from_points`, exactly the corners
  they read). A parallel may hold several ranges: a satellite pass crosses
  most parallels twice, on opposite sides of the globe. Read the values of
  the zone with its :meth:`~pyinterp.reduced_gaussian.Zone.indices` (fancy
  indexing of an array, Zarr or netCDF orthogonal selection), or with its
  :meth:`~pyinterp.reduced_gaussian.Zone.ranges` for readers that seek;
  :func:`~pyinterp.reduced_gaussian.coalesce` merges close ranges for fewer,
  larger reads without changing the zone.
* A :class:`~pyinterp.reduced_gaussian.Grid` holds the fields of one time
  step on the mesh or on a zone: float32 or float64 arrays of shape
  ``(points,)`` (2D) or ``(points, levels)`` (3D), C-contiguous, read without
  copy, and optionally the hybrid coefficients ``a`` and ``b`` of the half
  levels.
* A :class:`~pyinterp.reduced_gaussian.Series` holds grids in time on one
  mesh. Each step may hold its own zone:
  :func:`~pyinterp.reduced_gaussian.time_zones` computes them, each from the
  points whose date reads the step.

A point is located once on the mesh. Its cell, the four points around it on
the two parallels that bracket its latitude, then serves every field, every
level and both time steps it reads. The cell is a trapezoid, its two
parallels holding different numbers of points, and the bilinear weights
account for it. A NaN value is left out and the weights of the other corners
renormalized, level by level, so that a level under the orography of one
corner does not void the whole column.

Time steps
----------

By default a date reads the two steps that bracket it, blended linearly, or
the nearest one (``time_method="nearest"``). ``interval`` reads instead the two
ends of the analysis interval the date falls in, on a lattice whose phase is
the first step (00, 06, 12, 18 h for analyses; 03, 09, 15, 21 h for
forecasts): an interval that lost one of its ends is read off the other alone,
one that lost both is NaN, and
:func:`~pyinterp.reduced_gaussian.map_availability` counts the missing ends,
the flag the altimetry products publish. ``max_step_gap`` treats a wider pair
as a missing step. The dates are ``numpy.datetime64`` (handled at nanosecond
resolution) or integers in a unit of your choice; the durations are then
``numpy.timedelta64`` or integers, respectively.

Computing from the columns
--------------------------

:func:`~pyinterp.reduced_gaussian.interpolate` returns the interpolated fields:
a value per point for a 2D field, a column per point for a 3D field. Anything
computed point by point from them with numpy, such as a dry tropospheric
correction from the surface pressure, needs nothing more.
:func:`~pyinterp.reduced_gaussian.apply` computes from the columns without
holding them all in memory, in two ways:

* a **Python callable**, called on blocks of points with the columns, the
  coordinates of the points and their extra arguments, written with numpy
  operations over the block;
* a **native kernel**, a function compiled by numba with
  :func:`~pyinterp.reduced_gaussian.column_kernel` (``pip install numba``),
  called for each point from several threads without the GIL: for
  computations that loop over the levels and do not vectorize over points,
  such as a vertical integration. A kernel cannot raise; it returns 0, or a
  non-zero code that makes the point NaN.

For a series, ``blend="columns"`` (default) blends the columns of the two steps
before calling the function, ``blend="results"`` calls it on each step and
blends its results, which costs twice as much: it interpolates in time the
result rather than the fields, which differs when the function is nonlinear in
them.

NumPy functions
---------------

.. module:: pyinterp.reduced_gaussian

.. autosummary::
   :toctree: _generated/

   Mesh
   Zone
   Grid
   Series
   interpolate
   apply
   column_kernel
   ColumnKernel
   time_zones
   map_availability
   coalesce
   gaussian_latitudes
   octahedral_points
   regular_points

Low-level functions
-------------------

.. currentmodule:: pyinterp.core.reduced_gaussian

.. autosummary::
   :toctree: _generated/

   interpolate
   apply
   bracket

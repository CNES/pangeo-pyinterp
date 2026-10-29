""".. _example_xarray_interpolator:

Interpolating an xarray DataArray
=================================

:class:`pyinterp.backends.xarray.RegularGridInterpolator` is the one-call entry
point to interpolate a 2-, 3- or 4-D ``xarray.DataArray``: it identifies the
axes, builds the matching grid and dispatches to the bivariate, trivariate or
quadrivariate interpolator. In this example you will learn how to build it,
query it with a dictionary of coordinates (including dates), choose and tune
the interpolation method, and what happens at the dateline and outside the
grid.
"""

import matplotlib.pyplot
import numpy
import xarray

import pyinterp
import pyinterp.backends.xarray
import pyinterp.tests

# %%
# Building the Interpolator
# -------------------------
#
# We load a global grid of mean sea surface (MSS). The variable is stored as
# ``(lat, lon)``, but the interpolator does not care about the order of the
# dimensions: the longitude and latitude are recognized from the ``units``
# attribute of their coordinates (``degrees_east`` and ``degrees_north``, see
# :class:`pyinterp.cf.AxisLongitudeUnit` and
# :class:`pyinterp.cf.AxisLatitudeUnit`), and the data is transposed to
# ``(longitude, latitude)`` internally.
ds = pyinterp.tests.load_grid2d()
interpolator = pyinterp.backends.xarray.RegularGridInterpolator(ds.mss)
print(f"dims={ds.mss.dims}, ndim={interpolator.ndim}")
print(interpolator.grid)

# %%
# Querying with a Dictionary of Coordinates
# -----------------------------------------
#
# The interpolator is called with a dictionary that maps every dimension name
# to the coordinates of the points to evaluate. The values can be scalars,
# lists or arrays of any shape: they are broadcast against each other, and the
# result has their broadcast shape, here the shape of the 2-D target grid. The
# MSS is stored with longitudes in [0, 360), but we query it in [-180, 180).
mx, my = numpy.meshgrid(
    numpy.arange(-180, 180, 1.0) + 1 / 3,
    numpy.arange(-80, 80, 1.0) + 1 / 3,
    indexing="ij",
)
coords = {"lon": mx, "lat": my}
bilinear = interpolator(coords)

# %%
# Choosing the Method
# -------------------
#
# The ``method`` argument selects the interpolator. ``bilinear`` (default),
# ``nearest`` and ``idw`` only use the cell surrounding the point. Windowed
# methods (``bicubic``, ``c_spline``, ``akima``, ``steffen``...) fit a
# function on a window of neighboring nodes; its size is set by
# ``half_window_size_x`` and ``half_window_size_y``, and ``boundary_mode``
# decides whether the window may shrink near the edges of the grid
# (``"shrink"``) or whether the result is undefined there (``"undef"``, the
# default). The other keyword arguments, such as ``num_threads``, are forwarded
# to :func:`pyinterp.bivariate`.
bicubic = interpolator(
    coords,
    method="bicubic",
    half_window_size_x=3,
    half_window_size_y=3,
    boundary_mode="shrink",
)

# %%
# The two methods agree to within a millimeter; the largest differences
# follow the strongest gradients of the field. Note that a window containing a
# single masked (NaN) node gives NaN, so the windowed method loses a few
# points along the coasts.
print(
    f"NaN: bilinear={numpy.isnan(bilinear).sum()}, "
    f"bicubic={numpy.isnan(bicubic).sum()}"
)
fig, (ax1, ax2) = matplotlib.pyplot.subplots(
    2, 1, figsize=(10, 6), sharex=True, layout="constrained"
)
pcm = ax1.pcolormesh(mx, my, bilinear, cmap="viridis", vmin=-0.1, vmax=0.1)
fig.colorbar(pcm, ax=ax1, label="MSS (m)")
ax1.set_title("Bilinear interpolation, longitudes in [-180, 180)")
pcm = ax2.pcolormesh(
    mx, my, (bicubic - bilinear) * 1e3, cmap="RdBu_r", vmin=-0.5, vmax=0.5
)
fig.colorbar(pcm, ax=ax2, label="mm")
ax2.set_title("Bicubic minus bilinear")
ax2.set_xlabel("Longitude (degrees)")
for ax in (ax1, ax2):
    ax.set_ylabel("Latitude (degrees)")

# %%
# Longitude Periodicity
# ---------------------
#
# A longitude axis is always created as periodic (360 degrees). Longitudes are
# therefore never out of bounds: the same location written as -0.1, 359.9 or
# 719.9 degrees gives the same value, and points between the last column
# (359.75) and 360 are interpolated across the seam of the grid.
print(interpolator({"lon": [-0.1, 359.9, 719.9], "lat": -40.1}))

# %%
# Outside the Grid
# ----------------
#
# The latitudes cover [-89, 88.75]. A point beyond this range returns NaN by
# default; set ``bounds_error=True`` to raise a :class:`ValueError` instead.
outside = {"lon": -30.0, "lat": 89.5}
print(interpolator(outside))
try:
    interpolator(outside, bounds_error=True)
except ValueError as exc:
    print(f"ValueError: {exc}")

# %%
# Adding a Time Dimension
# -----------------------
#
# With a 3-D array the interpolator becomes trivariate. Here the total column
# water vapor is stored as ``(time, latitude, longitude)`` with daily values.
# A coordinate of type ``datetime64`` is detected as a temporal axis and moved
# to the third position, so the query dates are simply passed as
# ``datetime64`` values, at any resolution.
tcw = pyinterp.tests.load_grid3d().tcw
interpolator = pyinterp.backends.xarray.RegularGridInterpolator(tcw)
print(interpolator.grid)

dates = numpy.arange(
    numpy.datetime64("2002-07-02T00", "h"),
    numpy.datetime64("2002-07-12T00", "h"),
    numpy.timedelta64(3, "h"),
)
track = {
    "longitude": 10.3,
    "latitude": 45.1,
    "time": dates,
}

# %%
# The default ``bilinear`` method is linear along the time axis. The
# ``third_axis`` option (``"linear"`` or ``"nearest"``) selects the
# interpolation along time; for a 4-D array, ``fourth_axis`` plays the same
# role for the fourth dimension. These options, like ``half_window_size_x``,
# select the windowed version of ``bilinear``.
linear = interpolator(track)
nearest = interpolator(track, method="bilinear", third_axis="nearest")

fig, ax = matplotlib.pyplot.subplots(figsize=(10, 4), layout="constrained")
ax.plot(dates, linear, label="bilinear (linear in time)")
ax.plot(dates, nearest, label="bilinear, third_axis='nearest'")
ax.set_ylabel("Total column water vapor (kg m$^{-2}$)")
ax.set_title("Time series at 10.3E, 45.1N")
ax.legend()

# %%
# A 4-D array works the same way: with ``(time, level, latitude, longitude)``
# dimensions, the axes are reordered to ``(longitude, latitude, time,
# level)``, and the dictionary must name all four dimensions.
pressure = pyinterp.tests.load_grid4d().pressure
interpolator = pyinterp.backends.xarray.RegularGridInterpolator(pressure)
print(interpolator.grid)

# %%
# Labelled Results with the ``pyinterp`` Accessor
# -----------------------------------------------
#
# Once pyinterp and xarray are imported, data arrays and datasets have a
# ``pyinterp`` accessor. Its
# :meth:`~pyinterp.backends.xarray.DataArrayAccessor.interp` method follows
# :meth:`xarray.DataArray.interp`: the coordinates are passed as keyword
# arguments, and the result is a ``DataArray`` labelled with them, which keeps
# the name and the attributes of the variable. One-dimensional coordinates
# define a new grid, so there is no ``meshgrid`` to build.
mss = ds.mss.pyinterp.interp(
    lon=numpy.arange(-180, 180, 0.5),
    lat=numpy.arange(-80, 80, 0.5),
    method="bicubic",
)
print(mss)

# %%
# Coordinates given as data arrays sharing a dimension are interpolated point
# by point, along a track for instance. The keyword arguments that do not
# name a dimension are the options of the interpolator, and dates can be
# given as strings.
track = tcw.pyinterp.interp(
    longitude=xarray.DataArray(numpy.linspace(-30, 30, 50), dims="obs"),
    latitude=xarray.DataArray(numpy.linspace(-20, 40, 50), dims="obs"),
    time="2002-07-05T12:00",
    bounds_error=True,
)
print(track)

# %%
# When to Use It
# --------------
#
# The ``pyinterp`` accessor is the shortest path when the data is already an
# ``xarray.DataArray`` and a labelled result is wanted; it builds the grid at
# each call, so keep a
# :class:`~pyinterp.backends.xarray.RegularGridInterpolator` to interpolate
# the same variable many times. The interpolator is one class for any
# number of dimensions, coordinates passed by name, dates handled for you.
# It is a thin wrapper around
# :class:`~pyinterp.backends.xarray.Grid2D`,
# :class:`~pyinterp.backends.xarray.Grid3D` and
# :class:`~pyinterp.backends.xarray.Grid4D`; use these classes and their
# ``bivariate``, ``trivariate`` or ``quadrivariate`` methods, or build a
# :class:`pyinterp.Grid` and call :func:`pyinterp.bivariate` and friends,
# when you work with plain ``numpy`` arrays, need full control over the axes
# (a non-periodic longitude, for instance), or reuse a configuration object.
# These lower-level interfaces are shown in :ref:`example_2d_interpolation`,
# :ref:`example_3d_interpolation` and :ref:`example_4d_interpolation`.

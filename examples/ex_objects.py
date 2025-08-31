"""
.. _example_objects:

Interpolator Objects
====================

This example explains how to create the fundamental objects required for
interpolation with ``pyinterp``: the axes and the grid. These objects are the
building blocks for defining the space and data on which interpolation will be
performed.

First, let's import the necessary libraries and load a sample dataset.
"""
# %%
import timeit

import numpy
import pandas

import pyinterp
import pyinterp.backends.xarray
import pyinterp.tests

# Load a 3D test grid from the library's test data
ds = pyinterp.tests.load_grid3d()
lon, lat, time, tcw = (
    ds['longitude'].values,
    ds['latitude'].values,
    ds['time'].values,
    ds['tcw'].values,
)

# %%
# Defining Axes
# -------------
# An axis defines the coordinates for one dimension of the grid. ``pyinterp``
# provides specialized axis objects for performance and to handle specific cases
# like circular (e.g., longitude) and temporal axes.
#
# Regular Axis
# ++++++++++++
# A regular axis is a simple, monotonically increasing or decreasing array of
# coordinates. Here, we create an axis for latitude.
y_axis = pyinterp.Axis(lat)
print('Latitude axis:')
print(y_axis)

# %%
# You can use the :py:meth:`pyinterp.Axis.find_index` method to find the nearest
# grid index for a given coordinate.
print(f'Index for latitude 0.12°: {y_axis.find_index([0.12])}')

# %%
# Circular Axis (Longitude)
# +++++++++++++++++++++++++
# For axes that wrap around, like longitude, you can create a "circular" axis.
# This ensures that coordinates are correctly handled at the boundary (e.g.,
# -180° and 180° are treated as the same point).
x_axis = pyinterp.Axis(lon, is_circle=True)
print('\nLongitude axis (circular):')
print(x_axis)

# %%
# With a circular axis, boundary points are correctly identified as identical.
print('Are indices for -180° and 180° the same? '
      f'{x_axis.find_index([-180]) == x_axis.find_index([180])}')

# %%
# Temporal Axis
# +++++++++++++
# For time coordinates, ``pyinterp`` provides a highly optimized
# :py:class:`pyinterp.TemporalAxis` class.
t_axis = pyinterp.TemporalAxis(time)
print('Time axis:')
print(t_axis)

# %%
# Performance Comparison
# ----------------------
# ``pyinterp`` axis objects are implemented in C++ and are significantly faster
# for lookups than equivalent objects in libraries like ``pandas``. Let's
# compare the performance of :py:class:`pyinterp.Axis` against
# ``pandas.Index`` for finding indices.
#
# **Longitude Axis:**
values = lon[10:20] + 1 / 3
index = pandas.Index(lon)
print('Performance for Longitude Axis:')
print(f'  pandas.Index:  '
      f"{timeit.timeit('index.searchsorted(values)', globals=globals()):.6f}s")
print(f'  pyinterp.Axis: '
      f"{timeit.timeit('x_axis.find_index(values)', globals=globals()):.6f}s")

# %%
# **Time Axis:**
index = pandas.Index(time)
values = time + numpy.timedelta64(1, 'ns')
print('Performance for Time Axis:')
print(f'  pandas.Index:        '
      f"{timeit.timeit('index.searchsorted(values)', globals=globals()):.6f}s")
print(f'  pyinterp.TemporalAxis: '
      f"{timeit.timeit('t_axis.find_index(values)', globals=globals()):.6f}s")

# %%
# Creating a Grid
# ---------------
# Once the axes are defined, you can create a grid object that holds the data.
# The grid object takes the axes and the data array as input. The data must be
# organized to match the order of the axes: (time, latitude, longitude).
#
# Here, we create a :py:class:`pyinterp.Grid3D` object for the total column
# water vapor (TCW) data.
# Before constructing the tensor for pyinterp, we must begin to organize the
# data in a grid with the values of the axes in the first dimensions of the
# tensor.
#
# In our case, the time, latitude, longitude axes must be sorted in this order.
tcw = tcw.T
grid = pyinterp.Grid3D(x_axis, y_axis, t_axis, tcw)
print('Grid object:')
print(grid)

# %%
# Using the XArray Backend
# ------------------------
# For convenience, ``pyinterp`` provides a backend that can directly create a
# grid interpolator from an ``xarray.DataArray``. This avoids the need to

# manually create the axes and grid objects.
#
# The backend automatically detects the axis types (regular, circular, temporal)
# and creates the appropriate grid interpolator.
interpolator = pyinterp.backends.xarray.Grid3D(ds['tcw'])
print('Interpolator from xarray.DataArray:')
print(interpolator)

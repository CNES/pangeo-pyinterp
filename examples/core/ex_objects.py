""".. _example_objects:

Interpolator Objects
====================

This example explains how to create the fundamental objects required for
interpolation with ``pyinterp``: the axes and the grid. These objects are the
building blocks for defining the space and data on which interpolation will be
performed.

First, let's import the necessary libraries and load a sample dataset.
"""

# %%
import numpy as np

import pyinterp
import pyinterp.backends.xarray
import pyinterp.tests

# Load a 3D test grid from the library's test data
ds = pyinterp.tests.load_grid3d()
lon, lat, time, tcw = (
    ds["longitude"].values,
    ds["latitude"].values,
    ds["time"].values,
    ds["tcw"].values,
)

# %%
# Defining Axes
# -------------
# An axis defines the coordinates for one dimension of the grid. ``pyinterp``
# provides specialized axis objects for performance and to handle specific
# cases like circular (e.g., longitude) and temporal axes.
#
# Regular Axis
# ++++++++++++
# A regular axis is a simple, monotonically increasing or decreasing array of
# coordinates. Here, we create an axis for latitude.
y_axis = pyinterp.Axis(lat)
print("Latitude axis:")
print(y_axis)

# %%
# You can use the :py:meth:`pyinterp.Axis.find_index` method to find the
# nearest grid index for a given coordinate.
print(f"Index for latitude 0.12°: {y_axis.find_index(np.array([0.12]))}")

# %%
# Periodic Axis (Longitude)
# +++++++++++++++++++++++++
# For axes that wrap around, like longitude, you can create a "periodic" axis.
# This ensures that coordinates are correctly handled at the boundary (e.g.,
# -180° and 180° are treated as the same point).
x_axis = pyinterp.Axis(lon, period=360.0)
print("Longitude axis (periodic):")
print(x_axis)

# %%
# With a periodic axis, boundary points are correctly identified as identical.
idx_1 = x_axis.find_index(np.array([-180]))
idx_2 = x_axis.find_index(np.array([180]))
print(f"Are indices for -180° and 180° the same? {idx_1 == idx_2}")

# %%
# Temporal Axis
# +++++++++++++
# For time coordinates, ``pyinterp`` provides a highly optimized
# :py:class:`pyinterp.TemporalAxis` class.
t_axis = pyinterp.TemporalAxis(time)
print("Time axis:")
print(t_axis)

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
grid = pyinterp.Grid(x_axis, y_axis, t_axis, tcw)
print("Grid object:")
print(grid)

# %%
# Using the XArray Backend
# ------------------------
# For convenience, ``pyinterp`` provides a backend that can directly create a
# grid interpolator from an ``xarray.DataArray``. This avoids the need to
# manually create the axes and grid objects.
#
# The backend automatically detects the axis types (regular, circular,
# temporal) and creates the appropriate grid interpolator.
interpolator = pyinterp.backends.xarray.Grid3D(ds["tcw"])
print("Interpolator from xarray.DataArray:")
print(interpolator)

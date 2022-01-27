"""
***************************
Create interpolator objects
***************************

In this example, we are going to build the basic objects allowing to carry out
interpolations.

Before starting, we will examine the properties of a Cartesian grid and the
different classes associated with these objects.

The first step is to open the NetCDF file and load the data. We use here the
NetCDF4 library to detail the different steps, but we will see that we can
automate the steps described below using the xarray objects library.

Step-by-step creation of grids
##############################
"""
import timeit

import netCDF4
import numpy
import pandas
import xarray

import pyinterp
import pyinterp.backends.xarray
import pyinterp.tests

with netCDF4.Dataset(pyinterp.tests.grid3d_path()) as ds:
    lon, lat, time, time_units, tcw = ds.variables[
        "longitude"][:], ds.variables["latitude"][:], ds.variables[
            "time"][:], ds.variables["time"].units, ds.variables["tcw"][:]
    time = numpy.array(netCDF4.num2date(time, time_units),
                       dtype="datetime64[us]")

# %%
# This regular 3-dimensional grid is associated with three axes:
#
# * longitudes,
# * latitudes and
# * time.
#
# To perform the calculations quickly, we will build three objects that will be
# used by the interpolator to search for the data to be used. Let's start with
# the y-axis representing the latitude axis.
y_axis = pyinterp.Axis(lat)
y_axis

# %%
# For example, you can search for the closest point to 0.12 degrees north
# latitude.
y_axis.find_index([0.12])

# %%
# Then, the x-axis representing the longitudinal axis. In this case, the axis is
# an axis representing a 360 degree circle.
x_axis = pyinterp.Axis(lon, is_circle=True)
x_axis

# %%
# The values -180 and 180 degrees represent the same point on the axis.
x_axis.find_index([-180]) == x_axis.find_index([180])

# %%
# Finally, we create the time axis
t_axis = pyinterp.TemporalAxis(time)
t_axis

# %%
# As these objects must communicate in C++ memory space, we use objects specific
# to the library much faster than other data models and manage the axes
# representing a circle. For example if we compare these objects to Pandas
# indexes:
values = lon[10:20] + 1 / 3
index = pandas.Index(lon)
print("pandas.Index: %f" % timeit.timeit(
    "index.searchsorted(values)", globals=dict(index=index, values=values)))
print("pyinterp.Axis %f" % timeit.timeit(
    "x_axis.find_index(values)", globals=dict(x_axis=x_axis, values=values)))

# %%
# This time axis is also very efficient compared to the pandas index.
index = pandas.Index(time)
values = time + numpy.timedelta64(1, "ns")
print("pandas.Index: %f" % timeit.timeit(
    "index.searchsorted(values)", globals=dict(index=index, values=values)))
print("pyinterp.Axis %f" % timeit.timeit(
    "t_axis.find_index(values)", globals=dict(t_axis=t_axis, values=values)))

# %%
# Before constructing the tensor for pyinterp, we must begin to organize the
# tensor data so that it is properly stored in memory for pyinterp.

# %%
#   * The shape of the tensor must be (len(x_axis), len(y_axis), len(t_axis))
tcw = tcw.T
# %%
#   * The undefined values must be set to nan.
tcw[tcw.mask] = float("nan")

# %%
# Now we can build the object handling the regular 3-dimensional grid.
#
# .. note::
#   Grid data are not copied, the Grid3D class just keeps a reference on the
#   handled array. Axis data are copied for non-uniform axes, and only examined
#   for regular axes.
grid_3d = pyinterp.Grid3D(x_axis, y_axis, t_axis, tcw)
grid_3d

# %%
# xarray backend
# ##############
#
# The construction of these objects manipulating the :py:class:`regular grids
# <pyinterp.backends.xarray.RegularGridInterpolator>` can be done more easily
# using the `xarray <http://xarray.pydata.org/>`_ library and `CF
# <https://cfconventions.org/>`_ convention usually found in NetCDF files.
interpolator = pyinterp.backends.xarray.RegularGridInterpolator(
    xarray.open_dataset(pyinterp.tests.grid3d_path()).tcw)
interpolator.grid

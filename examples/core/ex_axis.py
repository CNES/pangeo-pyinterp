""".. _example_axis:

Manipulating Axes
=================

In pyinterp, axes are fundamental components for defining grids and performing
interpolations. This example demonstrates how to create and manipulate both
regular and irregular axes.

An axis in pyinterp is similar to how axes are defined in NetCDF files,
with properties like ``long_name``, ``units``, and ``standard_name``.

.. code::

    float lat(lat);
        lat:long_name = "latitude";
        lat:units = "degrees_north";
        lat:standard_name = "latitude";
    float lon(lon);
        lon:long_name = "longitude";
        lon:units = "degrees_east";
        lon:standard_name = "longitude";
"""
# %%
# Regular Axis
# ------------
#
# A regular axis has a constant step between its values. Let's create a regular
# axis representing latitudes from -90 to 90 degrees with a step of 0.25.
import numpy

import pyinterp

axis = pyinterp.Axis(numpy.arange(-90, 90, 0.25, dtype=numpy.float64))
print(axis)

# %%
# You can query the properties of the axis, such as whether it is ascending,
# regular, or represents a circle.
print(f'Is ascending? {axis.is_ascending()}')
print(f'Is regular? {axis.is_regular()}')
print(f'Is a circle? {axis.is_circle}')

# %%
# A common operation is to find the index of the value closest to a given
# coordinate.
index = axis.find_index(numpy.array([1e-3]))
print(f'Index of the closest value to 1e-3: {index}')

# %%
# You can also find the indices of the values surrounding a given coordinate.
indices = axis.find_indexes(numpy.array([1e-3]))
print(f'Indices surrounding 1e-3: {indices}')

# %%
# For more details on the available methods, refer to the
# :py:class:`pyinterp.Axis` documentation.

# %%
# Irregular Axis
# --------------
#
# An irregular axis has a non-constant step between its values. The search for
# indices is performed using a binary search, which is slightly slower than the
# simple calculation used for regular axes.
#
# Let's create an irregular axis using Mercator latitudes.
MERCATOR_LATITUDES = numpy.array(
    [
        -89.000000, -88.908818, -88.809323, -88.700757, -88.582294, -88.453032,
        -88.311987, -88.158087, -87.990161, -87.806932, -87.607008, -87.388869,
        -87.150861, -86.891178, -86.607851, -86.298736, -85.961495, -85.593582,
        -85.192224, -84.754402, -84.276831, -83.755939, -83.187844, -82.568330,
        -81.892820, -81.156357, -80.353575, -79.478674, -78.525397, -77.487013,
        -76.356296, -75.125518, -73.786444, -72.330344, -70.748017, -69.029837,
        -67.165823, -65.145744, -62.959262, -60.596124, -58.046413, -55.300856,
        -52.351206, -49.190700, -45.814573, -42.220632, -38.409866, -34.387043,
        -30.161252, -25.746331, -21.161107, -16.429384, -11.579629, -6.644331,
        -1.659041, 3.338836, 8.311423, 13.221792, 18.035297, 22.720709,
        27.251074, 31.604243, 35.763079, 39.715378, 43.453560, 46.974192,
        50.277423, 53.366377, 56.246554, 58.925270, 61.411164, 63.713764,
        65.843134, 67.809578, 69.623418, 71.294813, 72.833637, 74.249378,
        75.551083, 76.747318, 77.846146, 78.855128, 79.781321, 80.631294,
        81.411149, 82.126535, 82.782681, 83.384411, 83.936179, 84.442084,
        84.905904, 85.331111, 85.720897, 86.078198, 86.405707, 86.705898,
        86.981044, 87.233227, 87.464359, 87.676195, 87.870342, 88.048275,
        88.211348, 88.360799, 88.497766, 88.623291, 88.738328, 88.843755,
        88.940374
    ],
    dtype=numpy.float64,
)

axis = pyinterp.Axis(MERCATOR_LATITUDES)
print(axis)

# %%
# Let's check its properties. As expected, the axis is no longer regular.
print(f'Is ascending? {axis.is_ascending()}')
print(f'Is regular? {axis.is_regular()}')
print(f'Is a circle? {axis.is_circle}')

# %%
# You can query this axis in the same way as a regular axis.
index = axis.find_index(numpy.array([1e-3]))
print(f'Index of the closest value to 1e-3: {index}')

# %%
# Longitude Axis
# --------------
#
# Longitude axes often represent a full circle around the Earth. You can create
# such an axis by setting the ``is_circle`` parameter to ``True``.
axis = pyinterp.Axis(numpy.arange(0, 360, 1, dtype=numpy.float64),
                     is_circle=True)
print(axis)

# %%
# Now, the ``is_circle`` property is ``True``.
print(f'Is a circle? {axis.is_circle}')

# %%
# When finding indices on a circular axis, the axis wraps around. For example,
# an angle of 361 degrees is equivalent to 1 degree.
index = axis.find_index(numpy.array([361.0]))
print(f'Index of 361 degrees: {index}')

index = axis.find_index(numpy.array([1.0]))
print(f'Index of 1 degree: {index}')

# %%
# TemporalAxis
# ------------
#
# Time axes allow for manipulating axes representing dates or time differences.
# These objects are specialized to handle the 64-bit integers used by numpy to
# describe dates without losing information during calculations. In a netCDF
# file these axes are described as follows:
#
# .. code::
#
#     double time(time) ;
#         time:long_name = "time" ;
#         time:units = "days since 1990-1-1 0:0:0" ;
#
# .. note::
#
#   These axes can be regular or irregular as before.
dates = numpy.datetime64('2020-01-01') + numpy.arange(
    0, 10**6, step=500).astype('timedelta64[ms]')
axis = pyinterp.TemporalAxis(dates)
print(axis)

# %%
# It is possible to search for a date in this axis.
axis.find_index(numpy.array([numpy.datetime64('2020-01-01T00:10:34.000')]))

# %%
# You can pass any date unit to the axis.
axis.find_index(numpy.array([numpy.datetime64('2020-01-01')]))

# %%
# This object also makes it possible to manipulate timedeltas.
axis = pyinterp.TemporalAxis(dates - numpy.datetime64('2020-01-01'))
print(axis)

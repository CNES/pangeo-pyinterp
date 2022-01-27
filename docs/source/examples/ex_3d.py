"""
****************
3D interpolation
****************

Interpolation of a three-dimensional regular grid.

Trivariate
==========

The :py:func:`trivariate <pyinterp.trivariate>` interpolation allows obtaining
values at arbitrary points in a 3D space of a function defined on a grid.

The distribution contains a 3D field ``tcw.nc`` that will be used in this help.
This file is located in the ``src/pyinterp/tests/dataset`` directory at the root
of the project.

This method performs a bilinear interpolation in 2D space by considering the
axes of longitude and latitude of the grid, then performs a linear
interpolation in the third dimension. Its interface is similar to the
:py:func:`bivariate <pyinterp.bivariate>` class except for a third axis, which
is handled by this object.

.. note::

    When using a time axis, care must be taken to use the same unit of dates,
    between the axis defined and the dates supplied during interpolation. The
    function :py:meth:`pyinterp.TemporalAxis.safe_cast` automates this task and
    will warn you if there is an inconsistency during the date conversion.
"""
import cartopy.crs
import matplotlib
import matplotlib.pyplot
import numpy
import xarray

import pyinterp
import pyinterp.backends.xarray
import pyinterp.tests

# %%
# The first step is to load the data into memory and create the interpolator
# object:
ds = xarray.open_dataset(pyinterp.tests.grid3d_path())
interpolator = pyinterp.backends.xarray.Grid3D(ds.tcw)

# %%
# We will build a new grid that will be used to build a new interpolated grid.
#
# .. note ::
#
#   The coordinates used for interpolation are shifted to avoid using the
#   points of the trivariate function.
#
# .. warning ::
#
#   When using a time axis, care must be taken to use the same unit of dates,
#   between the axis defined and the dates supplied during interpolation. The
#   function :py:meth:`pyinterp.TemporalAxis.safe_cast` automates this task and
#   will warn you if there is an inconsistency during the date conversion.
mx, my, mz = numpy.meshgrid(numpy.arange(-180, 180, 0.25) + 1 / 3.0,
                            numpy.arange(-80, 80, 0.25) + 1 / 3.0,
                            numpy.array(["2002-07-02T15:00:00"],
                                        dtype="datetime64"),
                            indexing='ij')

# %%
# We interpolate our grid using a :py:meth:`classical
# <pyinterp.backends.xarray.Grid3D.trivariate>`:
trivariate = interpolator.trivariate(
    dict(longitude=mx.ravel(), latitude=my.ravel(), time=mz.ravel()))

# %%
# Bicubic on 3D grid
# ==================
#
# Used grid organizes the latitudes in descending order. We ask our
# constructor to flip this axis in order to correctly evaluate the bicubic
# interpolation from this 3D cube (only necessary to perform a bicubic
# interpolation).
interpolator = pyinterp.backends.xarray.Grid3D(ds.data_vars["tcw"],
                                               increasing_axes=True)

# %%
# We interpolate our grid using a :py:meth:`bicubic
# <pyinterp.backends.xarray.Grid3D.bicubic>` interpolation in space followed by
# a linear interpolation in the temporal axis:
bicubic = interpolator.bicubic(
    dict(longitude=mx.ravel(), latitude=my.ravel(), time=mz.ravel()))

# %%
# We transform our result cubes into a matrix.
trivariate = trivariate.reshape(mx.shape).squeeze(axis=2)
bicubic = bicubic.reshape(mx.shape).squeeze(axis=2)
lons = mx[:, 0].squeeze()
lats = my[0, :].squeeze()

# %%
# Let's visualize our results.
fig = matplotlib.pyplot.figure(figsize=(5, 8))
ax1 = fig.add_subplot(
    211, projection=cartopy.crs.PlateCarree(central_longitude=180))
pcm = ax1.pcolormesh(lons,
                     lats,
                     trivariate.T,
                     cmap='jet',
                     shading='auto',
                     transform=cartopy.crs.PlateCarree(),
                     vmin=0,
                     vmax=80)
ax1.coastlines()
ax1.set_extent([80, 170, -45, 30], crs=cartopy.crs.PlateCarree())
ax1.set_title("Trilinear")

ax2 = fig.add_subplot(
    212, projection=cartopy.crs.PlateCarree(central_longitude=180))
pcm = ax2.pcolormesh(lons,
                     lats,
                     bicubic.T,
                     cmap='jet',
                     shading='auto',
                     transform=cartopy.crs.PlateCarree(),
                     vmin=0,
                     vmax=80)
ax2.coastlines()
ax2.set_extent([80, 170, -45, 30], crs=cartopy.crs.PlateCarree())
ax2.set_title("Spline & Linear in time")
fig.colorbar(pcm, ax=[ax1, ax2], shrink=0.8)
fig.show()

"""
.. _example_3d_interpolation:

3D Interpolation
================

This example demonstrates how to perform 3D interpolation on a regular grid. The
pyinterp library supports both trivariate and bicubic interpolation for 3D data.
"""
import cartopy.crs
import matplotlib
import matplotlib.pyplot
import numpy

import pyinterp
import pyinterp.backends.xarray
import pyinterp.tests

# %%
# Trivariate Interpolation
# ------------------------
#
# Trivariate interpolation extends bivariate interpolation to three dimensions.
# It performs a bilinear interpolation on the 2D spatial plane (longitude and
# latitude) and then a linear interpolation on the third dimension (in this
# case, time).
#
# First, we load the 3D dataset and create the interpolator object.
ds = pyinterp.tests.load_grid3d()
interpolator = pyinterp.backends.xarray.Grid3D(ds.tcw)

# %%
# Next, we define the coordinates for interpolation. To avoid interpolating at
# the exact grid points, we introduce a slight shift.
#
# .. note::
#
#   When working with a time axis, ensure that the date units are consistent
#   between the grid and the interpolation coordinates. The
#   :py:meth:`pyinterp.TemporalAxis.safe_cast` method can help manage date
#   conversions and prevent inconsistencies.
mx, my, mz = numpy.meshgrid(numpy.arange(-180, 180, 0.25) + 1 / 3.0,
                            numpy.arange(-80, 80, 0.25) + 1 / 3.0,
                            numpy.array(['2002-07-02T15:00:00'],
                                        dtype='datetime64'),
                            indexing='ij')

# %%
# Now, we perform the trivariate interpolation.
trivariate = interpolator.trivariate({
    'longitude': mx.ravel(),
    'latitude': my.ravel(),
    'time': mz.ravel()
})

# %%
# Bicubic Interpolation on a 3D Grid
# ----------------------------------
#
# For smoother results, you can use bicubic interpolation for the spatial
# dimensions, followed by a linear interpolation on the third dimension.
#
# .. note::
#
#   Bicubic interpolation requires that the grid axes are strictly increasing.
#   If your latitudes are in descending order, you can set the `increasing_axes`
#   parameter to ``True`` to automatically flip them.
interpolator = pyinterp.backends.xarray.Grid3D(ds.data_vars['tcw'],
                                               increasing_axes=True)

# %%
# We then perform the bicubic interpolation.
bicubic = interpolator.bicubic({
    'longitude': mx.ravel(),
    'latitude': my.ravel(),
    'time': mz.ravel()
})

# %%
# To visualize the results, we reshape the output arrays and extract the
# longitude and latitude coordinates.
trivariate = trivariate.reshape(mx.shape).squeeze(axis=2)
bicubic = bicubic.reshape(mx.shape).squeeze(axis=2)
lons = mx[:, 0].squeeze()
lats = my[0, :].squeeze()

# %%
# Finally, let's plot the results of both trivariate and bicubic interpolation.
fig = matplotlib.pyplot.figure(figsize=(10, 8))
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.25)
ax1 = fig.add_subplot(
    211, projection=cartopy.crs.PlateCarree(central_longitude=180))
ax1.set_extent([-180, 180, -90, 90], crs=cartopy.crs.PlateCarree())
pcm = ax1.pcolormesh(lons,
                     lats,
                     trivariate.T,
                     cmap='jet',
                     shading='auto',
                     transform=cartopy.crs.PlateCarree())
ax1.coastlines()
ax1.set_title('Trivariate Interpolation')
ax2 = fig.add_subplot(
    212, projection=cartopy.crs.PlateCarree(central_longitude=180))
ax2.set_extent([-180, 180, -90, 90], crs=cartopy.crs.PlateCarree())
pcm = ax2.pcolormesh(lons,
                     lats,
                     bicubic.T,
                     cmap='jet',
                     shading='auto',
                     transform=cartopy.crs.PlateCarree())
ax2.coastlines()
ax2.set_title('Bicubic Interpolation')
fig.colorbar(pcm, ax=[ax1, ax2], shrink=0.8)

# %%
# Same plot as above, but zoomed into a specific region to better highlight the
# differences between the two interpolation methods.
fig = matplotlib.pyplot.figure(figsize=(5, 8))
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.25)
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
ax1.set_title('Trilinear')

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
ax2.set_title('Spline & Linear in time')
fig.colorbar(pcm, ax=[ax1, ax2], shrink=0.8)

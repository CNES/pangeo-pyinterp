"""
.. _example_4d_interpolation:

4D Interpolation
================

This example demonstrates how to perform 4D interpolation on a regular grid. The
pyinterp library supports both quadrivariate and bicubic interpolation for 4D
data.
"""
import cartopy.crs
import matplotlib
import matplotlib.pyplot
import numpy

import pyinterp
import pyinterp.backends.xarray
import pyinterp.tests

# %%
# Quadrivariate Interpolation
# ---------------------------
#
# Quadrivariate interpolation extends trivariate interpolation to four
# dimensions. It performs bilinear interpolation on the 2D spatial plane
# (longitude and latitude) and then linear interpolation on the third and fourth
# dimensions.
#
# First, we load the 4D dataset and create the interpolator object.
ds = pyinterp.tests.load_grid4d()
interpolator = pyinterp.backends.xarray.Grid4D(ds.pressure)

# %%
# Next, we define the coordinates for interpolation.
#
# .. warning::
#
#   When using a time axis, ensure that the date units are consistent
#   between the grid and the interpolation coordinates. The
#   :py:meth:`pyinterp.TemporalAxis.safe_cast` method can help manage date
#   conversions and prevent inconsistencies.
mx, my, mz, mu = numpy.meshgrid(numpy.arange(-125, -70, 0.5),
                                numpy.arange(25, 50, 0.5),
                                numpy.datetime64('2000-01-01T12:00'),
                                0.5,
                                indexing='ij')

# %%
# Now, we perform the quadrivariate interpolation.
quadrivariate = interpolator.quadrivariate({
    'longitude': mx.ravel(),
    'latitude': my.ravel(),
    'time': mz.ravel(),
    'level': mu.ravel()
}).reshape(mx.shape)

# %%
# Bicubic Interpolation on a 4D Grid
# ----------------------------------
#
# For smoother results, you can use bicubic interpolation for the spatial
# dimensions, followed by linear interpolation on the other dimensions.
#
# .. note::
#
#   Bicubic interpolation requires that the grid axes are strictly increasing.
#   If your latitudes are in descending order, you can set the `increasing_axes`
#   parameter to ``True`` to automatically flip them.
interpolator = pyinterp.backends.xarray.Grid4D(ds.pressure,
                                               increasing_axes=True)

# %%
# We then perform the bicubic interpolation.
bicubic = interpolator.bicubic(
    {
        'longitude': mx.ravel(),
        'latitude': my.ravel(),
        'time': mz.ravel(),
        'level': mu.ravel()
    },
    nx=2,
    ny=2).reshape(mx.shape)

# %%
# To visualize the results, we reshape the output arrays and extract the
# longitude and latitude coordinates.
quadrivariate = quadrivariate.squeeze(axis=(2, 3))
bicubic = bicubic.squeeze(axis=(2, 3))
lons = mx[:, 0].squeeze()
lats = my[0, :].squeeze()

# %%
# Finally, let's plot the results of both quadrivariate and bicubic
# interpolation.
#
# .. note::
#
#   The resolution of the example grid is low (one pixel per degree), so the
#   bicubic interpolation may not find enough pixels at the edges, resulting
#   in undefined values.
fig = matplotlib.pyplot.figure(figsize=(10, 8))
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.25)
ax1 = fig.add_subplot(
    211, projection=cartopy.crs.PlateCarree(central_longitude=180))
ax1.set_extent([lons.min(), lons.max(),
                lats.min(), lats.max()],
               crs=cartopy.crs.PlateCarree())
pcm = ax1.pcolormesh(lons,
                     lats,
                     quadrivariate.T,
                     cmap='jet',
                     shading='auto',
                     transform=cartopy.crs.PlateCarree())
ax1.coastlines()
ax1.set_title('Quadrivariate Interpolation')
ax2 = fig.add_subplot(
    212, projection=cartopy.crs.PlateCarree(central_longitude=180))
ax2.set_extent([lons.min(), lons.max(),
                lats.min(), lats.max()],
               crs=cartopy.crs.PlateCarree())
pcm = ax2.pcolormesh(lons,
                     lats,
                     bicubic.T,
                     cmap='jet',
                     shading='auto',
                     transform=cartopy.crs.PlateCarree())
ax2.coastlines()
ax2.set_title('Bicubic Interpolation')
fig.colorbar(pcm, ax=[ax1, ax2], shrink=0.8)

"""
.. _example_fill_undef:

Filling Undefined Values
========================

When working with gridded data, undefined values (NaNs) can be problematic for
interpolation, especially near land/sea masks. If any of the grid points used
for interpolation are undefined, the result will also be undefined. This example
demonstrates how to fill these undefined values in a grid to allow for seamless
interpolation.
"""
# %%
# The Problem with Undefined Values
# ---------------------------------
#
# To illustrate the issue, consider the following grid where some values are
# undefined (represented by red points). If we want to interpolate the value at
# the gray point using bilinear interpolation, the calculation will fail because
# one of the surrounding grid points is undefined. However, the green point can
# be interpolated without any issues because all its surrounding points are
# defined.
import cartopy.crs
import cartopy.feature
import matplotlib.pyplot
import numpy

import pyinterp.backends.xarray
import pyinterp.fill
import pyinterp.tests

fig = matplotlib.pyplot.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection=cartopy.crs.PlateCarree())
ax.set_extent([-6, 1, 47.5, 51.5], crs=cartopy.crs.PlateCarree())
ax.add_feature(cartopy.feature.LAND.with_scale('110m'))
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

lons, lats = numpy.meshgrid(numpy.arange(-6, 2),
                            numpy.arange(47.5, 52.5),
                            indexing='ij')
mask = numpy.array([
    [1, 1, 1, 0, 0, 0, 0, 0],  # yapf: disable
    [1, 1, 0, 0, 0, 0, 0, 0],  # yapf: disable
    [1, 1, 1, 1, 1, 1, 0, 0],  # yapf: disable
    [1, 0, 0, 1, 1, 1, 1, 1],  # yapf: disable
    [1, 1, 1, 0, 0, 0, 0, 0]
]).T
ax.scatter(lons.ravel(),
           lats.ravel(),
           c=mask,
           cmap='bwr_r',
           transform=cartopy.crs.PlateCarree(),
           vmin=0,
           vmax=1)
ax.plot([-3.5], [49], linestyle='', marker='.', color='dimgray', markersize=15)
ax.plot([-2.5], [50], linestyle='', marker='.', color='green', markersize=15)
fig.show()

# %%
# .. note::
#
#   This issue does not affect nearest-neighbor interpolation, as it does not
#   perform any arithmetic operations on the grid values.
#
# Filling with LOESS (Local Regression)
# -------------------------------------
#
# The :py:func:`pyinterp.fill.loess` function provides a way to fill undefined
# values using weighted local regression. This method extrapolates values at the
# boundary between defined and undefined regions. You need to specify the number
# of pixels to consider along the X and Y axes.
#
# Let's start by loading a sample grid.
ds = pyinterp.tests.load_grid2d()
grid = pyinterp.backends.xarray.Grid2D(ds.mss)

# %%
# Now, we can fill the undefined values using the LOESS method.
filled = pyinterp.fill.loess(grid, nx=3, ny=3)

# %%
# Let's visualize the original and filled grids to see the result.
fig = matplotlib.pyplot.figure(figsize=(10, 8))
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.25)
ax1 = fig.add_subplot(
    211, projection=cartopy.crs.PlateCarree(central_longitude=180))
lons, lats = numpy.meshgrid(grid.x, grid.y, indexing='ij')
pcm = ax1.pcolormesh(lons,
                     lats,
                     ds.mss.T,
                     cmap='jet',
                     shading='auto',
                     transform=cartopy.crs.PlateCarree(),
                     vmin=-0.1,
                     vmax=0.1)
ax1.coastlines()
ax1.set_title('Original MSS')
ax1.set_extent([0, 170, -45, 30], crs=cartopy.crs.PlateCarree())
ax2 = fig.add_subplot(
    212, projection=cartopy.crs.PlateCarree(central_longitude=180))
pcm = ax2.pcolormesh(lons,
                     lats,
                     filled,
                     cmap='jet',
                     shading='auto',
                     transform=cartopy.crs.PlateCarree(),
                     vmin=-0.1,
                     vmax=0.1)
ax2.coastlines()
ax2.set_title('Filled MSS with LOESS')
ax2.set_extent([0, 170, -45, 30], crs=cartopy.crs.PlateCarree())
fig.colorbar(pcm, ax=[ax1, ax2], shrink=0.8)

# %%
# Filling with Gauss-Seidel Relaxation
# ------------------------------------
#
# Another method for filling undefined values is the Gauss-Seidel relaxation
# technique, available through the :py:func:`pyinterp.fill.gauss_seidel`
# function. This iterative method is generally faster than LOESS.
#
# The function returns a tuple containing the filled grid and a convergence
# flag.
converged, filled = pyinterp.fill.gauss_seidel(grid)

# %%
# Let's visualize the result of the Gauss-Seidel relaxation.
fig = matplotlib.pyplot.figure(figsize=(10, 4))
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
ax1 = fig.add_subplot(
    111, projection=cartopy.crs.PlateCarree(central_longitude=180))
pcm = ax1.pcolormesh(lons,
                     lats,
                     filled,
                     cmap='jet',
                     shading='auto',
                     transform=cartopy.crs.PlateCarree(),
                     vmin=-0.1,
                     vmax=0.1)
ax1.coastlines()
ax1.set_title('Filled MSS with Gauss-Seidel')
ax1.set_extent([0, 170, -45, 30], crs=cartopy.crs.PlateCarree())
fig.colorbar(pcm, ax=ax1, shrink=0.8)

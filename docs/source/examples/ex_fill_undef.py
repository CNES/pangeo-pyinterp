"""
***************
Fill NaN values
***************

The undefined values in the grids do not allow interpolation of values located
in the neighborhood. This behavior is a concern when you need to interpolate
values near the land/sea mask of some maps. The library provides two functions
to fill the undefined values.

LOESS
=====

The :py:func:`first <pyinterp.fill.loess>` method applies a weighted local
regression to extrapolate the boundary between defined and undefined values. The
user must indicate the number of pixels on the X and Y axes to be considered in
the calculation. For example:
"""
import os
import pathlib
import cartopy.crs
import matplotlib.pyplot
import numpy
import pyinterp.backends.xarray
# Module that handles the filling of undefined values.
import pyinterp.fill
import xarray

DATASET = pathlib.Path(os.environ['DATASET'])
MSS = DATASET.joinpath("mss.nc")

#%%
# Load the data into memory
ds = xarray.open_dataset(MSS)
grid = pyinterp.backends.xarray.Grid2D(ds.mss)

#%%
# The function filling the holes near the mask is called
filled = pyinterp.fill.loess(grid, nx=3, ny=3)

#%%
# The image below illustrates the result:
fig = matplotlib.pyplot.figure(figsize=(10, 10))
fig.patch.set_alpha(0.0)
ax1 = fig.add_subplot(
    211, projection=cartopy.crs.PlateCarree(central_longitude=180))
lons, lats = numpy.meshgrid(grid.x, grid.y, indexing='ij')
pcm = ax1.pcolormesh(lons,
                     lats,
                     ds.mss.T,
                     cmap='jet',
                     transform=cartopy.crs.PlateCarree(),
                     vmin=-0.1,
                     vmax=0.1)
ax1.coastlines()
ax1.set_title("Original MSS")
ax1.set_extent([40, 170, -45, 30], crs=cartopy.crs.PlateCarree())
ax2 = fig.add_subplot(212, projection=cartopy.crs.PlateCarree(central_longitude=180))
pcm = ax2.pcolormesh(lons,
                     lats,
                     filled,
                     cmap='jet',
                     transform=cartopy.crs.PlateCarree(),
                     vmin=-0.1,
                     vmax=0.1)
ax2.coastlines()
ax2.set_title("MSS modified using the LOESS filter")
ax2.set_extent([40, 170, -45, 30], crs=cartopy.crs.PlateCarree())
fig.colorbar(pcm, ax=[ax1, ax2], shrink=0.8)
fig.show()

#%%
# Gauss-Seidel
# ============
# The :py:func:`second <pyinterp.fill.gauss_seidel>` method consists of replacing
# all undefined values (NaN) in a grid using the Gauss-Seidel method by
# relaxation. This `link
# <https://math.berkeley.edu/~wilken/228A.F07/chr_lecture.pdf>`_ contains more
# information on the method used.
has_converged, filled = pyinterp.fill.gauss_seidel(grid)

#%%
# The image below illustrates the result:
fig = matplotlib.pyplot.figure(figsize=(10, 10))
fig.patch.set_alpha(0.0)
ax1 = fig.add_subplot(
    211, projection=cartopy.crs.PlateCarree(central_longitude=180))
pcm = ax1.pcolormesh(lons,
                     lats,
                     ds.mss.T,
                     cmap='jet',
                     transform=cartopy.crs.PlateCarree(),
                     vmin=-0.1,
                     vmax=0.1)
ax1.coastlines()
ax1.set_title("Original MSS")
ax1.set_extent([40, 170, -45, 30], crs=cartopy.crs.PlateCarree())
ax2 = fig.add_subplot(212, projection=cartopy.crs.PlateCarree(central_longitude=180))
pcm = ax2.pcolormesh(lons,
                     lats,
                     filled,
                     cmap='jet',
                     transform=cartopy.crs.PlateCarree(),
                     vmin=-0.1,
                     vmax=0.1)
ax2.coastlines()
ax2.set_title("MSS modified using Gauss-Seidel")
ax2.set_extent([40, 170, -45, 30], crs=cartopy.crs.PlateCarree())
fig.colorbar(pcm, ax=[ax1, ax2], shrink=0.8)
fig.show()
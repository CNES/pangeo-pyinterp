"""
*******
Binning
*******

Statistical data binning is a way to group several more or less continuous
values into a smaller number of *bins*. For example, if you have irregularly
distributed data over the oceans, you can organize these observations into a
lower number of geographical intervals (for example, by grouping them all five
degrees into latitudes and longitudes).

In this example, we will calculate drifter velocity statistics on the Black Sea
over a period of 9 years.
"""
import os
import pathlib
import cartopy.crs
import matplotlib
import matplotlib.pyplot
import numpy
import pyinterp
import pyinterp.backends.xarray
import xarray

DATASET = pathlib.Path(os.environ['DATASET'])
AOML = DATASET.joinpath("aoml_v2019.nc")

#%%
# The first step is to load the data into memory and create the interpolator
# object:
ds = xarray.open_dataset(AOML)

#%%
# Let's start by calculating the standard for vectors u and v.
norm = (ds.ud**2 + ds.vd**2)**0.5

#%%
# Now, we will describe the grid used to calculate our :py:class:`binned
# <pyinterp.Binning2D>` statics.
binning = pyinterp.Binning2D(
    pyinterp.Axis(numpy.arange(27, 42, 0.3), is_circle=True),
    pyinterp.Axis(numpy.arange(40, 47, 0.3)))
binning

#%%
# We push the loaded data into the different defined bins using :ref:`simple
# binning <bilinear_binning>`.
binning.clear()
binning.push(ds.lon, ds.lat, norm, True)

#%%
# .. note ::
#
#   If the processed data is larger than the available RAM, it's possible to use
#   Dask to parallel the calculation. To do this, an instance must be built,
#   then the data must be added using the :py:meth:`push_delayed
#   <pyinterp.Binning.push_delayed>` method. This method will return a graph,
#   which when executed will return a new instance containing the calculated
#   statics.
#
#   .. code:: python
#
#       binning = binning.push_delayed(lon, lat, data).compute()
#
# It is possible to retrieve other statistical :py:meth:`variables
# <pyinterp.Binning2D.variable>` such as variance, minimum, maximum, etc.
nearest = binning.variable('mean')

#%%
# Then, we push the loaded data into the different defined bins using
# :ref:`linear binning <bilinear_binning>`.
binning.clear()
binning.push(ds.lon, ds.lat, norm, False)
linear = binning.variable('mean')

#%%
# We visualize our result
fig = matplotlib.pyplot.figure(figsize=(10, 8))
ax1 = fig.add_subplot(211, projection=cartopy.crs.PlateCarree())
lon, lat = numpy.meshgrid(binning.x, binning.y, indexing='ij')
pcm = ax1.pcolormesh(lon,
                     lat,
                     nearest,
                     cmap='jet',
                     vmin=0,
                     vmax=1,
                     transform=cartopy.crs.PlateCarree())
ax1.coastlines()
ax1.set_title("Simple binning.")

ax2 = fig.add_subplot(212, projection=cartopy.crs.PlateCarree())
lon, lat = numpy.meshgrid(binning.x, binning.y, indexing='ij')
pcm = ax2.pcolormesh(lon,
                     lat,
                     linear,
                     cmap='jet',
                     vmin=0,
                     vmax=1,
                     transform=cartopy.crs.PlateCarree())
ax2.coastlines()
ax2.set_title("Linear binning.")
fig.colorbar(pcm, ax=[ax1, ax2], shrink=0.8)
fig.show()
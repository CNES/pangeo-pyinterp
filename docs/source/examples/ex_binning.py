"""
*******
Binning
*******

=========
Binning2D
=========

Statistical data binning is a way to group several more or less continuous
values into a smaller number of *bins*. For example, if you have irregularly
distributed data over the oceans, you can organize these observations into a
lower number of geographical intervals (for example, by grouping them all five
degrees into latitudes and longitudes).

In this example, we will calculate drifter velocity statistics on the Black Sea
over a period of 9 years.
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
ds = xarray.open_dataset(pyinterp.tests.aoml_path())

# %%
# Let's start by calculating the standard for vectors u and v.
norm = (ds.ud**2 + ds.vd**2)**0.5

# %%
# Now, we will describe the grid used to calculate our :py:class:`binned
# <pyinterp.Binning2D>` statics.
binning = pyinterp.Binning2D(
    pyinterp.Axis(numpy.arange(27, 42, 0.3), is_circle=True),
    pyinterp.Axis(numpy.arange(40, 47, 0.3)))
binning

# %%
# We push the loaded data into the different defined bins using :ref:`simple
# binning <bilinear_binning>`.
binning.clear()
binning.push(ds.lon, ds.lat, norm, True)

# %%
# .. note ::
#
#   If the processed data is larger than the available RAM, it's possible to use
#   Dask to parallel the calculation. To do this, an instance must be built,
#   then the data must be added using the :py:meth:`push_delayed
#   <pyinterp.Binning2D.push_delayed>` method. This method will return a graph,
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

# %%
# Then, we push the loaded data into the different defined bins using
# :ref:`linear binning <bilinear_binning>`.
binning.clear()
binning.push(ds.lon, ds.lat, norm, False)
linear = binning.variable('mean')

# %%
# We visualize our result
fig = matplotlib.pyplot.figure(figsize=(10, 8))
ax1 = fig.add_subplot(211, projection=cartopy.crs.PlateCarree())
lon, lat = numpy.meshgrid(binning.x, binning.y, indexing='ij')
pcm = ax1.pcolormesh(lon,
                     lat,
                     nearest,
                     cmap='jet',
                     shading='auto',
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
                     shading='auto',
                     vmin=0,
                     vmax=1,
                     transform=cartopy.crs.PlateCarree())
ax2.coastlines()
ax2.set_title("Linear binning.")
fig.colorbar(pcm, ax=[ax1, ax2], shrink=0.8)
fig.show()

# %%
# ===========
# Histogram2D
# ===========
#
# :py:class:`This class<pyinterp.Histogram2D>`, like the previous one, allows
# calculating a binning using distribution and obtains the median value of the
# pixels. histograms. In addition, this approach calculates the quantiles of the
#
# Note that the algorithm used defines a maximum size of the number of bins
# handled by each histogram. If the number of observations is greater than the
# capacity of the histogram, the histogram will be compressed to best present
# this distribution in limited memory size. The description of the exact
# algorithm is in the article `A Streaming Parallel Decision Tree Algorithm
# <http://jmlr.org/papers/v11/ben-haim10a.html>`_.
hist2d = pyinterp.Histogram2D(
    pyinterp.Axis(numpy.arange(27, 42, 0.3), is_circle=True),
    pyinterp.Axis(numpy.arange(40, 47, 0.3)))
hist2d

# %%
# We push the loaded data into the different defined bins using the method
# :py:meth:`push <pyinterp.Histogram2D.push>`.
hist2d.push(ds.lon, ds.lat, norm)

# %%
# We visualize the mean vs median of the distribution.
fig = matplotlib.pyplot.figure(figsize=(10, 8))
ax1 = fig.add_subplot(211, projection=cartopy.crs.PlateCarree())
lon, lat = numpy.meshgrid(binning.x, binning.y, indexing='ij')
pcm = ax1.pcolormesh(lon,
                     lat,
                     nearest,
                     cmap='jet',
                     shading='auto',
                     vmin=0,
                     vmax=1,
                     transform=cartopy.crs.PlateCarree())
ax1.coastlines()
ax1.set_title("Mean")

ax2 = fig.add_subplot(212, projection=cartopy.crs.PlateCarree())
lon, lat = numpy.meshgrid(binning.x, binning.y, indexing='ij')
pcm = ax2.pcolormesh(lon,
                     lat,
                     hist2d.variable("quantile", 0.5),
                     cmap='jet',
                     shading='auto',
                     vmin=0,
                     vmax=1,
                     transform=cartopy.crs.PlateCarree())
ax2.coastlines()
ax2.set_title("Median")
fig.colorbar(pcm, ax=[ax1, ax2], shrink=0.8)
fig.show()

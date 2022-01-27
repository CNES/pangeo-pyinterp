"""
*****************
Unstructured grid
*****************

Interpolation of unstructured grids.

The interpolation of this object is based on a :py:class:`R*Tree
<pyinterp.RTree>` structure. To begin with, we start by building this
object. By default, this object considers the WGS-84 geodetic coordinate system.
But you can define another one using the class :py:class:`System
<pyinterp.geodetic.System>`.
"""

# %%
import matplotlib.pyplot
import numpy

import pyinterp

mesh = pyinterp.RTree()

# %%
# Then, we will insert points into the tree. The class allows you to add points
# using two algorithms. The first one called :py:meth:`packing
# <pyinterp.RTree.packing>`, will enable you to enter the values in the tree at
# once. This mechanism is the recommended solution to create an optimized
# in-memory structure, both in terms of construction time and queries. When this
# is not possible, you can insert new information into the tree as you go along
# using the :py:meth:`insert <pyinterp.RTree.insert>` method.
SIZE = 2000
X0, X1 = 80, 170
Y0, Y1 = -45, 30
lons = numpy.random.uniform(low=X0, high=X1, size=(SIZE, ))
lats = numpy.random.uniform(low=Y0, high=Y1, size=(SIZE, ))
data = numpy.random.random(size=(SIZE, ))

# %%
# Populates the search tree
mesh.packing(numpy.vstack((lons, lats)).T, data)

# %%
# When the tree is created, you can interpolate data with three algorithms:
#
# * :py:meth:`Inverse Distance Weighting
#   <pyinterp.RTree.inverse_distance_weighting>` or IDW
# * :py:meth:`Radial Basis Function
#   <pyinterp.RTree.radial_basis_function>` or RBF
# * :py:meth:`Window Function
#   <pyinterp.RTree.window_function>`
#
# .. note::
#
#     When comparing an RBF to IDW, IDW will never predict values higher than
#     the maximum measured value or lower than the minimum measured value.
#     However, RBFs can predict values higher than the maximum values and lower
#     than the minimum measured values.
#
#     The window function restricts the analyzed data set to a range near the
#     point of interest. The weighting factor decreases the effect of points
#     further away from the interpolated section of the point.
#
# We start by interpolating using the IDW method
STEP = 1 / 32
mx, my = numpy.meshgrid(numpy.arange(X0, X1 + STEP, STEP),
                        numpy.arange(Y0, Y1 + STEP, STEP),
                        indexing="ij")

idw, neighbors = mesh.inverse_distance_weighting(
    numpy.vstack((mx.ravel(), my.ravel())).T,
    within=False,  # Extrapolation is forbidden
    k=11,  # We are looking for at most 11 neighbors
    radius=600000,
    num_threads=0)
idw = idw.reshape(mx.shape)

# %%
# Interpolation with RBF method
rbf, neighbors = mesh.radial_basis_function(
    numpy.vstack((mx.ravel(), my.ravel())).T,
    within=False,  # Extrapolation is forbidden
    k=11,  # We are looking for at most 11 neighbors
    radius=600000,
    rbf="thin_plate",
    num_threads=0)
rbf = rbf.reshape(mx.shape)

# %%
# Interpolation with a Window Function
wf, neighbors = mesh.window_function(
    numpy.vstack((mx.ravel(), my.ravel())).T,
    within=False,  # Extrapolation is forbidden
    k=11,
    radius=600000,
    wf="parzen",
    num_threads=0)
wf = wf.reshape(mx.shape)

# %%
# Let's visualize our interpolated data
fig = matplotlib.pyplot.figure(figsize=(10, 20))
ax1 = fig.add_subplot(311)
pcm = ax1.pcolormesh(mx, my, idw, cmap='jet', shading='auto', vmin=0, vmax=1)
ax1.set_title("IDW interpolation")
ax2 = fig.add_subplot(312)
pcm = ax2.pcolormesh(mx, my, rbf, cmap='jet', shading='auto', vmin=0, vmax=1)
ax2.set_title("RBF interpolation")
ax3 = fig.add_subplot(313)
pcm = ax3.pcolormesh(mx, my, wf, cmap='jet', shading='auto', vmin=0, vmax=1)
ax3.set_title("Window function interpolation")
fig.colorbar(pcm, ax=[ax1, ax2, ax3], shrink=0.8)
fig.show()

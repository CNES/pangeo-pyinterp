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

#%%
import matplotlib.pyplot
import numpy
import pyinterp

mesh = pyinterp.RTree()

#%%
# Then, we will insert points into the tree. The class allows you to add points
# using two algorithms. The first one, called :py:meth:`packing
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
# When the tree is created, you can interpolate data with two algorithms:
#
# * :py:meth:`Inverse Distance Weighting
#   <pyinterp.RTree.inverse_distance_weighting>` or IDW
# * :py:meth:`Radial Basis Function
#   <pyinterp.RTree.radial_basis_function>` or RBF
#
# .. note::
#
#     When comparing an RBF to IDW, IDW will never predict values higher than the
#     maximum measured value or lower than the minimum measured value. However,
#     RBFs can predict values higher than the maximum values and lower than the
#     minimum measured values.
#
# We start by interpolating using the IDW method
STEP = 1 / 32
mx, my = numpy.meshgrid(numpy.arange(X0, X1 + STEP, STEP),
                        numpy.arange(Y0, Y1 + STEP, STEP),
                        indexing="ij")

idw, neighbors = mesh.inverse_distance_weighting(
    numpy.vstack((mx.flatten(), my.flatten())).T,
    within=True,  # Extrapolation is forbidden
    k=11,  # We are looking for at most 11 neighbours
    num_threads=0)
idw = idw.reshape(mx.shape)

#%%
# The with the RBF method
rbf, neighbors = mesh.radial_basis_function(
    numpy.vstack((mx.flatten(), my.flatten())).T,
    within=True,  # Extrapolation is forbidden
    k=11,  # We are looking for at most 11 neighbours
    rbf="linear",
    num_threads=0)
rbf = rbf.reshape(mx.shape)

#%%
# Let's visualize our interpolated data
fig = matplotlib.pyplot.figure(figsize=(10, 8))
ax1 = fig.add_subplot(211)
pcm = ax1.pcolormesh(mx, my, idw, cmap='jet', shading='auto', vmin=0, vmax=1)
ax1.set_title("IDW interpolation")
ax2 = fig.add_subplot(212)
pcm = ax2.pcolormesh(mx, my, rbf, cmap='jet', shading='auto', vmin=0, vmax=1)
ax2.set_title("RBF interpolation")
fig.colorbar(pcm, ax=[ax1, ax2], shrink=0.8)
fig.show()
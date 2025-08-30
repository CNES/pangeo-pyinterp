"""
*****************
Unstructured grid
*****************

Interpolation of unstructured grids.

The interpolation of this object is based on a :py:class:`R*Tree
<pyinterp.RTree>` structure. To begin with, we start by building this
object. By default, this object considers the WGS-84 geodetic coordinate system.
But you can define another one using the class :py:class:`Spheroid
<pyinterp.geodetic.Spheroid>`.

.. note::

  By default, the class converts coordinates from the WGS-84 geodetic system
  to a Cartesian coordinate system. However, if you set the parameter ``ecef``
  to ``True``, this transformation is disabled. In this case, both input and
  output coordinates are expected to be in the Cartesian coordinate system,
  and the RTree will handle only Cartesian coordinates without any conversion.
"""

# %%
import matplotlib.pyplot
import numpy

import pyinterp

mesh = pyinterp.RTree()


# %%
# We will create a synthetic topography-like field to illustrate the
# interpolation methods.
def topography_field(lons, lats):
    """A topography-like field."""
    return (numpy.sin(numpy.radians(lons) * 3) *
            numpy.cos(numpy.radians(lats) * 2) +
            0.5 * numpy.sin(numpy.radians(lons) * 5) *
            numpy.sin(numpy.radians(lats) * 4))


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
generator = numpy.random.Generator(numpy.random.PCG64(0))
lons = generator.uniform(low=X0, high=X1, size=(SIZE, ))
lats = generator.uniform(low=Y0, high=Y1, size=(SIZE, ))
data = topography_field(lons, lats)

# %%
# Populates the search tree
mesh.packing(numpy.vstack((lons, lats)).T, data)

# %%
# When the tree is created, you can interpolate data with four algorithms:
#
# * :py:meth:`Inverse Distance Weighting
#   <pyinterp.RTree.inverse_distance_weighting>` or IDW
# * :py:meth:`Radial Basis Function
#   <pyinterp.RTree.radial_basis_function>` or RBF
# * :py:meth:`Window Function
#   <pyinterp.RTree.window_function>`
# * :py:meth:`Universal Kriging
#   <pyinterp.RTree.universal_kriging>`
#
# Inverse Distance Weighting (IDW), Radial Basis Function (RBF), and Kriging are
# all interpolation methods used to estimate a value for a target location based
# on the values of surrounding sample points. However, each method approaches
# this estimation differently.
#
# IDW uses a weighted average of the surrounding sample points, where the weight
# assigned to each point is inversely proportional to its distance from the
# target location. The further away a sample point is from the target location,
# the less influence it has on the estimated value. This method is relatively
# simple to implement and computationally efficient, but it can produce
# over-smoothed results in areas with a lot of sample points and under-smoothed
# results in areas with few sample points.
#
# The Window Function method, much like IDW, uses a weighted average of the
# surrounding sample points. However, instead of the weight being solely
# determined by the inverse distance, it is determined by a kernel function (the
# "window"). This function gives the most weight to points at the center of the
# window and progressively less weight to points further away. This can
# sometimes provide a smoother result than IDW.
#
# RBF, on the other hand, models the spatial relationship between sample points
# and the target location by using a mathematical function (radial basis
# function) that is based on the distance between the points. The radial basis
# function is usually Gaussian, multiquadric, or inverse multiquadric. The
# estimated value at the target location is obtained by summing up the weighted
# contributions of all sample points. This method is more flexible than IDW as
# it can produce a wide range of interpolation results, but it can also be
# computationally expensive and susceptible to overfitting if not implemented
# carefully.
#
# Kriging, also known as Gaussian process regression, is a geostatistical method
# that models the spatial structure of the underlying data by using a covariance
# matrix. The estimated value at the target location is obtained by solving a
# set of linear equations that balance the fit to the sample points and the
# smoothness of the estimated surface. Kriging can produce more accurate results
# than IDW and RBF in many cases, but it requires a good understanding of the
# spatial structure of the data and can be computationally demanding.
#
# In summary, the choice of interpolation method is a trade-off between
# simplicity, computational cost, and the quality of the results.
#
# * **Inverse Distance Weighting (IDW)** and **Window Function** are
#   straightforward and fast, making them excellent choices for quick
#   interpolations or when dealing with very large datasets where
#   computational efficiency is a priority. However, they may produce overly
#   smoothed results.
#
# * **Radial Basis Functions (RBF)** offer more flexibility and can capture
#   more complex spatial relationships, but they come at a higher
#   computational cost and require careful parameter tuning to avoid
#   overfitting.
#
# * **Universal Kriging** is the most sophisticated method, often yielding the
#   most accurate results by modeling the underlying spatial correlation of the
#   data. This accuracy, however, comes with the highest computational expense
#   and requires a good understanding of geostatistics to configure properly.
#
# Ultimately, the best method depends on the specific requirements of your
# application, including the nature of your data, the desired accuracy, and
# the available computational resources. The following visual comparison will
# help illustrate the practical differences between these four techniques.
#
# We start by interpolating using the IDW method
STEP = 1 / 32
mx, my = numpy.meshgrid(numpy.arange(X0, X1 + STEP, STEP),
                        numpy.arange(Y0, Y1 + STEP, STEP),
                        indexing='ij')

idw, neighbors = mesh.inverse_distance_weighting(
    numpy.vstack((mx.ravel(), my.ravel())).T,
    within=False,  # Extrapolation is forbidden
    k=11,  # We are looking for at most 11 neighbors
    num_threads=0)
idw = idw.reshape(mx.shape)

# %%
# Interpolation with RBF method
rbf, neighbors = mesh.radial_basis_function(
    numpy.vstack((mx.ravel(), my.ravel())).T,
    within=False,  # Extrapolation is forbidden
    k=11,  # We are looking for at most 11 neighbors
    rbf='thin_plate',
    smooth=1e-4,
    num_threads=0)
rbf = rbf.reshape(mx.shape)

# %%
# Interpolation with a Window Function
wf, neighbors = mesh.window_function(
    numpy.vstack((mx.ravel(), my.ravel())).T,
    within=False,  # Extrapolation is forbidden
    k=11,
    wf='parzen',
    num_threads=0)
wf = wf.reshape(mx.shape)

# %%
# Interpolation with a Universal Kriging
kriging, neighbors = mesh.universal_kriging(
    numpy.vstack((mx.ravel(), my.ravel())).T,
    within=False,  # Extrapolation is forbidden
    k=11,
    covariance='gaussian',
    alpha=1_000_000,
    num_threads=0)
kriging = kriging.reshape(mx.shape)

# %%
# Let's visualize the "true" field
vmax = 1.2
vmin = -1.3
fig = matplotlib.pyplot.figure(figsize=(6, 5))
ax = fig.add_subplot(111)
pcm = ax.pcolormesh(mx,
                    my,
                    topography_field(mx, my),
                    cmap='jet',
                    shading='auto',
                    vmin=vmin,
                    vmax=vmax)
ax.set_title('True field')
fig.colorbar(pcm, ax=ax, shrink=0.8, location='bottom')

# %%
# Let's visualize our interpolated data

fig = matplotlib.pyplot.figure(figsize=(12, 10))
fig.suptitle('Comparison of Interpolation Methods', fontsize=16)

# IDW interpolation
ax1 = fig.add_subplot(221)
pcm = ax1.pcolormesh(mx,
                     my,
                     idw,
                     cmap='jet',
                     shading='auto',
                     vmin=vmin,
                     vmax=vmax)
ax1.set_title('IDW interpolation')

# RBF interpolation
ax2 = fig.add_subplot(222)
pcm = ax2.pcolormesh(mx,
                     my,
                     rbf,
                     cmap='jet',
                     shading='auto',
                     vmin=vmin,
                     vmax=vmax)
ax2.set_title('RBF interpolation')

# Window function interpolation
ax3 = fig.add_subplot(223)
pcm = ax3.pcolormesh(mx,
                     my,
                     wf,
                     cmap='jet',
                     shading='auto',
                     vmin=vmin,
                     vmax=vmax)
ax3.set_title('Window function interpolation')

# Universal Kriging interpolation
ax4 = fig.add_subplot(224)
pcm = ax4.pcolormesh(mx,
                     my,
                     kriging,
                     cmap='jet',
                     shading='auto',
                     vmin=vmin,
                     vmax=vmax)
ax4.set_title('Universal Kriging interpolation')

fig.colorbar(pcm, ax=[ax1, ax2, ax3, ax4], shrink=0.6, location='bottom')

# Compute the true field values for the grid
true_field = topography_field(mx.ravel(), my.ravel()).reshape(mx.shape)

# Calculate relative errors for each interpolation method
idw_error = numpy.abs((idw - true_field) / true_field)
rbf_error = numpy.abs((rbf - true_field) / true_field)
wf_error = numpy.abs((wf - true_field) / true_field)
kriging_error = numpy.abs((kriging - true_field) / true_field)

# %%
# Plot the relative errors
fig = matplotlib.pyplot.figure(figsize=(12, 10))
fig.suptitle('Relative Errors', fontsize=16)

# IDW Relative Error
ax1 = fig.add_subplot(221)
pcm = ax1.pcolormesh(mx,
                     my,
                     idw_error,
                     cmap='jet',
                     shading='auto',
                     vmin=0,
                     vmax=1)
ax1.set_title('IDW Relative Error')

# RBF Relative Error
ax2 = fig.add_subplot(222)
ax2.pcolormesh(mx, my, rbf_error, cmap='jet', shading='auto', vmin=0, vmax=1)
ax2.set_title('RBF Relative Error')

# Window Function Relative Error
ax3 = fig.add_subplot(223)
ax3.pcolormesh(mx, my, wf_error, cmap='jet', shading='auto', vmin=0, vmax=1)
ax3.set_title('Window Function Relative Error')

# Universal Kriging Relative Error
ax4 = fig.add_subplot(224)
ax4.pcolormesh(mx,
               my,
               kriging_error,
               cmap='jet',
               shading='auto',
               vmin=0,
               vmax=1)
ax4.set_title('Universal Kriging Relative Error')

# Add a single colorbar for all error plots
cbar = fig.colorbar(pcm,
                    ax=[ax1, ax2, ax3, ax4],
                    shrink=0.8,
                    location='bottom')
cbar.set_label('Relative Error')

Xarray
------

The :doc:`previous chapter <examples>` presents the full operation of all the
functions implemented in this library and interfaces with the `Xarray
<http://xarray.pydata.org/en/stable/index.html>`_ library to build 2D, 3D, and
4D grids:

* :py:class:`pyinterp.backends.xarray.Grid2D`
* :py:class:`pyinterp.backends.xarray.Grid3D`
* :py:class:`pyinterp.backends.xarray.Grid4D`

These grids are the entry points for all functions in the library, except for
the :py:class:`RTree <pyinterp.RTree>` class. It is advised to use the ``Xarray``
library to build the grids needed for the functions as we have seen in the
previous examples.

The module ``pyinterp.backends.xarray`` provides the class
:py:class:`RegularGridInterpolator
<pyinterp.backends.xarray.RegularGridInterpolator>` that automatically builds
the interpolator compatible with the Xarray, i.e. the 2D, 3D, or 4D grid
depending on the number of dimensions of the array. A :py:meth:`unique method
<pyinterp.backends.xarray.RegularGridInterpolator.__call__>` allows performing a
linear or spline interpolation. For example:

.. code:: python

    import pyinterp.backends.xarray as pbx
    import numpy as np
    import xarray as xr

    ds = xr.load_dataset("tests/dataset/mss.nc")
    interpolator = pbx.RegularGridInterpolator(ds.mss)

    mx, my = np.meshgrid(np.arange(-180, 180, 1),
                         np.arange(-89, 89, 1),
                         indexing='ij')

    mss = interpolator(dict(lon=mx.flatten(), lat=my.flatten()))
    mss = interpolator(
        dict(lon=mx.flatten(), lat=my.flatten()),
        method='bicubic')
    mss = mss.reshape(mx.shape)

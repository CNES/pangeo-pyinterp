Tutorial
--------

2D interpolation
================

Bivariate
#########

Perform a bivariate interpolation of gridded data points.

The distribution contains a 2D field ``mss.nc`` that will be used in this help.
This file is located in the ``tests/dataset`` directory at the root of the
project.

.. warning ::

    This file is an old version of the sub-sampled quarter step MSS CNES/CLS. Do
    not use it for scientific purposes, download the latest updated
    high-resolution version instead `here <https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/mss.html>`_.

The first step is to load the data into memory:

.. code:: python

    import netCDF4
    import pyinterp.bivariate

    ds = netCDF4.Dataset("tests/dataset/mss.nc")

Afterwards, build the axes associated with the grid:

.. code:: python

    import pyinterp.core

    x_axis = pyinterp.core.Axis(ds.variables["lon"][:], is_circle=True)
    y_axis = pyinterp.core.Axis(ds.variables["lat"][:])

Finally, we can construct the interpolation object:

.. code:: python

    # The shape of the bivariate values must be (len(x_axis), len(y_axis))
    mss = ds.variables["mss"][:].T
    # The undefined values must be set to nan.
    mss[mss.mask] = float("nan")
    interpolator = pyinterp.bivariate.Bivariate(x_axis, y_axis, mss.data)

We will then build the coordinates on which we want to interpolate our grid:

.. code:: python

    import numpy as np

    # The coordinates used for interpolation are shifted to avoid using the
    # points of the bivariate function.
    mx, my = np.meshgrid(np.arange(-180, 180, 1) + 1 / 3.0,
                         np.arange(-89, 89, 1) + 1 / 3.0,
                         indexing='ij')

The grid is interpolated to the desired coordinates: ::

    mss = interpolator.evaluate(mx.flatten(), my.flatten()).reshape(mx.shape)

An experimental module of the library simplifies the use of the library by using
XArray and CF information contained in NetCDF files. This module implements all
the other interpolators of the regular grids presented below.

.. code:: python

    import pyinterp.backends.xarray
    import xarray as xr

    ds = xr.open_dataset("tests/dataset/mss.nc")
    interpolator = pyinterp.backends.xarray.Bivariate(ds, "mss")
    mss = interpolator.evaluate(dict(lon=mx.flatten(), lat=my.flatten()))

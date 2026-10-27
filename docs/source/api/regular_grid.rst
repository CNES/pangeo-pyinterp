Interpolation on Regular Grids
==============================

Interpolation of data defined on regular (possibly unevenly spaced) grids.
For an ``xarray.DataArray``, use
:class:`~pyinterp.backends.xarray.RegularGridInterpolator`, which identifies
the axes and selects the interpolator from the number of dimensions;
otherwise call the functions below on a :class:`pyinterp.Grid` built from
:doc:`axes <axis>` and a ``numpy`` array.

.. currentmodule:: pyinterp.backends.xarray

Classes
-------

.. autosummary::
   :toctree: _generated/

   RegularGridInterpolator

.. currentmodule:: pyinterp

Interpolation Functions
-----------------------

.. autosummary::
   :toctree: _generated/

   univariate
   bivariate
   trivariate
   quadrivariate
   univariate_derivative

Gap Filling
===========

.. currentmodule:: pyinterp.core.fill

Routines for filling undefined (NaN) values in gridded data. These functions
use iterative methods to interpolate missing values based on surrounding
known values.

Each function accepts a configuration object created from the corresponding
class in :py:mod:`pyinterp.core.config.fill`.

.. autosummary::

   fft_inpaint
   gauss_seidel
   loess
   multigrid

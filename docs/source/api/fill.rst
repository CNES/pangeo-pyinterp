Gap Filling
===========
Algorithms that replace the undefined values (NaN) of a grid, typically
before interpolating near coastlines or data gaps.
:func:`~pyinterp.fill.gauss_seidel`, :func:`~pyinterp.fill.multigrid` and
:func:`~pyinterp.fill.fft_inpaint` solve the problem iteratively and modify
the 2D array in place, :func:`~pyinterp.fill.loess` returns a new array, and
:func:`~pyinterp.fill.matrix` and :func:`~pyinterp.fill.vector` fill by linear
interpolation. The iterative methods and LOESS accept keyword arguments or a
configuration object (see :doc:`config`).

.. currentmodule:: pyinterp.fill

Algorithms
----------

.. autosummary::
   :toctree: _generated/

   fft_inpaint
   gauss_seidel
   loess
   multigrid
   matrix
   vector

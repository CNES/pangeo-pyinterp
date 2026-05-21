Dask Integration
================

The :mod:`pyinterp.dask` module provides helpers that compute the
:doc:`statistics` containers in parallel on
`Dask <https://docs.dask.org/>`_ arrays. Each helper processes blocks
independently using the same C++ kernels exposed by the in-memory API, then
merges the per-block results with the ``+=`` operator implemented by the
underlying holders.

Dask is an optional dependency; install it with ``pip install dask[array]``.

.. currentmodule:: pyinterp.dask

Overview
--------

The functions in this module are thin wrappers that mirror the eager classes
of :mod:`pyinterp`. The table below maps each Dask helper to the in-memory
counterpart it produces:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Dask helper
     - Returns
     - Eager equivalent
   * - :func:`descriptive_statistics`
     - :class:`pyinterp.DescriptiveStatistics`
     - :class:`pyinterp.DescriptiveStatistics`
   * - :func:`tdigest`
     - :class:`pyinterp.TDigest`
     - :class:`pyinterp.TDigest`
   * - :func:`binning1d`
     - :class:`pyinterp.Binning1D`
     - :class:`pyinterp.Binning1D`
   * - :func:`binning2d`
     - :class:`pyinterp.Binning2D`
     - :class:`pyinterp.Binning2D`
   * - :func:`histogram2d`
     - :class:`pyinterp.Histogram2D`
     - :class:`pyinterp.Histogram2D`

Descriptive statistics & quantiles
----------------------------------

.. autosummary::
   :toctree: _generated/

   descriptive_statistics
   tdigest

Binning containers
------------------

.. autosummary::
   :toctree: _generated/

   binning1d
   binning2d

Histograms
----------

.. autosummary::
   :toctree: _generated/

   histogram2d

See also
--------

* :doc:`statistics` — eager (in-memory) versions of the containers returned
  by the helpers above.

Statistics & Binning
====================
Containers that accumulate statistics incrementally, so that data can be
pushed in several batches and partial results merged with ``+=``. Use
:class:`~pyinterp.DescriptiveStatistics` for moments over arrays, the binning
and histogram classes to compute statistics on a grid, and
:class:`~pyinterp.TDigest` for approximate quantiles.

.. note::

   All the containers documented here can be populated in parallel from
   `Dask <https://docs.dask.org/>`_ arrays using the helpers in
   :doc:`dask`.

.. currentmodule:: pyinterp

Descriptive Statistics
----------------------

.. autosummary::
   :toctree: _generated/

   DescriptiveStatistics

Binning Containers
------------------

.. autosummary::
   :toctree: _generated/

   Binning1D
   Binning1DFloat32
   Binning1DFloat64
   Binning2D
   Binning2DFloat32
   Binning2DFloat64

Histograms
----------

.. autosummary::
   :toctree: _generated/

   Histogram2D
   Histogram2DFloat32
   Histogram2DFloat64

Quantile Estimation
-------------------

.. autosummary::
   :toctree: _generated/

   TDigest
   TDigestFloat32
   TDigestFloat64

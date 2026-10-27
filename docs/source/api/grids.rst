Grid Structures
===============

.. currentmodule:: pyinterp.backends.xarray

Xarray Backends
---------------
High-level grid containers compatible with xarray.

These classes provide a convenient interface for creating interpolators directly
from ``xarray.DataArray`` objects. They automatically handle coordinate systems,
including temporal axes and geodetic units.

.. autosummary::
    :toctree: _generated

    Grid2D
    Grid3D
    Grid4D

.. currentmodule:: pyinterp

Core Grids
----------
The fundamental grid structure used internally by the library.

.. autosummary::
    :toctree: _generated

    Grid
    GridHolder

Core Grids Aliases
------------------
Aliases of :class:`pyinterp.Grid` used in type hints; they are the same class
at runtime.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``Grid1D``
     - Alias for :class:`pyinterp.Grid` with one axis.
   * - ``Grid2D``
     - Alias for :class:`pyinterp.Grid` with two axes.
   * - ``Grid3D``
     - Alias for :class:`pyinterp.Grid` with three axes; the third one may be
       a :class:`pyinterp.TemporalAxis`.
   * - ``Grid4D``
     - Alias for :class:`pyinterp.Grid` with four axes; the third one may be
       a :class:`pyinterp.TemporalAxis`.

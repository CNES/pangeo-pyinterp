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
Aliases for commonly used grid.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :class:`Grid1D`
     - Alias for :class:`pyinterp.Grid` with one spatial dimension.
   * - :class:`Grid2D`
     - Alias for :class:`pyinterp.Grid` with two spatial dimensions.
   * - :class:`Grid3D`
     - Alias for :class:`pyinterp.Grid` with three spatial dimensions.
   * - :class:`Grid4D`
     - Alias for :class:`pyinterp.Grid` with three spatial dimensions and one temporal dimension.

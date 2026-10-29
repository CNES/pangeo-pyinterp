Conservative Regridding
=======================
First order conservative regridding between rectilinear grids: each target
cell receives the mean of the source values weighted by the area of their
overlap with the cell, which preserves the integral of the field. Unlike the
interpolators, which sample a field at points, it averages the field: use it
to coarsen a grid, or to regrid fluxes and budgets. See
:ref:`example_conservative_regrid`.

The cells of a dimension are given by their centers, a 1D array, or by their
bounds, an array of shape (n, 2) as the CF bounds variables. The bounds of
cells given by their centers are the midpoints between consecutive centers,
latitudes being clamped to ±90°; a dimension of a single cell must therefore
be given by its bounds. Longitudes are periodic: the source and target grids
can use different conventions, such as [0, 360) and [-180, 180).

Undefined source values (NaN) are ignored by default: a target cell receives
the mean of the defined values it overlaps (``conservative_normed`` in
xESMF). ``min_valid_fraction`` sets the fraction of the area of the target
cell that the defined values must cover, and ``normalize=False`` counts the
undefined values as zero (``conservative`` in xESMF).

Sphere and ellipsoid
--------------------

The area of the cell :math:`[\lambda_1, \lambda_2] \times [\varphi_1,
\varphi_2]` is :math:`R^2 (\lambda_2 - \lambda_1)(\sin\beta_2 -
\sin\beta_1)`, with :math:`\beta` the authalic latitude and :math:`R` the
authalic radius. The overlap of two cells is therefore the product of the
overlaps of their sides, and the regridding is applied along each axis in
turn, without building a 2D weight matrix.

The areas are computed **on a sphere** by default, where :math:`\beta =
\varphi`, as ESMF and xESMF do. Pass ``spheroid=Spheroid()``
(:class:`pyinterp.geometry.geographic.Spheroid`, WGS-84 by default) to compute
them exactly on the ellipsoid: coarsening a global field to 1°, the results
change by about :math:`10^{-6}` of the field.

Differences with xESMF
----------------------

The cells are longitude/latitude boxes, bounded by meridians and parallels.
ESMF draws the edges of the cells as great circles, which bulge towards the
poles by an amount proportional to the square of the width of the cell and to
:math:`\sin 2\varphi`: the results differ most at mid-latitudes, and least at
the equator and at the poles. Largest differences between pyinterp and xESMF,
in fraction of the amplitude of a smooth global field coarsened from 0.25°,
by band of latitude:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15 15 15

   * - Target grid
     - 0°–15°
     - 15°–45°
     - 45°–75°
     - 75°–85°
     - 85°–90°
   * - 1°
     - :math:`7 \times 10^{-6}`
     - :math:`2 \times 10^{-5}`
     - :math:`2 \times 10^{-5}`
     - :math:`3 \times 10^{-6}`
     - :math:`3 \times 10^{-7}`
   * - 5°
     - :math:`2 \times 10^{-4}`
     - :math:`4 \times 10^{-4}`
     - :math:`4 \times 10^{-4}`
     - :math:`7 \times 10^{-5}`
     - :math:`5 \times 10^{-6}`

Next to undefined values, the slivers of the great circles reach defined
values that the boxes do not: some target cells get a value from xESMF and NaN
from pyinterp (96 of the 64,800 cells of the 1° grid with the mask tested),
and the means of partly undefined cells differ more, up to
:math:`3 \times 10^{-3}` of the field at 5°. pyinterp preserves the integral
of the boxes to :math:`10^{-12}`, xESMF to about :math:`10^{-5}`.
Curvilinear and unstructured grids are not supported.

xarray accessor
---------------

.. currentmodule:: pyinterp.backends.xarray

.. autosummary::
   :toctree: _generated/

   DataArrayAccessor.regrid
   DatasetAccessor.regrid

NumPy functions
---------------

.. currentmodule:: pyinterp.regrid

.. autosummary::
   :toctree: _generated/

   conservative
   cell_bounds
   cells

Low-level functions
-------------------

.. currentmodule:: pyinterp.core.regrid

.. autosummary::
   :toctree: _generated/

   conservative
   latitude_measure
   overlap_matrix

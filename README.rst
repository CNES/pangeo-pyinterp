########
pyinterp
########

|PyPI| |Conda| |Build Status| |License| |Binder|

**Fast interpolation of geo-referenced data for Python and xarray.**

pyinterp interpolates gridded fields in 2, 3 or 4 dimensions and scattered
observations, such as model outputs, satellite tracks or in situ
measurements. Its core is written in C++ and runs on all the cores of the
machine; it knows that longitudes wrap around the globe, that a time axis
holds dates, and that a field may have undefined values, near the coasts for
instance.

Installation
============

.. code-block:: bash

   pip install "pyinterp[xarray]"

or, with conda:

.. code-block:: bash

   conda install -c conda-forge pyinterp

Binary wheels are published for Linux, macOS (Apple Silicon) and Windows, for
Python 3.12 and later. The ``xarray`` extra is needed only for the xarray
interface; see the `installation guide
<https://cnes.github.io/pangeo-pyinterp/setup.html>`_.

Quick start
===========

Once ``pyinterp`` is imported, xarray data arrays have a ``pyinterp``
accessor that works like ``DataArray.interp``:

.. code-block:: python

   import numpy as np
   import xarray as xr

   import pyinterp
   import pyinterp.tests

   # A global mean sea surface, stored as (lat, lon) with longitudes in [0, 360)
   ds = pyinterp.tests.load_grid2d()

   # Interpolate on a new grid: the longitude axis is periodic, so the grid can
   # be queried in [-180, 180)
   mss = ds.mss.pyinterp.interp(
       lon=np.arange(-180, 180, 0.1),
       lat=np.arange(-80, 80, 0.1),
       method="bicubic",
   )

   # Interpolate point by point, along a track
   track = ds.mss.pyinterp.interp(
       lon=xr.DataArray(np.linspace(-170, 170, 1000), dims="obs"),
       lat=xr.DataArray(np.linspace(-60, 60, 1000), dims="obs"),
   )

Scattered observations are indexed in an R*-tree:

.. code-block:: python

   from pyinterp.geometry import geographic

   rng = np.random.default_rng(0)
   lon, lat = rng.uniform(-180, 180, 10_000), rng.uniform(-80, 80, 10_000)
   values = np.cos(np.radians(lat)) * np.sin(np.radians(lon))

   # Index the observations on the WGS-84 ellipsoid
   tree = pyinterp.RTree3D(spheroid=geographic.Spheroid())
   tree.packing(np.column_stack((lon, lat)), values)

   # Estimate the field at new positions from the 11 nearest observations
   estimate, neighbors = pyinterp.inverse_distance_weighting(
       tree, np.array([[10.0, 45.0], [-70.0, -30.0]]), k=11
   )

The `gallery <https://cnes.github.io/pangeo-pyinterp/auto_examples/index.html>`_
shows these and the other features on real data.

Why pyinterp?
=============

``xarray.DataArray.interp`` and ``scipy.interpolate`` are general-purpose
tools. pyinterp is built for geo-referenced data:

* **Periodic longitudes**: a grid stored in [0, 360) is interpolated in
  [-180, 180), and across the antimeridian, without rolling the data.
* **Dates**: time axes of type ``datetime64`` are handled with integer
  arithmetic, at any resolution, without conversion to floats.
* **Speed**: the interpolators are written in C++ and process the points in
  parallel; the axes find the cell of a point by a direct computation when
  they are regular. Bilinear interpolation runs 5 times faster than SciPy on
  one core and 34 times faster on all the cores; see the `benchmarks
  <https://cnes.github.io/pangeo-pyinterp/benchmarks.html>`_.
* **Undefined values**: the gaps of a field, such as the land of an ocean
  grid, can be filled beforehand, so that the windowed methods (bicubic,
  splines, Akima, Steffen...) do not spread them; at the edges of the grid,
  their window can shrink to the available data.
* **Beyond grids**: scattered data, statistics and geometry use the same
  coordinates and the same conventions.

What is inside
==============

* `Regular grids
  <https://cnes.github.io/pangeo-pyinterp/api/regular_grid.html>`_: 1D to 4D
  interpolation (nearest, bilinear, inverse distance weighting, bicubic,
  splines, Akima, Steffen, polynomial), with temporal axes, through
  `xarray <https://cnes.github.io/pangeo-pyinterp/api/grids.html>`_ or NumPy.
* `Scattered data
  <https://cnes.github.io/pangeo-pyinterp/api/unstructured_grid.html>`_:
  R*-trees in Cartesian or geodetic coordinates, with inverse distance
  weighting, radial basis functions, kriging and window functions, and
  `optimal interpolation
  <https://cnes.github.io/pangeo-pyinterp/api/optimal_interpolation.html>`_.
* `Gap filling <https://cnes.github.io/pangeo-pyinterp/api/fill.html>`_:
  LOESS, Gauss-Seidel relaxation, multigrid and FFT inpainting.
* `Statistics <https://cnes.github.io/pangeo-pyinterp/api/statistics.html>`_:
  1D and 2D binning, histograms, descriptive statistics and T-Digest
  quantiles, which can be merged across `Dask
  <https://cnes.github.io/pangeo-pyinterp/api/dask.html>`_ chunks.
* `Geometry <https://cnes.github.io/pangeo-pyinterp/api/geometry.html>`_:
  bindings to Boost.Geometry in Cartesian and geographic coordinates,
  `geohash <https://cnes.github.io/pangeo-pyinterp/api/geohash.html>`_
  encoding, and `orbit
  <https://cnes.github.io/pangeo-pyinterp/api/orbit.html>`_ utilities for
  satellite tracks.

The `design notes <https://cnes.github.io/pangeo-pyinterp/design.html>`_
explain how the axes, the R*-trees and the coordinate conversions work.

Contributing and citing
=======================

pyinterp is developed by CNES and CLS under the BSD-3-Clause license. Bug
reports, questions and pull requests are welcome on `GitHub
<https://github.com/CNES/pangeo-pyinterp/issues>`_; see the `contributing
guide <https://github.com/CNES/pangeo-pyinterp/blob/master/CONTRIBUTING.md>`_.
If pyinterp helps your research, please cite it using the metadata of the
`CITATION.cff <https://github.com/CNES/pangeo-pyinterp/blob/master/CITATION.cff>`_
file.

.. |PyPI| image:: https://img.shields.io/pypi/v/pyinterp.svg
    :target: https://pypi.org/project/pyinterp/
.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/pyinterp.svg
    :target: https://anaconda.org/conda-forge/pyinterp
.. |Build Status| image:: https://github.com/CNES/pangeo-pyinterp/actions/workflows/ci.yaml/badge.svg
    :target: https://github.com/CNES/pangeo-pyinterp/actions
.. |License| image:: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
.. |Binder| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/CNES/pangeo-pyinterp/master?urlpath=lab/tree/notebooks/auto_examples/

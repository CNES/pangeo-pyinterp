Changelog
#########

Unreleased
----------

New Features
~~~~~~~~~~~~

* **xarray accessor**: data arrays and datasets have a ``pyinterp`` accessor
  once ``pyinterp`` and xarray are imported, in either order; ``pyinterp``
  still does not import xarray, the accessor is registered when xarray is
  imported. ``da.pyinterp.interp`` and ``ds.pyinterp.interp`` follow
  ``xarray.DataArray.interp``: the coordinates are passed as keyword
  arguments, one-dimensional coordinates define a grid and data arrays sharing
  a dimension are interpolated point by point. The result is a ``DataArray``
  (or a ``Dataset``) labelled with the coordinates, which keeps the name and
  the attributes of the variables. The other keyword arguments are the options
  of the interpolator.
* **Interpolation on a grid of points**: ``core.bivariate_outer``,
  ``core.trivariate_outer`` and ``core.quadrivariate_outer`` interpolate with
  the geometric methods (``bilinear``, ``idw``, ``nearest``) at all the
  combinations of 1D coordinates (outer product). Each coordinate is located
  once on its axis instead of once per point, and the result is that of the
  point-by-point interpolation, periodic longitudes, undefined values and
  ``bounds_error`` included. The ``pyinterp`` accessor uses them when every
  coordinate is a scalar or a 1D array along its own dimension: regridding a
  global 0.25° field onto a 0.1° grid (5.8 million points) with the bilinear
  method takes 14 ms on one core instead of 144 ms (7 ms instead of 29 ms on
  all cores).
* **Conservative regridding**: ``da.pyinterp.regrid`` and
  ``ds.pyinterp.regrid`` regrid variables between rectilinear grids, giving
  each target cell the mean of the source values weighted by the area of
  their overlap with the cell, which preserves the integral of the field
  (``conservative_normed`` in xESMF, or ``conservative`` with
  ``normalize=False``). The target cells are given by their centers or by
  their bounds, keyword arguments named after the dimensions as for
  ``interp``; the CF bounds variables of a dataset give the source cells, and
  the result holds the bounds of the target cells. Longitudes are periodic,
  so the grids can use different conventions; undefined values are ignored,
  with an optional minimum fraction of the target cell they must cover; the
  other dimensions (time, depth) and their coordinates are kept. The areas
  are computed on a sphere, as in ESMF, or exactly on an ellipsoid with
  ``spheroid``. ``pyinterp.regrid.conservative`` is the NumPy layer. The
  overlap of two longitude/latitude boxes factorizes into an overlap along
  each axis, so the regridding applies two sparse 1D operators and computes
  no 2D weights: coarsening a global 0.25° grid to 1° takes 1.2 ms on one
  core (0.7 ms on all cores), where xESMF takes 12 s to compute its weights
  and 124 ms to apply them. The cells are bounded by parallels, where ESMF
  draws great circles: the results differ by up to 2·10⁻⁵ of the amplitude of
  a smooth field at mid-latitudes with 1° cells, 4·10⁻⁴ with 5° cells.

* **Reduced Gaussian grids**: ``pyinterp.reduced_gaussian`` interpolates
  the reduced Gaussian grids of the ECMWF analyses and forecasts in 2D, in 3D
  (a column of model levels per point) and in time, from numeric inputs only:
  the latitudes of the parallels, their number of points and the values, so no
  GRIB library is needed. A ``Mesh`` describes the geometry; a ``Zone``, built
  from the mesh alone before any value is read, holds the part a box or a set
  of points reads, as ranges of the flat indices of the mesh (exactly the
  corners of the points: 3,507 of the 2.1 million points of an N640 mesh for a
  one-hour pass at 1 Hz); a ``Grid`` reads float32 or float64 fields without
  copy, and a ``Series`` interpolates between time steps that may each hold
  their own zone (``time_zones``). Each point is located once on the mesh, and
  its trapezoidal cell serves every field, level and time step. A date reads
  the two steps that bracket it, the nearest one, or the two ends of its
  analysis interval, read off the remaining end when one is missing;
  ``map_availability`` counts the missing ends. ``apply`` computes values per
  point from the interpolated columns with a Python function called on blocks
  of points, or with a numba kernel (``column_kernel``) called for each point
  from several threads without the GIL: a vertical integration over 137 levels
  at 3,600 points takes 2 ms, against 0.75 s for a Python loop. See the new
  example, which includes the reading of GRIB files with ecCodes.

* **Geodesic distances in RTree3D**: Added the ``distance`` option to
  ``RTree3D``. With ``distance="geodesic"`` and a spheroid, the tree still
  searches its nodes with straight-line ECEF distances, but ranks the neighbors
  by geodesic distance on the spheroid and returns geodesic distances: it finds
  the same neighbors as the geographic ``RTree``, at the same distances, 5 to
  10 times faster. The default, ``"euclidean"``, keeps the previous behavior.

Dependencies
~~~~~~~~~~~~

* **Python 3.15**: wheels are built for CPython 3.15 and its free-threaded
  build 3.15t, and the CI tests Python 3.15 on Linux and Windows.
* **xarray is optional**: it is no longer installed with ``pyinterp``. Install
  ``pyinterp[xarray]`` to use ``pyinterp.backends.xarray``, the ``pyinterp``
  accessor and ``pyinterp.geohash.to_xarray``, which now say so when xarray is
  missing.
* **dask is optional**: it is no longer installed with ``pyinterp`` either.
  Install ``pyinterp[dask]`` to use the ``pyinterp.dask`` helpers, which say
  so when dask is missing. NumPy is now the only required dependency.
* **numba is optional**: install ``pyinterp[numba]`` to compile the column
  kernels of ``pyinterp.reduced_gaussian.column_kernel``.

* **pocketfft**: Updated the vendored submodule to the latest commit.

Performance
~~~~~~~~~~~

* **Multi-threading**: The points are no longer split into one fixed block
  per thread: the threads claim small contiguous chunks (at least 256 points,
  32 per thread on large inputs) as they finish the previous ones, and the
  calling thread works too. A thread given expensive points, or running on an
  efficiency core, no longer finishes alone while the others wait. On all the
  cores of an Apple M3 Max, the grid interpolations are 1.15 to 1.3 times
  faster, the R*-tree estimators 1.05 to 1.15 times and LOESS 1.08 times. The
  results are unchanged.
* **Windowed bicubic interpolation**: ``bicubic`` computes the partial
  derivatives only at the corners of the cell holding the point, instead of at
  every node of the window, and keeps them while the following points fall in
  the same cell: 3.3 times faster on random points with the default window
  (0.67 → 2.2 million points per second on one thread) and with a 4 × 4 window
  (1.2 → 4.1). The results are unchanged.
* **Other windowed interpolations**: Preparing a window no longer copies the
  data twice, the spline methods no longer allocate memory for each row of the
  window, and ``c_spline_not_a_knot`` and ``polynomial`` no longer allocate
  memory for each spline they fit. On random points, the windowed methods are
  1.2 to 3 times faster; the results are unchanged.
* **Nearest neighbors of RTree3D and RTree4D**: the Cartesian trees search the
  k nearest neighbors depth first, closest children first, into buffers
  reused by each thread, instead of running the best-first query of Boost,
  which allocates its queues for every point. The nodes hold 8 elements
  instead of 16, the queries are no longer copied into a new vector per point,
  the query points are converted in parallel and sorted along the Morton curve
  by a radix sort. With one million observations and one million points,
  ``inverse_distance_weighting`` and ``query`` are about 2 times faster on one
  thread and 2.5 times faster on all cores. The neighbors and distances are
  unchanged; only observations at exactly the same distance may come in
  another order, or another of them be kept as the last neighbor.

* **Python extension**: The extension is no longer compiled with ``-Os`` in
  release builds, which disabled the vectorization and inlining of the Eigen
  kernels.
* **Fill methods**: The LOESS window weights are computed once per thread
  (2.7 to 3.6 times faster), and Gauss-Seidel only sweeps the unknown cells.
* **Interpolation**: Windowed, bilinear and bicubic interpolations no longer
  copy the window arrays or allocate memory for each point, nearest and IDW
  interpolations share the corner distances across layers, and periodic axis
  lookups avoid integer divisions.
* **R-tree**: k-NN searches no longer use the incremental query, and the
  query points of a batch are searched in spatial order.
* **Statistics**: T-digest quantiles are computed in a single pass,
  ``Binning1D`` uses a dedicated update when no weights are given, Cartesian
  linear binning weights are computed in closed form, and strided arrays are
  iterated with incremental offsets.
* **Geometry**: Point-in-polygon tests are filtered with the envelope of the
  container.
* **Geohash**: ``geohash.encode`` and ``geohash.transform`` are faster.
  When zooming out, ``transform`` now returns each code once, in ascending
  order.
* **Date utilities**: ``datetime64`` values are split with a resolution known
  at compile time.
* **Benchmarks**: Added a ``benchmarks`` package, run with
  ``python -m benchmarks``, covering 180 cases in nine families. Results can be
  saved and compared with a previous run.

Bug Fixes
~~~~~~~~~

* **Intersections with an edge on the equator**: Boost.Geometry 1.92 and
  earlier compute a wrong longitude for the intersection of a segment with a
  segment lying on the equator, so the intersection of a line and a polygon
  with such an edge could extend beyond the polygon (#36). The conda-forge
  package patched the Boost headers, the wheels did not: the build now writes
  a corrected copy of the header when the Boost headers found still contain
  the error (boostorg/geometry#1483), so every build gets the fix.
* **conda builds on Windows**: rattler-build defines
  ``CONDA_BUILD_CROSS_COMPILATION`` for native builds too, set to ``0``, and
  CMake took any definition for a cross-compilation: it looked for the Python
  headers alone, which failed for Python 3.15 on Windows. Only ``1`` now
  selects the cross-compilation path, as in ``setup.py``. On Windows,
  ``CMAKE_ARGS`` also lost the backslashes of its paths when split; they are
  now kept.
* **Geohash of an invalid latitude**: ``geohash.encode``, ``geohash.GeoHash``
  and the integer encoder took any latitude: outside [-90, 90], or NaN, the
  conversion to the integer code was undefined behavior, which gave the code
  of a pole on arm64 and an arbitrary one on x86-64. They now raise a
  ``ValueError`` for a latitude outside [-90, 90] or NaN, and for a
  longitude that is not finite; finite longitudes are still wrapped.
* **Track decomposition across the antimeridian**:
  ``geometry.satellite.decompose_track`` took the plain minimum and maximum of
  the longitudes of a segment, so a segment crossing 180° (or Greenwich, for
  longitudes in [0°, 360°)) got the complement of the arc it covers, almost the
  whole globe. The box now follows the track: its western edge lies in
  [-180°, 180°) and its eastern edge may exceed 180°. Merged boxes and
  ``filter_by_extent`` compare longitudes modulo 360°, and measurements without
  a position are left out of the box. ``infer_orbit_direction`` takes each
  difference of longitudes as the shortest arc, so a crossing is no longer read
  as travel in the opposite direction.
* **Track decomposition across a gap in the data**: when a gap hid the polar
  pass between two passes, the ``latitude_bands`` strategy joined them into one
  segment lying half a turn of longitude apart. A segment is now also cut where
  the latitude turns back between the thresholds. The ``monotonic`` strategy
  no longer gives the turning sample to both segments around it: as documented,
  segments do not overlap, and that sample goes with the nearer one in
  longitude.
* **xarray interpolator coordinates**: ``RegularGridInterpolator`` and the
  ``Grid2D``/``Grid3D``/``Grid4D`` methods of ``pyinterp.backends.xarray``
  rejected scalars and lists, although their documentation used them; only
  one-dimensional arrays were accepted. The coordinates can now be scalars,
  lists or arrays of any shape, broadcast against each other, and the result
  has their broadcast shape. Dates can also be given as strings.
  ``RegularGridInterpolator`` is now listed in the ``__all__`` of the module.
* **Bilinear with windowed options**: ``bilinear`` names both a geometric and a
  windowed method, and the string always selected the geometric one, so
  combining it with ``third_axis``, ``fourth_axis``, ``half_window_size_x``,
  ``half_window_size_y`` or ``boundary_mode`` raised a ``TypeError``. These
  options now select the windowed bilinear interpolator.
* **Dask helpers**: ``pyinterp.dask.binning1d``, ``binning2d`` and
  ``histogram2d`` rebuilt the container of each chunk from the axes of the
  template, so a ``float32`` template gave a ``float64`` result and the
  compression of a ``Histogram2D`` was ignored. Each chunk now starts from an
  empty copy of the template, which keeps its data type, range, spheroid and
  compression.
* **GeoHash to xarray**: ``pyinterp.geohash.to_xarray`` swapped the units of the
  coordinates: the longitude was labelled ``degrees_north`` and the latitude
  ``degrees_east``.
* **Polynomial interpolation**: Fixed the ``polynomial`` univariate method,
  which returned wrong values when its window held more than four points. The
  divided differences were computed in place and read values that had already
  been overwritten. This defect was introduced in 2026.2.0.
* **Linear binning on grid nodes**: ``Binning2D`` with linear binning in a
  Cartesian plane (no spheroid) could ignore a value located exactly on a grid
  node. When the node was the upper corner of its cell, which is always the case
  on the last row and column and also happens inside grids whose step is not
  exactly representable (e.g. 0.1), the area of the sub-cell covering the whole
  cell came out slightly larger than the cell because of rounding errors and
  was discarded, so the value got no weight at all. Such an area is now clamped
  to the area of the cell.
* **Multigrid convergence**: Fixed ``fill.multigrid``, whose coarse-grid
  correction did not match the fine problem: each V-cycle removed only about
  15 % of the error, and large grids did not converge. The restriction now uses
  9-point full weighting, and the number of cycles no longer grows with the
  grid size: a 720 × 360 grid converges in 12 cycles (2.8 s → 182 ms).
* **Periodic LOESS**: ``fill.loess`` ignored ``is_periodic``: the neighbors
  across the seam got a zero weight. Results only change at the seam and at the
  reflected edges.
* **T-digest**: The size of ``TDigest`` is now bounded by its compression.
  Below the median, the centroids never merged, so the digest grew with the
  number of values, and computing it over a whole array was quadratic
  (470 ms → 6 ms for 81,000 values). The quantiles of weighted data, which were
  computed from the number of samples instead of the total weight, are fixed,
  as is the merge of two digests, which dropped the values not yet merged.
* **Pickled RTree3D**: Pickling ``RTree3D`` now preserves its coordinate system
  and distance: a Cartesian tree was restored as a geodetic one, with the WGS84
  spheroid. States saved by earlier versions are still accepted.
* **EquatorCoordinates**: ``orbit.EquatorCoordinates.undefined`` creates its
  ``NaT`` with an explicit unit, as the generic unit is deprecated by NumPy.

Build System
~~~~~~~~~~~~

* **Release workflow**: The source distribution is built and published by the
  workflow that publishes the wheels, which waits for both before publishing
  anything. Two workflows used to publish them, so PyPI reported the
  provenance of the 2026.10.0 files as inconsistent, and the source
  distribution could be published without the wheels.
* **Precompiled header**: The Python extension is compiled with a precompiled
  header holding the Boost.Geometry, Eigen and nanobind headers, which accounted
  for about half of the compilation time. It can be disabled with the
  ``PYINTERP_USE_PCH`` CMake option.
* **Geographic overlay operations**: The bindings of the geographic
  ``difference``, ``intersection`` and ``union`` are instantiated in one
  translation unit per strategy, so that the slowest files of the build are
  compiled in parallel.
* **Continuous integration**: Compilations are cached with ccache in the
  coverage, Linux and macOS jobs, and the timeout of the Linux jobs is raised to
  30 minutes.
* **Missing include**: Added the nanobind string header missing from
  the geohash bindings (``cxx/src/pybind/geohash/string.cpp``).
* **build_ext options**: ``--build-unittests`` now builds the C++ unit tests
  together with the extension; it was accepted but had no effect. The
  ``c-compiler`` option of ``pip install -C`` is removed: the project compiles
  no C code, and ``setup.py`` rejected it, so the build failed.
* **Minimum Python version**: The package metadata requires Python 3.12, the
  oldest version supported since Python 3.11 was dropped; it still accepted
  3.11, for which no wheel is built.
* **Wheels**: The wheels are built with cibuildwheel 4.3, so the CPython 3.14
  wheels are compiled against the final 3.14 release instead of a pre-release.
  The installer of Miniforge, which provides Boost to the Linux builds, is
  pinned to a release and checked against its SHA-256.
* **CMake policy CMP0144**: A typo kept the policy unset, so ``find_package``
  ignored the upper-case ``<PACKAGENAME>_ROOT`` variables such as
  ``BOOST_ROOT``.

Documentation
~~~~~~~~~~~~~

* **RTree3D**: Documented the geodesic mode of ``RTree3D`` and its differences
  with the geographic ``RTree``, which evaluates geodesic distances to its nodes
  and queries 10 to 20 times slower than ``RTree3D`` with a spheroid.
* **API reference**: Added pages for the configuration objects
  (``pyinterp.core.config``), ``dateutils`` and ``period``, the missing
  geometry and satellite functions, and an introduction to each page. Fixed
  docstring examples that used wrong parameter names, and the Sphinx warnings
  on ``OIResult`` and on ambiguous return types.
* **Examples and guides**: The two examples of the "core" section (axes and
  grids) were missing from the gallery. Fixed the text of the examples that
  contradicted their code or the current API, made ``ex_dateutils``
  reproducible, and corrected the migration guide (bicubic, TDigest, fill,
  Loess and satellite sections) and old changelog entries.
* **New examples**: Added five examples to the gallery: interpolating an
  xarray ``DataArray`` with ``RegularGridInterpolator``, the distance modes of
  ``RTree3D`` (Cartesian, chord and geodesic), binning on the sphere with a
  spheroid, robust per-bin statistics with ``Histogram2D`` and ``TDigest``, and
  statistics on chunked arrays with the ``pyinterp.dask`` helpers.
* **Building from source**: Rewrote the build guide. It now starts with a quick
  start based on the conda environments of the CI, lists the current
  requirements (C++23, Boost 1.90), documents every ``build_ext`` option, CMake
  option and environment variable, and adds sections on speeding up the build
  (precompiled header, Ninja, ccache), on static analysis with clang-tidy and
  pre-commit, and on building the documentation. The obsolete coverage
  commands are replaced by ``scripts/generate_coverage.sh``.
* **Benchmarks**: A new page compares pyinterp with SciPy, xarray and xESMF
  for bilinear interpolation at scattered points, regridding, cubic
  interpolation and inverse distance weighting, after describing what the
  libraries do differently (periodic longitudes, undefined values). The
  timings are reproduced with ``python -m benchmarks.compare``.
* **README**: Rewritten for new users, with the installation, a quick start
  and a comparison with xarray and SciPy. Its notes on the axes, the R*-trees
  and the coordinate conversions moved to a new Design Notes page.

2026.10.0
---------

Bug Fixes
~~~~~~~~~

* **Crossover nearest vertex indices**
  (`#41 <https://github.com/CNES/pangeo-pyinterp/issues/41>`_): Fixed
  ``geometry.satellite.find_crossovers``, which could return ``index1`` and
  ``index2`` values that were not those of the vertices nearest to the
  crossover point. The golden-section search used to locate them worked on
  truncated integer indices: when its two probes collapsed onto the same
  vertex, the half of the interval holding the nearest vertex was discarded.
  The indices could be off by several vertices and, with a tight
  ``predicate``, a valid crossover could be rejected. The crossover location
  itself was not affected. All the vertices are now examined by default, as
  ``geodetic.calculate_crossover`` did in 2025.11.0.
* **Geohash ``where``**
  (`#40 <https://github.com/CNES/pangeo-pyinterp/issues/40>`_): Fixed three
  defects of ``geohash.where``:

  * The codes were read with a length equal to the number of columns of the
    array instead of the size of its items, so the keys returned were
    truncated or straddled neighboring codes, and arrays with more than 12
    columns were rejected. This regression was introduced in 2026.2.0.
  * A code occurring exactly twice was reported with the bounds of its first
    occurrence only. The bounds now cover all the occurrences of a code,
    whether they are adjacent or not; isolated occurrences were previously
    ignored.
  * The function is exported again from ``pyinterp.geohash``; since 2026.2.0
    it was only reachable as ``pyinterp.core.geohash.where``.

  Arrays that are not C-contiguous, such as transposed arrays, are now
  accepted.

New Features
~~~~~~~~~~~~

* **Faster nearest vertex search for crossovers**: Added the
  ``assume_unimodal`` option to ``geometry.satellite.find_crossovers``. When
  set, the nearest vertices are located with a bisection search instead of an
  exhaustive scan, which makes the whole call about 8 % faster on the Cartesian
  plane and about 20 % faster with geodetic calculations. It requires the
  distance to the crossover point to be strictly unimodal along each track,
  which holds for clean half-orbits but not for tracks holding duplicated
  vertices or covering more than a half-orbit.


2026.7.0
--------

This release overhauls version resolution across the build, refreshes the
vendored dependencies, and improves the documentation around Dask-backed
binning.

Build System
~~~~~~~~~~~~

* **Unified version resolution**: Added a CMake-based version resolution
  mechanism (``cmake/PyinterpVersion.cmake``) that derives the project version
  consistently for both the Python and C++ components. A ``VERSION.txt`` file
  (expandable by ``git archive``) and a companion ``scripts/resolve_version.py``
  mirror the same logic in Python, keeping ``setup.py`` and ``setuptools_scm``
  in sync. ``CMakeLists.txt`` now conditionally builds the Python bindings and
  tests. Stale conda ``meta.yaml``/``conda_build_config.yaml`` files were
  removed and the CI environment configurations moved under
  ``.github/environments``.
* **Include directories**: Dropped the ``PRIVATE`` keyword from
  ``include_directories`` in ``CMakeLists.txt`` for broader CMake
  compatibility.

Dependencies
~~~~~~~~~~~~

* **nanobind**: Updated the vendored submodule to v2.13.0.
* **pocketfft**: Updated the vendored submodule to the latest commit.

Examples & Documentation
~~~~~~~~~~~~~~~~~~~~~~~~

* Expanded the Dask usage notes in the binning example
  (``examples/stats/ex_binning.py``) with additional statistics, and updated
  the migration guide to reflect the changes to the ``Binning`` and Dask
  integration APIs.
* Geometry examples now use the ``constrained`` figure layout for clearer
  visualizations.


2026.6.0
--------

Bug Fixes
~~~~~~~~~

* **Periodic axis interpolation across the 0/360 seam**: Fixed the geometric
  interpolators (``bivariate``, ``trivariate`` and ``quadrivariate`` with the
  ``bilinear``, ``nearest`` and ``idw`` methods) so that points falling in the
  wrap-around cell of a periodic axis are interpolated correctly. The upper
  bracketing node was read with its raw coordinate (e.g. ``0`` instead of
  ``360``), collapsing the interpolation weight across the seam: values stayed
  pinned to the lower node and then jumped discontinuously at the period
  boundary. The upper node coordinate is now normalised relative to the lower
  node, restoring a smooth, monotonic interpolation across the boundary. The
  windowed interpolators (``bicubic`` and the spline family) were unaffected.


2026.4.0
--------

This release adds satellite ground-track decomposition, ships official PyPI
wheels for the first time, and refreshes the supported Python matrix.

New Features
~~~~~~~~~~~~

* **Satellite Track Decomposition**: Added decomposition algorithms in
  ``geometry.satellite`` that segment a satellite ground track into ascending
  and descending passes, with new ``DecompositionOptions`` and related enums to
  control the algorithm. An accompanying example
  (``examples/orbit/ex_decompose_track.py``) demonstrates typical usage.
* **Pairwise Distances**: Added the ``for_each_point_pairwise_distance``
  algorithm to compute pairwise distances between points in both the
  ``geometry.cartesian`` and ``geometry.geographic`` algorithm submodules.

Examples & Documentation
~~~~~~~~~~~~~~~~~~~~~~~~

* 2D and 3D map examples now use ``set_global()`` instead of ``set_extent()``
  for clearer global visualizations.

Packaging & Distribution
~~~~~~~~~~~~~~~~~~~~~~~~

* **PyPI Wheels**: pyinterp is now distributed as binary wheels on PyPI in
  addition to conda-forge. Linux ``x86_64``/``aarch64`` (manylinux_2_28),
  macOS ``arm64`` (deployment target 13.4), and Windows ``x86_64`` are built
  via cibuildwheel.
* **Free-Threaded Python**: A free-threaded CPython 3.14 wheel
  (``cp314t``) is now produced alongside the GIL-enabled build.
* **Cross-Compilation Support**: ``setup.py`` and the CMake configuration
  honour a host Python executable and propagate cross-compilation hints,
  enabling reliable builds in restricted toolchain environments
  (e.g. conda-build sysroots).

Supported Python Versions
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Dropped**: Python 3.11.
* **Supported**: CPython 3.12, 3.13, 3.14, and 3.14 free-threaded.

Build System
~~~~~~~~~~~~

* **C++ Standard Library**: Replaced ``std::jthread`` with ``std::thread`` plus
  explicit ``join()`` in the parallel-for utility for compatibility with
  Apple libc++ availability gating.
* **Boost ≥ 1.90**: CMake now enforces a minimum Boost version of 1.90 with
  improved detection logic and clearer error reporting.
* **CMake Minimum Version**: Bumped CMake minimum required version and updated
  Python target handling for nanobind compatibility.
* **Compiler Checks**: Added an explicit IEC 559 (IEEE 754) compliance check
  and additional cross-compilation-aware compiler probes.
* **Windows**: Worked around a ``constexpr``/MSVC issue in the ``area_km2``
  helper using the ``__CONSTEXPR`` macro.
* **Packaging**: Added ``MANIFEST.in`` and ensured ``pyinterp/_version.py``
  is included in source distributions.

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~

* Reworked the release workflows to build and publish wheels via
  ``cibuildwheel`` and the source distribution via ``pypa/build``.
* Pre-release tags (``X.Y.Z<suffix>``) are routed to TestPyPI; plain
  ``X.Y.Z`` tags publish to PyPI. Both use OIDC trusted publishing.

2026.2.0
--------

This release represents a **major architectural rewrite** of pyinterp with significant
breaking changes. Please refer to the :doc:`migration` guide for detailed upgrade
instructions.

Architecture & Framework
~~~~~~~~~~~~~~~~~~~~~~~~

* **Binding Framework Migration**: Migrated from pybind11 to nanobind for Python
  bindings, providing improved performance and reduced binary size.
* **C++ Directory Restructuring**: Reorganized C++ source code with new directory
  structure (``cxx/src/pybind/`` for bindings, ``cxx/src/library/`` for core
  implementations).
* **Namespace Reorganization**: Refactored C++ namespaces from ``pyinterp::detail::*``
  to organized modules: ``pyinterp::math::``, ``pyinterp::geometry::``,
  ``pyinterp::config::``, and ``pyinterp::pybind::``.
* **Modern C++ Features**: Adopted C++20/C++23 features including concepts,
  ``constexpr``, ``std::format``, ``std::ranges``, three-way comparison operator
  (``<=>``), and ``[[nodiscard]]`` attributes.

New Configuration System
~~~~~~~~~~~~~~~~~~~~~~~~

* **Type-Safe Configuration Objects**: Introduced a new ``config`` module with
  builder-pattern configuration objects. The public ``pyinterp.fill`` and
  ``pyinterp.rtree`` functions accept either a configuration object or the
  equivalent keyword arguments.
* **Configuration Submodules**: Added ``config.fill``, ``config.geometric``,
  ``config.rtree``, and ``config.windowed`` for domain-specific configurations.
* **Fluent API**: Configuration objects support method chaining with
  ``.with_max_iterations()``, ``.with_epsilon()``, ``.with_num_threads()``, etc.

Geometry Module
~~~~~~~~~~~~~~~

* **Module Restructuring**: Replaced the ``geodetic`` module with a unified
  ``geometry`` module supporting multiple coordinate systems.
* **Geographic Coordinates**: The ``geometry.geographic`` submodule replaces the
  former ``geodetic`` module with all geodetic operations.
* **Cartesian Coordinates**: Added new ``geometry.cartesian`` submodule for
  Cartesian coordinate system operations.
* **Satellite Operations**: Added new ``geometry.satellite`` submodule with
  satellite-specific algorithms (swath calculation, crossover detection).
* **Algorithm Submodules**: Geometric algorithms are now organized in
  ``algorithms`` submodules for each coordinate system (40+ algorithms available).

Math & Interpolation
~~~~~~~~~~~~~~~~~~~~

* **Math Module Reorganization**: New ``math`` namespace with organized submodules
  for interpolation, statistics, FFT, and DCT operations.
* **Interpolation Architecture**: Redesigned interpolation system with clear
  separation into univariate, bivariate, and geometric interpolation.
* **Interpolation Caching**: Added caching layer for improved interpolation
  performance.
* **Window Functions**: Added window function support for localized interpolation.

New Features
~~~~~~~~~~~~

* **T-Digest**: Added ``TDigest`` class for streaming quantile estimation with
  typed variants (``TDigestFloat32``, ``TDigestFloat64``).
* **Dask Integration**: Added ``dask`` module with distributed computation support
  for ``binning1d()``, ``binning2d()``, ``descriptive_statistics()``,
  ``histogram2d()``, and ``tdigest()``.
* **Regular Grid Interpolator**: New ``regular_grid_interpolator`` module with
  string-based method selection (``"bilinear"``, ``"bicubic"``, ``"nearest"``,
  ``"akima"``, etc.) and configuration object support.
* **Typed Class Variants**: Added Float32/Float64 typed variants for ``Binning1D``,
  ``Binning2D``, ``Histogram2D``, ``RTree3D``, and ``TDigest``.
* **Enhanced Pickle Support**: Added proper pickle serialization for ``Grid``
  objects via ``GridHolder``.
* **Parallel Processing**: Added parallel-for utilities for multi-threaded
  operations.

Xarray Backend
~~~~~~~~~~~~~~

* **Simplified API**: The ``backends.xarray`` module has been significantly
  simplified with automatic detection of geodetic and temporal axes using CF
  conventions.
* **Automatic Axis Detection**: Longitude/latitude axes are automatically detected
  from CF-compliant ``units`` attributes; temporal axes are detected from
  ``datetime64`` dtypes.
* **Removed Parameters**: The ``increasing_axes`` and ``geodetic`` parameters have
  been removed from ``RegularGridInterpolator``; axis detection is now automatic.
* **Unified Method Parameter**: The ``bicubic()`` method is removed; use the
  ``method="bicubic"`` parameter in the interpolation call instead.
* **Simplified Keyword Arguments**: The ``bicubic_kwargs`` parameter is removed;
  bicubic options are now passed directly as keyword arguments.

Breaking Changes - Removed Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following Python modules have been removed and their functionality relocated:

* ``pyinterp.geodetic`` → Use ``pyinterp.geometry.geographic``
* ``pyinterp.statistics`` → Use ``pyinterp.DescriptiveStatistics``
* ``pyinterp.interpolator`` → Use ``pyinterp.regular_grid_interpolator``
* ``pyinterp.grid`` → Use ``pyinterp.Grid`` (with aliases ``Grid2D``, ``Grid3D``,
  ``Grid4D``)
* ``pyinterp.histogram2d`` → Use ``pyinterp.Histogram2D``
* ``pyinterp.period`` → Moved to the compiled core (``pyinterp.core.period``),
  still available as ``from pyinterp import period``
* ``pyinterp.binning`` → Use ``pyinterp.Binning1D``, ``pyinterp.Binning2D``
* ``pyinterp.interface`` → Removed

Breaking Changes - Removed Classes & Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``AxisInt64`` → Use ``TemporalAxis`` for datetime axes
* ``RadialBasisFunction`` enum → Use ``config.rtree.RBFKernel``
* ``WindowFunction`` enum → Use ``config.rtree.WindowKernel``
* ``GeoHash`` (top-level) → Use ``pyinterp.geohash.GeoHash``
* Orbit classes (``EquatorCoordinates``, ``Orbit``, ``Pass``, ``Swath``) and
  functions (``calculate_orbit``, ``calculate_pass``, ``calculate_swath``) → No
  longer re-exported from the top-level ``pyinterp`` namespace; import them from
  ``pyinterp.orbit``
* ``bicubic()`` function → Use ``regular_grid_interpolator.bivariate()``
* ``interpolate1d()`` function → Use ``regular_grid_interpolator.univariate()``
* ``__date__`` attribute → Removed (``__version__`` is kept)

Breaking Changes - API Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **orbit.interpolate()**: Parameter ``wgs`` renamed to ``coordinates``, default
  ``half_window_size`` changed from 10 to 3.
* **Axis class**: Parameter ``is_circle`` replaced with ``period``, property
  ``is_circle`` renamed to ``is_periodic``.
* **Enum Values**: All enum values now use UPPERCASE (e.g., ``FirstGuess.ZERO``,
  ``RBFKernel.GAUSSIAN``).
* **Fill Functions**: ``pyinterp.fill`` functions operate on NumPy arrays instead
  of ``Grid`` objects; options are passed as keyword arguments or as a
  ``config.fill`` object.
* **RTree Methods**: Interpolation is done by the ``pyinterp.rtree`` functions
  (or the ``RTree3D`` methods), with options passed as keyword arguments or as a
  ``config.rtree`` object.

Build System & Tooling
~~~~~~~~~~~~~~~~~~~~~~

* **Platform-Specific Conda Files**: Added separate conda environment files for
  each platform (``conda/linux.yml``, ``conda/win.yml``, ``conda/osx.yml``).
* **Updated CI Workflows**: Enhanced GitHub Actions workflows for testing and
  coverage.
* **Test Suite Reorganization**: Tests reorganized into categorical subdirectories
  (``tests/config/``, ``tests/geometry/``, ``tests/math/``, etc.).

2025.11.0
---------
* Deprecated the ``universal_kriging`` method in favor of the more versatile
  ``kriging`` method, which now supports both simple and universal kriging.
  Updated the documentation accordingly.
* Refactored the RBF class to improve epsilon handling, enhance interpolation
  matrix computation, and optimize the distance calculation method.
* Enabled builds on Linux ARM64 architectures.
* Replaced the Olson, D.K. algorithm with the Vermeille, H. algorithm for
  geodetic to ECEF conversions, improving numerical stability and accuracy.
* Optimized the "spline not a knot" fitting model by using Eigen::SparseLU for
  solving linear systems, resulting in better performance.
* Adopted ruff for code formatting and linting, and improved code typing with
  more precise type hints and annotations.
* Updated pre-commit hooks to their latest versions in
  ``.pre-commit-config.yaml``.
* Upgraded the Eigen library to v5.0.0 to gain performance
  improvements and new features.
* Added Eigen as a git submodule to ensure consistent builds across
  different environments.
* Reorganized the source directory structure to clearly separate C++ and Python
  code, improving maintainability.

2025.8.1
--------
* [Fix] Refactored BMI2-related functions to use _pdep_u64 and _pext_u64
  directly, removed custom assembly implementations, fixed a typo in have_bim2
  (renamed to have_bmi2), and added missing include headers for better
  compatibility and maintainability.

2025.8.0
--------
* Update subproject commit for pybind11 to v3.0.0
* Update copyright year in LICENSE and specify license type in setup.py
* Refactor nearest function to compare absolute distances directly, improving
  clarity and consistency in logic (#32).

2025.3.0
--------

* Introduced a new fitting model, "c_spline_not_a_knot", for spline interpolation.
* Upgraded the C++ standard to C++20.
* Updated the pre-commit hooks in ``.pre-commit-config.yaml`` to their latest versions.
* Increased the minimum required CMake version to 3.10.
* Addressed tidy warnings.

2024.11.0
---------

* `#27 <https://github.com/CNES/pangeo-pyinterp/issues/27>`_: Introduced a new
  feature to the RTree object to handle Cartesian coordinates, bypassing the
  LLA/ECEF conversion .
* Introduced GitHub Actions CI workflow for coverage and testing across Linux,
  macOS, and Windows.
* Added new dependencies to ``conda/environment.yml``: boost-cpp, cmake,
  coverage, dask, eigen, gtest, lcov, pytest-cov, setuptools, and xarray.
* Added CMake policy CMP0167 in ``CMakeLists.txt``.
* Updated pre-commit hooks to newer versions in ``.pre-commit-config.yaml``.
* Modified ``CMakeLists.txt`` to adjust BLAS libraries configuration.
* Updated ``README.rst`` badges to use GitHub Actions instead of Azure DevOps.
* Renamed conda environment from RTD to pyinterp in ``conda/environment.yml``.
* Deleted Azure Pipelines configuration (``azure-pipelines.yml``).
* Removed docformatter pre-commit hook from ``.pre-commit-config.yaml``.

2024.6.0
--------
* Fix compatibility with Numpy 2.0.0
* Minor bug fixes.

2024.3.0
--------
* Enhance the performance of the spline/bicubic interpolation:

  - The bicubic interpolation is now 40% faster.
  - The cubic spline interpolation is now 30% faster.

2024.2.0
--------
* Fix equator point selection in orbit.py
* Fix skewing of first pass in calculate_orbit() function
* Add support for Grid2DUInt8 interpolation
* Swap min_corner and max_corner in geodetic box constructor if necessary

2024.1.0
--------
* Fix issues in the test suite.
* Fix linter warnings.

2023.11.0
---------
* Add intersection method to LineString class
* RegularGridInterpolator does not handle interpolator keywords.

2023.10.2
---------
* Fix issues with Python 3.12

2023.10.1
---------
* Fill time series by linear interpolation.
* Fill unknown positions by linear interpolation.

2023.10.0
---------
* Period list management.
* Documentation fixes.
* Geohash box must be within [-180, 180[.
* Fix a bug in the calculation of the orbit from the ephemeris.

2023.5.0
--------
* If MKL is not wanted, use the default BLAS implementation available. On MacOS
  this is Accelerate. On Linux this is OpenBLAS or a Generic BLAS implementation.

2023.2.1
--------
* Cleanup module dependencies.
* If MKL is not found, use the default BLAS implementation if available.

2023.2.0
--------
* Looking for multiple crossovers.
* Calculate intersection between polygon and linestring.
* Added few utility methods for polygons.
* Calculate the closest point on a linestring.
* Added Universal Krige interpolation for unstructured grids.

2023.1.0
--------
* Add a method to simplify a line string.
* Minor fixes.
* Bug fixes with Python and the Global Interpreters Lock.
* Update pybind11 to v2.10.3

2022.10.1
---------
* Add an option to calculate the 1D binning to limit the upper and lower range
  of bins used.
* Add missing prerequisites for pip install.
* Documentation fixes.

2022.10.0
---------
* `#13 <https://github.com/CNES/pangeo-pyinterp/issues/13>`_: Extract coordinates of the K nearest neighbors of a point.
* `#12 <https://github.com/CNES/pangeo-pyinterp/issues/12>`_: Only the X-axis can be circular.
* Pure geodetic RTree.
* Bug fixes.
* Documentation fixes.

2022.9.1
--------
* Orbit interpolation improvement.

2022.9.0
--------
* Improves the accuracy of swath calculation.
* Improves TemporalAxis performance.
* Fixes bugs in the documentation.
* Python interface files are not installed properly.

0.16.0 (4 August 2022)
----------------------
* Using Red-Black Gauss-Seidel.
* Refactor loess functions for 3D/4D grids.
* Bicubic interpolation failed on singleton axis (#11)
* Fill Grid4D with Loess filter (#10)

0.15.2 (5 July 2022)
--------------------
* Disable assembly code on OSX/ARM64.
* Publish AxisInt64.
* Crossover point calculation is not accessible.

0.15.1 (6 June 2022)
--------------------
* Calculate crossovers in a Cartesian plane.
* Fix a bug in the documentation.

0.15.0 (5 June 2022)
--------------------
* Geodetic RTree
* Added support for PyPy and Pyston.
* Removal of NetCDF4 dependency.
* Orbit interpolation.
* Intersection search using WGS/Spherical coordinates.

0.14.0 (12 April 2022)
----------------------
* Add a Gaussian filter function to window functions.
* Select GeoHash contained in a multi-polygon.
* Construct Geodetic objects from GeoJSON.
* Addition of geodetic algorithms.
* Implement the shallow copy operation.
* Fixes lint/compiler warnings.
* Refactor RTree class.
* Refactor Crossovers/Linestrings class.
* Update the building system.

0.13.0 (11 March 2022)
----------------------
* One-dimensional binning.
* Update pybind11 to v2.9.1
* Fixes minor issues and documentation.
* Refactoring the API documentation.
* Automatic standardization of longitude when encoding GeoHash.
* Fixes lint warnings.

0.12.0 (2 February 2022)
------------------------
* Extract test data into auxiliary files.
* Install pre-commit hooks.
* Upload automatically the package on pypi.
* Calculate crossovers between line string.


0.11.0 (3 January 2022)
-----------------------
* Change of the text representations of the grids.
* Change of the text representations of the axes.
* Display of dates in case of error in the time axes.
* Bug fix: If the step is negative in an axis, it's impossible to detect that
  the values are evenly spaced.

0.10.0 (17 December 2021)
-------------------------
* Refactor regular grid interpolators.
* Enhanced test coverage.
* Using structured binding declaration.
* Refactor GeoHash module
* Deleting GeoHash indexes and storage (we're using pyarrow for this now)

0.9.4 (1 December 2021)
------------------------
* Correction of a spelling mistake for a keyword.

0.9.3 (22 November 2021)
------------------------
* Compilation fails on Windows with Python 3.10
* Warnings from matplotlib are removed.
* Use ravel instead of flatten for numpy arrays.

0.9.2 (11 November 2021)
------------------------
* Add boxcar windowed function.
* Add support for Python 3.10
* Use ravel instead of flatten for numpy arrays.

0.9.1 (30 October 2021)
-----------------------
* Add arguments for windowed functions.

0.9.0 (22 October 2021)
-----------------------
* Smoothing of an unstructured interpolated mesh with a window function.

0.8.1 (3 October 2021)
----------------------
* Handling interpolation of a mask.
* Generates stubs for the core modules.

0.8.0 (30 August 2021)
----------------------
* Calculates univariate statistics.
* Binning using streaming histogram to evaluate quantiles.
* Refactoring of the tests to include them in the distribution.
* Using unordered_map instead of maps to optimize geohash queries.
* Store the geohash index on fs mapping.
* Timedelta64 is also temporal axes.

0.7.1 (25 April 2021)
---------------------
* find_index does not handle circles.

0.7.0 (23 April 2021)
---------------------
* Within functionality flipped for IDW.
* Checks if the coordinates are covered by a polygon.
* Calculates distance between geodetic objects.

0.6.1 (6 March 2021)
--------------------
* An axis can be constructed from constant values.

0.6.0 (5 March 2021)
--------------------
* Added utilities for numpy dates.
* Modification of the documentation.

0.5.1 (24 January 2021)
-------------------------
* Fixed a bug in TemporalAxis.

0.5.0 (1 January 2021)
-------------------------
* Bicubic interpolation.
* Corrections of various problems.
* Modification of the geohash interface.
* Optimization of temporal axes.

0.4.0 (22 September 2020)
-------------------------
* Indexing data with geohash.
* Parallelize the calculation of Binning with dask.

0.3.2 (13 May 2020)
-------------------------
* Release GIL during RTree insertion or packing.

0.3.1 (17 April 2020)
-------------------------
* Fixed a bug in Loess filter.

0.3.0 (15 April 2020)
-------------------------
* Selection of interpolation methods for the third and fourth axes.
* Fixed a bug on 4D interpolation with the xarray backend.

0.2.0 (16 February 2020)
-------------------------
* Publish the "Axis.find_indexes" method.
* The Loess filter can process both undefined and defined data.
* Updating documentation.

0.1.1 (17 January 2020)
-------------------------
* Use of MKL in the Conda package.

0.1.0 (4 January 2020)
-------------------------
* Simplification of interpolations with Xarray.
* Support for numpy datetime64.
* Added 4D interpolation.

0.0.8 (7 December 2019)
-------------------------
* Added interpolation by RBF in RTree.
* Improvement of RTree class performance.

0.0.7 (13 November 2019)
-------------------------
* Addition of binned bivariate statistics.
* Addition of bicubic interpolation in 3D space.
* Improved generations of the C++ extension.
* Analysis of test coverage.
* Minor bug fixes.

0.0.6 (4 October 2019)
-------------------------
* Improvement of axis performance.
* Fixed performance problems with bilinear, bicubic and trivariate
  interpolations.
* Handling of time axes in the xarray backend.
* Access to main objects from the main module.

0.0.5 (19 September 2019)
-------------------------
* Use of the conda-forge channel.

0.0.4 (16 September 2019)
-------------------------
* Simplification of the Xarray backend.
* Merging of the conda-forge recipe.
* Fix documentation issues.

0.0.3 (29 July, 2019)
---------------------
* Optimization of memory management.
* Improving bicubic interpolation performance.
* Addition of methods to fill undefined values to solve interpolation problems
  near the coasts.
* Major redesign to separate grid management from interpolation routines.

0.0.2 (12 July, 2019)
---------------------
* Handle bound error on structured grid interpolation.

0.0.1 (8 July, 2019)
--------------------
* Initial release.

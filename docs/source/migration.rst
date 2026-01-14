Migration Guide
###############

This guide helps users migrate from older versions of pyinterp to the current API.

.. note::

   Version 2026.1.0 is a major architectural rewrite with significant breaking
   changes. This guide covers all the API changes you need to know to migrate
   your code.

Removed Python Modules
======================

The following Python modules have been removed and their functionality relocated.
Most classes are now available directly from the ``pyinterp`` namespace:

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Old Module
     - New Location
     - Notes
   * - ``pyinterp.geodetic``
     - ``pyinterp.geometry.geographic``
     - Module restructured
   * - ``pyinterp.statistics``
     - ``pyinterp.DescriptiveStatistics``
     - Class moved to top-level
   * - ``pyinterp.interpolator``
     - ``pyinterp.regular_grid_interpolator``
     - New module with improved API
   * - ``pyinterp.grid``
     - ``pyinterp.Grid``
     - Single generic class
   * - ``pyinterp.histogram2d``
     - ``pyinterp.Histogram2D``
     - Class moved to top-level
   * - ``pyinterp.binning``
     - ``pyinterp.Binning1D``, ``pyinterp.Binning2D``
     - Classes moved to top-level
   * - ``pyinterp.interface``
     - (removed)
     - No longer needed

**Example Migration:**

Before (Old API)::

    from pyinterp.grid import Grid2D
    from pyinterp.histogram2d import Histogram2D
    from pyinterp.binning import Binning2D
    from pyinterp.geodetic import Coordinates

After (New API)::

    from pyinterp import Grid2D, Histogram2D, Binning2D
    from pyinterp.geometry.geographic import Coordinates

Typed Class Variants
====================

Several classes now provide typed variants for explicit float precision control:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Base Class
     - Float32 Variant
     - Float64 Variant
   * - ``Binning1D``
     - ``Binning1DFloat32``
     - ``Binning1DFloat64``
   * - ``Binning2D``
     - ``Binning2DFloat32``
     - ``Binning2DFloat64``
   * - ``Histogram2D``
     - ``Histogram2DFloat32``
     - ``Histogram2DFloat64``
   * - ``RTree3D``
     - ``RTree3DFloat32``
     - ``RTree3DFloat64``
   * - ``TDigest``
     - ``TDigestFloat32``
     - ``TDigestFloat64``

The base class names (without suffix) default to Float64 precision.

Grid Class Changes
==================

The ``Grid`` class is now a generic class with dimension-specific aliases:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Old Class
     - New Class
     - Notes
   * - ``grid.Grid2D``
     - ``pyinterp.Grid2D``
     - Alias to ``Grid`` (2D)
   * - ``grid.Grid3D``
     - ``pyinterp.Grid3D``
     - Alias to ``Grid`` (3D)
   * - ``grid.Grid4D``
     - ``pyinterp.Grid4D``
     - Alias to ``Grid`` (4D)

All grid classes are now available directly from ``pyinterp``.

Regular Grid Interpolator
=========================

The interpolation API has been redesigned with a new string-based method
selection and configuration object support:

**Old API (removed):**

- ``pyinterp.bicubic()``
- ``pyinterp.interpolate1d()``
- ``pyinterp.interpolator.bivariate()``
- ``pyinterp.interpolator.trivariate()``

**New API:**

- ``pyinterp.regular_grid_interpolator.univariate()``
- ``pyinterp.regular_grid_interpolator.univariate_derivative()``
- ``pyinterp.regular_grid_interpolator.bivariate()``
- ``pyinterp.regular_grid_interpolator.trivariate()``
- ``pyinterp.regular_grid_interpolator.quadrivariate()``

**Available interpolation methods (string-based):**

- ``"linear"`` - Linear interpolation
- ``"nearest"`` - Nearest neighbor
- ``"akima"`` - Akima spline
- ``"bilinear"`` - Bilinear (2D)
- ``"bicubic"`` - Bicubic (2D)
- ``"c_spline"`` - Cubic spline
- ``"c_spline_periodic"`` - Cubic spline (periodic)
- ``"c_spline_not_a_knot"`` - Cubic spline (not-a-knot)
- ``"polynomial"`` - Polynomial
- ``"steffen"`` - Steffen spline

**Example Migration:**

Before (Old API)::

    from pyinterp import bicubic

    result = bicubic(grid, x, y, nx=3, ny=3)

After (New API)::

    from pyinterp import regular_grid_interpolator

    result = regular_grid_interpolator.bivariate(grid, x, y, method="bicubic")

    # Or with configuration object for advanced options:
    from pyinterp.config.geometric import Bivariate

    config = Bivariate().with_window_size(3, 3)
    result = regular_grid_interpolator.bivariate(grid, x, y, config=config)

Dask Integration
================

A new ``dask`` module provides distributed computation support:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``pyinterp.dask.binning1d()``
     - Distributed 1D binning
   * - ``pyinterp.dask.binning2d()``
     - Distributed 2D binning
   * - ``pyinterp.dask.descriptive_statistics()``
     - Distributed statistics computation
   * - ``pyinterp.dask.histogram2d()``
     - Distributed 2D histogram
   * - ``pyinterp.dask.tdigest()``
     - Distributed T-Digest quantile estimation

**Example Usage:**

::

    import dask.array as da
    from pyinterp.dask import binning2d

    # Create dask arrays
    x = da.from_array(lon_data, chunks=1000)
    y = da.from_array(lat_data, chunks=1000)
    z = da.from_array(values, chunks=1000)

    # Perform distributed binning
    result = binning2d(x, y, z, x_axis, y_axis)

T-Digest for Streaming Quantiles
================================

A new ``TDigest`` class provides efficient streaming quantile estimation:

::

    from pyinterp import TDigest

    # Create a T-Digest
    td = TDigest()

    # Add values (can be done in streaming fashion)
    td.update(values)

    # Query quantiles
    median = td.quantile(0.5)
    percentile_95 = td.quantile(0.95)

    # Get CDF value
    cdf = td.cdf(100.0)

Satellite Geometry Operations
=============================

A new ``geometry.satellite`` submodule provides satellite-specific algorithms:

- ``geometry.satellite.rotation`` - Satellite rotation operations
- ``geometry.satellite.transforms.swath`` - Swath transform algorithms

These replace the removed orbit classes (``EquatorCoordinates``, ``Orbit``,
``Pass``, ``Swath``) and functions (``calculate_orbit``, ``calculate_pass``,
``calculate_swath``).

Orbit Interpolation Changes
===========================

The ``orbit.interpolate()`` function has been updated:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Old Parameter
     - New Parameter
     - Notes
   * - ``wgs``
     - ``coordinates``
     - Parameter renamed
   * - ``half_window_size=10``
     - ``half_window_size=3``
     - Default value changed

**Example Migration:**

Before (Old API)::

    from pyinterp import orbit
    from pyinterp.geodetic import Coordinates

    result = orbit.interpolate(
        lon, lat, xp, xi,
        wgs=Coordinates(),
        half_window_size=10
    )

After (New API)::

    from pyinterp import orbit
    from pyinterp.geometry.geographic import Coordinates

    result = orbit.interpolate(
        lon, lat, xp, xi,
        coordinates=Coordinates(),
        half_window_size=3  # New default, adjust if needed
    )

Xarray Backend (backends.xarray)
================================

The xarray backend module has been significantly simplified. Axis detection is
now automatic using CF conventions, and the API is more streamlined.

Constructor Changes
-------------------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Old Parameter
     - New Behavior
     - Notes
   * - ``increasing_axes=True``
     - (removed)
     - Axes are handled automatically
   * - ``geodetic=True``
     - (removed)
     - Auto-detected via CF ``units``
   * - (manual axis setup)
     - Automatic detection
     - Lon/lat and temporal axes detected

Interpolation Method Changes
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Old API
     - New API
     - Notes
   * - ``grid.bicubic(coords, ...)``
     - ``grid.bivariate(coords, method="bicubic", ...)``
     - Method unified
   * - ``bicubic_kwargs={...}``
     - Direct keyword arguments
     - No wrapper dict needed
   * - ``interpolator="bilinear"``
     - ``method="bilinear"``
     - Parameter renamed

**Example Migration:**

Before (Old API)::

    from pyinterp.backends.xarray import RegularGridInterpolator
    import xarray as xr

    # Open dataset
    sst = xr.open_dataarray("sst.nc")

    # Create interpolator with explicit parameters
    interp = RegularGridInterpolator(
        sst,
        increasing_axes=True,
        geodetic=True
    )

    # Interpolate with bilinear
    result = interp(
        {"lon": lon_values, "lat": lat_values},
        method="bilinear",
        bounds_error=False,
        num_threads=4
    )

    # Bicubic interpolation required separate method or bicubic_kwargs
    result_bicubic = interp(
        {"lon": lon_values, "lat": lat_values},
        method="bicubic",
        bicubic_kwargs={"nx": 3, "ny": 3}
    )

After (New API)::

    from pyinterp.backends.xarray import RegularGridInterpolator
    import xarray as xr

    # Open dataset
    sst = xr.open_dataarray("sst.nc")

    # Create interpolator - automatic axis detection
    interp = RegularGridInterpolator(sst)

    # Interpolate with bilinear
    result = interp(
        {"lon": lon_values, "lat": lat_values},
        method="bilinear",
        bounds_error=False,
        num_threads=4
    )

    # Bicubic interpolation with direct keyword arguments
    result_bicubic = interp(
        {"lon": lon_values, "lat": lat_values},
        method="bicubic",
        half_window_size_x=3,
        half_window_size_y=3
    )

**Temporal Axis Handling:**

Temporal axes (with ``datetime64`` dtype) are now automatically detected::

    # 3D grid with time dimension
    sst_time = xr.open_dataarray("sst_time.nc")  # (lon, lat, time)
    interp = RegularGridInterpolator(sst_time)

    # Interpolate with datetime64 coordinates
    result = interp(
        {
            "lon": [10.5, 20.3],
            "lat": [45.2, -30.1],
            "time": np.array(["2020-01-15", "2020-02-15"], dtype="datetime64")
        },
        method="bilinear"
    )

API Changes Quick Reference
===========================

The following tables summarize the key API changes in recent releases.

Binding Framework
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Old
     - New
     - Notes
   * - pybind11
     - nanobind
     - Binding framework migration

Axis Class
----------

The ``Axis`` class now uses a ``period`` parameter instead of ``is_circle`` to
define periodic axes. This change provides more flexibility, allowing you to
specify any period value (e.g., 360 for longitude degrees, 3600 for seconds in
an hour, etc.).

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Old Parameter
     - New Parameter
     - Notes
   * - ``is_circle=True``
     - ``period=360.0``
     - Longitude axis (360 degrees)
   * - ``is_circle=True``
     - ``period=<value>``
     - Any periodic axis
   * - ``axis.is_circle``
     - ``axis.is_periodic``
     - Property renamed
   * - (not available)
     - ``axis.period``
     - New property to get period value

**Example Migration:**

Before (Old API)::

    # Old: Boolean flag for circular axis
    lon_axis = pyinterp.Axis(
        numpy.arange(0, 360, 1, dtype=numpy.float64),
        is_circle=True
    )
    print(f'Is a circle? {lon_axis.is_circle}')

After (New API)::

    # New: Specify the actual period
    lon_axis = pyinterp.Axis(
        numpy.arange(0, 360, 1, dtype=numpy.float64),
        period=360.0  # Period in degrees
    )
    print(f'Is periodic? {lon_axis.is_periodic}')
    print(f'Period: {lon_axis.period}')

    # Example: Time axis with 1-hour period (3600 seconds)
    time_axis = pyinterp.Axis(values, period=3600.0)

Removed Classes
---------------

The following classes have been removed:

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Removed
     - Replacement
     - Notes
   * - ``AxisInt64``
     - ``TemporalAxis``
     - Use ``TemporalAxis`` for datetime axes
   * - ``AxisBoundary`` (enum)
     - String parameters
     - Use string values like ``"expand"``, ``"wrap"``

Geometry Module Restructuring
-----------------------------

The ``geodetic`` module has been restructured into a unified ``geometry`` module
supporting both geographic and Cartesian coordinate systems:

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Old Path
     - New Path
     - Notes
   * - ``core.geodetic.Box``
     - ``core.geometry.geographic.Box``
     - Renamed module
   * - ``core.geodetic.Point``
     - ``core.geometry.geographic.Point``
     - Renamed module
   * - ``core.geodetic.Polygon``
     - ``core.geometry.geographic.Polygon``
     - Renamed module
   * - ``core.geodetic.LineString``
     - ``core.geometry.geographic.LineString``
     - Renamed module
   * - ``core.geodetic.MultiPolygon``
     - ``core.geometry.geographic.MultiPolygon``
     - Renamed module
   * - ``core.geodetic.Coordinates``
     - ``core.geometry.geographic.Coordinates``
     - Renamed module
   * - ``core.geodetic.RTree``
     - ``core.geometry.geographic.RTree``
     - Renamed module
   * - ``core.geodetic.Spheroid``
     - ``core.geometry.geographic.Spheroid``
     - Renamed module
   * - (new)
     - ``core.geometry.cartesian.*``
     - New Cartesian coordinate system

Additionally, geometry algorithms are now in a separate ``algorithms`` submodule:

- ``geometry.geographic.algorithms.area()``
- ``geometry.geographic.algorithms.distance()``
- ``geometry.cartesian.algorithms.buffer()``
- etc.

Fill Functions API
------------------

Fill functions now use configuration objects instead of keyword arguments:

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Old Parameter
     - New Configuration
     - Notes
   * - ``is_circle=True``
     - ``config.with_is_periodic(True)``
     - Renamed and moved to config
   * - ``first_guess=FirstGuess.Zero``
     - ``config.with_first_guess(FirstGuess.ZERO)``
     - Enum values now UPPERCASE
   * - ``max_iterations=100``
     - ``config.with_max_iterations(100)``
     - Moved to config object
   * - ``epsilon=1e-6``
     - ``config.with_epsilon(1e-6)``
     - Moved to config object
   * - ``num_threads=0``
     - ``config.with_num_threads(0)``
     - Moved to config object
   * - ``relaxation=1.0``
     - ``config.with_relaxation(1.0)``
     - Gauss-Seidel specific

**Example Migration:**

Before (Old API)::

    from pyinterp.core.fill import gauss_seidel_float64, FirstGuess

    iterations, residual = gauss_seidel_float64(
        grid,
        first_guess=FirstGuess.ZonalAverage,
        is_circle=True,
        max_iterations=100,
        epsilon=1e-6,
        relaxation=1.0,
        num_threads=4
    )

After (New API)::

    from pyinterp.core.fill import gauss_seidel
    from pyinterp.core.config.fill import GaussSeidel, FirstGuess

    config = GaussSeidel() \
        .with_first_guess(FirstGuess.ZONAL_AVERAGE) \
        .with_is_periodic(True) \
        .with_max_iterations(100) \
        .with_epsilon(1e-6) \
        .with_relaxation(1.0) \
        .with_num_threads(4)

    iterations, residual = gauss_seidel(grid, config)

RTree Interpolation Methods
---------------------------

RTree interpolation methods now use configuration objects:

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Old Enum/Parameter
     - New Enum/Configuration
     - Notes
   * - ``RadialBasisFunction.Gaussian``
     - ``RBFKernel.GAUSSIAN``
     - Enum renamed, values UPPERCASE
   * - ``WindowFunction.Hamming``
     - ``WindowKernel.HAMMING``
     - Enum renamed, values UPPERCASE
   * - ``CovarianceFunction.Matern_32``
     - ``CovarianceFunction.MATERN_32``
     - Values now UPPERCASE
   * - ``within=True``
     - ``config.with_boundary_check(BoundaryCheck.WITHIN)``
     - Renamed parameter

**Example Migration:**

Before (Old API)::

    result = rtree.radial_basis_function(
        coordinates,
        radius=1000.0,
        k=10,
        rbf=RadialBasisFunction.Gaussian,
        epsilon=1.0,
        smooth=0.0,
        within=True,
        num_threads=4
    )

After (New API)::

    from pyinterp.core.config.rtree import RadialBasisFunction, RBFKernel, BoundaryCheck

    config = RadialBasisFunction() \
        .with_radius(1000.0) \
        .with_k(10) \
        .with_rbf(RBFKernel.GAUSSIAN) \
        .with_epsilon(1.0) \
        .with_smooth(0.0) \
        .with_boundary_check(BoundaryCheck.WITHIN) \
        .with_num_threads(4)

    result = rtree.radial_basis_function(coordinates, config)

Loess Parameters
----------------

The ``loess`` function parameter names have been clarified:

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Old Parameter
     - New Parameter
     - Notes
   * - ``nx``
     - ``half_window_size_x``
     - Renamed for clarity
   * - ``ny``
     - ``half_window_size_y``
     - Renamed for clarity
   * - ``value_type=ValueType.All``
     - ``config.with_value_type(LoessValueType.ALL)``
     - Moved to config, values UPPERCASE

Type Hints Module
-----------------

The type hints module has been renamed:

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Old Import
     - New Import
     - Notes
   * - ``from pyinterp.typing import ...``
     - ``from pyinterp.type_hints import ...``
     - Module renamed

Summary of Enum Value Changes
-----------------------------

All enum values now use UPPERCASE with underscores:

.. list-table::
   :header-rows: 1
   :widths: 40 40

   * - Old Value
     - New Value
   * - ``FirstGuess.Zero``
     - ``FirstGuess.ZERO``
   * - ``FirstGuess.ZonalAverage``
     - ``FirstGuess.ZONAL_AVERAGE``
   * - ``ValueType.All``
     - ``LoessValueType.ALL``
   * - ``ValueType.Defined``
     - ``LoessValueType.DEFINED``
   * - ``ValueType.Undefined``
     - ``LoessValueType.UNDEFINED``
   * - ``RadialBasisFunction.Gaussian``
     - ``RBFKernel.GAUSSIAN``
   * - ``RadialBasisFunction.Multiquadric``
     - ``RBFKernel.MULTIQUADRIC``
   * - ``WindowFunction.Hamming``
     - ``WindowKernel.HAMMING``
   * - ``WindowFunction.Blackman``
     - ``WindowKernel.BLACKMAN``
   * - ``CovarianceFunction.Matern_32``
     - ``CovarianceFunction.MATERN_32``

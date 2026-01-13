Migration Guide
###############

This guide helps users migrate from older versions of pyinterp to the current API.

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

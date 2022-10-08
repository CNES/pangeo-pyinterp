Changelog
#########

2022.10.0
---------
* issues #13: Extract coordinates of the K nearest neighbors of a point.
* issues #12: Only the X-axis can be circular.
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

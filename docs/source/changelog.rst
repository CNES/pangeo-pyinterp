Changelog
#########

0.1.1 (17 January 2020)
-------------------------
* Use of MKL in the Conda package

0.1.0 (4 January 2020)
-------------------------
* Simplification of interpolations with Xarray
* Support for numpy datetime64
* Added 4D interpolation

0.0.8 (7 December 2019)
-------------------------
* Added interpolation by RBF in RTree
* Improvement of RTree class performance

0.0.7 (13 November 2019)
-------------------------
* Addition of binned bivariate statistics.
* Addition of bicubic interpolation in 3D space.
* Improved generation of the C++ extension.
* Analysis of test coverage.
* Minor bug fixes.

0.0.6 (4 October 2019)
-------------------------
* Improvement of axis performance.
* Fixed a performance problem with bilinear, bicubic and trivariate
  interpolations.
* Handling of time axes in the xarray backend.
* Access to main objects from the main module.

0.0.5 (19 September 2019)
-------------------------
* Use of the conda-forge channel

0.0.4 (16 September 2019)
-------------------------
* Simplification of the Xarray backend
* Merging of the conda-forge recipe
* Fix documentation issues

0.0.3 (29 July, 2019)
---------------------
* Optimization of memory management
* Improving bicubic interpolation performance
* Addition of methods to fill undefined values to solve interpolation problems
  near the coasts.
* Major redesign to separate grid management from interpolation routines.

0.0.2 (12 July, 2019)
---------------------
* Handle bounds error on structured grid interpolation

0.0.1 (8 July, 2019)
--------------------
* Initial release

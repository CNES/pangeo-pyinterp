About this project
==================

The motivation of this project is to provide tools for interpolating
geo-referenced data used in the field of geosciences. There are other libraries
that cover this problem, but written entirely in Python, the performance of
these projects was not quite sufficient for our needs. That is why this project
was created.

This first version can interpolate 2D fields using :py:class:`bivariate
<pyinterp.bivariate.Bivariate>` and :py:class:`bicubic
<pyinterp.bicubic.Bicubic>` interpolators, 3D fields using :py:class:`trivariate
<pyinterp.trivariate.Trivariate>` interpolators and :py:class:`unstructured
grids <pyinterp.rtree.RTree>`.

The library core is written in C++ using the `Boost C++ Libararies
<https://www.boost.org/>`_, `Eigen3 <http://eigen.tuxfamily.org/>`_, `GNU
Scientific Library <https://www.gnu.org/software/gsl/>`_ and `pybind11
<https://github.com/pybind/pybind11/>`_ libraries.

This software also uses `CMake <https://cmake.org/>`_ to configure the project
and `Googletest <https://github.com/google/googletest>`_ to perform unit testing
of the library kernel.
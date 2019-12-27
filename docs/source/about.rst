About this project
==================

The motivation of this project is to provide tools for interpolating
geo-referenced data used in the field of geosciences. There are other libraries
that cover this problem, but written entirely in Python, the performance of
these projects was not quite sufficient for our needs. That is why this project
was created.

This version can interpolate 2D fields using :py:func:`bivariate
<pyinterp.bivariate>` and :py:func:`bicubic <pyinterp.bicubic>` interpolators,
3D fields using :py:func:`trivariate <pyinterp.trivariate>` and ``bicubic``, 4D
fields using :py:func:`quadrivariate <pyinterp.quadrivariate>` interpolators and
:py:class:`unstructured grids <pyinterp.RTree>`. You can also apply a data
:py:class:`binning <pyinterp.Binning2D>` on bivariate field by simple or linear
binning.

The undefined values in the grids do not allow interpolation of values located
in the neighborhood. This behavior is a concern when you need to interpolate
values near the mask of some fields. The library provides utilities to fill the
undefined values:

* :py:func:`loess <pyinterp.fill.loess>` to fill the undefined values on the
  boundary between the defined/undefined values using local regression
* :py:func:`gauss_seidel <pyinterp.fill.gauss_seidel>` to fill all undefined
  values (NaN) in a grid using the Gauss-Seidel method by relaxation.

The library :doc:`core <core_api>` is written in C++ using the `Boost C++
Libararies <https://www.boost.org/>`_, `Eigen3 <http://eigen.tuxfamily.org/>`_,
`GNU Scientific Library <https://www.gnu.org/software/gsl/>`_ and `pybind11
<https://github.com/pybind/pybind11/>`_ libraries.

This software also uses `CMake <https://cmake.org/>`_ to configure the project
and `Googletest <https://github.com/google/googletest>`_ to perform unit testing
of the library kernel.
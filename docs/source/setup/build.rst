Building
********

We will present how to compile the code, install, and run the various scripts
with `setuptools <https://setuptools.readthedocs.io/en/latest/>`_.

.. _requirements:

Requirements
============

Because of the programs written in Python, and some parts of the library in
C++, you must have Python 3, at least Python version 3.6, a C++ compiler and
`cmake <https://cmake.org/>`_ installed on your system to build the library.

.. note::

   The C++ compiler must support the ISO C++ 2020 standard

The compiling C++ requires the following development library:
    * `Boost C++ Libraries <https://www.boost.org/>`_
    * `Eigen3 <http://eigen.tuxfamily.org/>`_

You can install these packages on Ubuntu by typing the following command:

.. code-block:: bash

    sudo apt-get install g++ cmake libeigen3-dev libboost-dev

You need, also, to install Python libraries before configuring and installing
this software:

* `dask <https://dask.org/>`_
* `numpy <https://www.numpy.org/>`_
* `xarray <http://xarray.pydata.org/en/stable/>`_

You can install these packages on Ubuntu by typing the following command:

.. code-block:: bash

    sudo apt-get install python3-numpy python3-xarray python3-dask

Compilation
===========

Once you have satisfied the requirements detailed above, to build the library,
type the command ``python3 setup.py build_ext`` at the root of the project.

You can specify, among other things, the following options:
    * ``--build-unittests`` to build the unit tests of the C++ extension.
    * ``--c-compiler`` to select the C compiler to use.
    * ``--cmake-args`` to pass additional arguments to CMake.
    * ``--code-coverage`` to enable coverage reporting on the C++ extension.
    * ``--cxx-compiler`` to select the C++ compiler to use.
    * ``--debug`` to compile the C++ library in Debug mode.
    * ``--generator`` to specify the generator to use with CMake.
    * ``--mkl`` to use MKL as BLAS library
    * ``--reconfigure`` to force CMake to reconfigure the project.

Run the ``python setup.py build_ext --help`` command to view all the options
available for building the library.

Project version
===============

The version is never stored in the sources: it is resolved at build time from
whichever of the following is available, the first one winning.

1. The ``PYINTERP_VERSION`` environment variable (``-DPYINTERP_VERSION=`` for a
   direct CMake invocation). Use it to pin the version explicitly.
2. ``git describe`` against the release tags. This is the normal case in a
   checkout.
3. ``VERSION.txt``. The file holds a ``$Format:...$`` placeholder that
   ``git archive`` expands, so source tarballs — including the ones GitHub
   generates — carry their version even though they have no ``.git``
   directory.
4. ``pyinterp/_version.py``, written by ``setuptools_scm`` into the sdist.
5. ``PKG-INFO``, the sdist metadata.

If every source fails, the build does not stop: it falls back to ``0.0.0`` and
warns. A version of ``0.0.0`` therefore means the version could not be
determined, not that a release is numbered that way.

``setup.py`` resolves the version once and passes it on to CMake, so the Python
package and the compiled extension always report the same value. Which source
was used is printed in the CMake configuration summary, and is available at
runtime:

.. code-block:: python

    import pyinterp

    pyinterp.__version__        # version of the Python package
    pyinterp.__core_version__   # version the C++ extension was compiled with
    pyinterp.core.__version_source__

A mismatch between the first two means the compiled extension is stale with
respect to the Python sources, which happens in an editable install that has
not been rebuilt.

Using the C++ library from CMake
================================

The C++ kernel can be consumed on its own, without the Python bindings. When
pyinterp is pulled in as a sub-project, the Python extension, nanobind and the
tests are skipped automatically:

.. code-block:: cmake

    include(FetchContent)
    FetchContent_Declare(
      pyinterp
      GIT_REPOSITORY https://github.com/CNES/pangeo-pyinterp.git
      GIT_TAG        2026.6.0
      GIT_SUBMODULES_RECURSE TRUE)
    FetchContent_MakeAvailable(pyinterp)

    target_link_libraries(my_target PRIVATE pyinterp::pyinterp)
    message(STATUS "Using pyinterp ${pyinterp_VERSION}")

``pyinterp_VERSION`` holds the normalized ``major.minor.patch`` version and
``pyinterp_VERSION_FULL`` the full descriptive string. Note that ``project()``
does not export ``<name>_VERSION`` to a parent scope on its own, so these two
cache variables are the supported way to read the version of an embedded copy.
The same information is available to the compiler through
``pyinterp/version.hpp``:

.. code-block:: cpp

    #include "pyinterp/version.hpp"

    static_assert(pyinterp::version_major >= 2026);

The behaviour can be overridden with the ``PYINTERP_BUILD_PYTHON_BINDINGS`` and
``PYINTERP_BUILD_TESTS`` options, which default to ``ON`` only when pyinterp is
the top-level project.

Testing
=======

Requirements
------------

Running tests require the following Python libraries:
    * `pytest <https://docs.pytest.org/en/latest/>`_
    * `numpy <https://www.numpy.org/>`_
    * `xarray <http://xarray.pydata.org/en/stable/>`_


Running tests
-------------

The distribution contains a set of test cases that can be processed with the
standard Python test framework. To run the full test suite,
use the following at the root of the project:

.. code-block:: bash

    pytest -v -ra

Generating the test coverage report
-----------------------------------

C++ kernel library
^^^^^^^^^^^^^^^^^^

To generate the unit test coverage report on the C++ extension, perform the
following steps:

.. code-block:: bash

    python setup.py build_ext --code-coverage --build-unittests
    python setup.py gtest
    genhtml coverage_cpp.lcov --output-directory htmllcov

The first command compiles the extension to generate a coverage mapping to allow
code coverage analysis. The second command runs the C++ unit tests and generates
the coverage report. The third command generates the associated HTML report with
`lcov <http://ltp.sourceforge.net/coverage/lcov.php>`_. The generated report is
available in the ``htmllcov`` directory located at the root of the project.

.. note::

    It's not possible to generate this report on Windows.

Python library
^^^^^^^^^^^^^^

To generate the unit test coverage report on the Python code, perform the
following step:

.. code-block:: bash

      pytest -v -ra --cov=pyinterp --cov-report=html

The HTML report is available in the ``htmlcov`` directory located at the root of
the project.

Global coverage report
^^^^^^^^^^^^^^^^^^^^^^

Is it possible to generate a global coverage report by combining the two previous
reports? To do this, type the following command:

.. code-block:: bash

    python setup.py build_ext --code-coverage --build-unittests
    python setup.py build
    python setup.py gtest
    pytest -v -ra --cov=pyinterp --cov-report=lcov --measure-coverage
    lcov --add-tracefile coverage_cpp.lcov --add-tracefile coverage.lcov --output-file merged_coverage.lcov
    lcov -r merged_coverage.lcov "${CONDA_PREFIX}/*" "/usr/*" "*/third_party/*" --output-file filtered_merged_coverage.lcov
    genhtml filtered_merged_coverage.lcov --output-directory htmllcov

The steps to generate a global coverage report are as follows:

1. Compile the extension to generate a coverage mapping for code coverage
   analysis.
2. Compile the Python extension.
3. Run the C++ unit tests and generate the coverage report.
4. Run the Python unit tests and generate the coverage report. The option
   ``--measure-coverage`` is used to reduce the number of data processed during
   the Python test, speeding up the process as the C++ extension is compiled
   without optimization.
5. Merge the two coverage reports.
6. Filter the coverage report to remove the system and third-party libraries.
7. Generate the associated HTML report with `lcov
   <http://ltp.sourceforge.net/coverage/lcov.php>`_.

The generated report is available in the ``htmllcov`` directory located at the root of the project.

Automatic Documentation
=======================

`Sphinx <http://www.sphinx-doc.org/en/master/>`_ manages the source code of this
documentation. It is possible to generate it to produce a local mini WEB site to
read and navigate it.
To do this, type the following command: ::

    sphinx-build -b html docs/source docs/build

.. note::

    The documentation uses `sphinx-book-theme
    <https://github.com/executablebooks/sphinx-book-theme>`_ as HTML style. This
    package must be available before running the above command. You can install
    it with conda-forge or pip.

Install
=======

To install this library, type the command ``python3 -m pip install .``.

To pass options to the ``build_ext`` command, use the ``--config-settings`` or
``-C`` option to pip. For instance, to compile the library with MKL as the BLAS
library and using the Visual Studio 17 2022 generator, run the following
command:

.. code-block:: bash

    python3 -m pip install . -Cmkl=yes -Cgenerator="Visual Studio 17 2022"

The available options are a subset of the ``setup.py build_ext`` command options
for use with pip to install the library:

* ``c-compiler``
* ``cxx-compiler``
* ``generator``
* ``cmake-args``
* ``mkl``

.. note::

    Only ``mkl`` option is a boolean option. The others are strings.

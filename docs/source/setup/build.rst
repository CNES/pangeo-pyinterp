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

   The C++ compiler must support the ISO C++ 2017 standard

The compiling C++ requires the following development library:
    * `Boost C++ Libraries <https://www.boost.org/>`_
    * `Eigen3 <http://eigen.tuxfamily.org/>`_
    * `GNU Scientific Library <https://www.gnu.org/software/gsl/>`_

You can install these packages on Ubuntu by typing the following command:

.. code-block:: bash

    sudo apt-get install g++ cmake libeigen3-dev libboost-dev libgsl-dev

You need, also, to install Python libraries before configuring and installing
this software:

* `numpy <https://www.numpy.org/>`_

You can install these packages on Ubuntu by typing the following command:

.. code-block:: bash

    sudo apt-get install python3-numpy

Compilation
===========

Once you have satisfied the requirements detailed above, to build the library,
type the command ``python3 setup.py build_ext`` at the root of the project.

You can specify, among other things, the following options:
    * ``--boost-root`` to specify the Preferred Boost installation prefix.
    * ``--build-unittests`` to build the unit tests of the C++ extension.
    * ``--conda-forge`` to use the generation parameters of the conda-forge
      package.
    * ``--code-coverage`` to enable coverage reporting on the C++ extension.
    * ``--c-compiler`` to select the C compiler to use.
    * ``--cxx-compiler`` to select the C++ compiler to use.
    * ``--debug`` to compile the C++ library in Debug mode.
    * ``--eigen-root`` to specify the Eigen3 include directory.
    * ``--gsl-root`` to specify the Preferred GSL installation prefix.
    * ``--mkl-root`` to specify the MKL directory.
    * ``--mkl`` to use MKL as BLAS library
    * ``--reconfigure`` to force CMake to reconfigure the project.

Run the ``python setup.py build_ext --help`` command to view all the options
available for building the library.

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

    python setup.py test

Generating the test coverage report
-----------------------------------

C++ kernel library
^^^^^^^^^^^^^^^^^^

To generate the unit test coverage report on the C++ extension, perform the
following steps:

.. code-block:: bash

    python setup.py build_ext --code-coverage --build-unittests
    python setup.py test --ext-coverage

The first command compiles the extension to generate a coverage mapping to allow
code coverage analysis. The second command performs the Python and C++ unit
tests, analyze the coverage of the C++ code, and generates the associated HTML
report with `lcov <http://ltp.sourceforge.net/coverage/lcov.php>`_. The
generated report is available in the ``htmllcov`` directory located at the root
of the project.

.. note::

    It's not possible to generate this report on Windows.

Python library
^^^^^^^^^^^^^^

To generate the unit test coverage report on the Python code, perform the
following step:

.. code-block:: bash

      python setup.py test --pytest-args="--cov=pyinterp --cov-report=html"

The HTML report is available in the ``htmlcov`` directory located at the root of
the project.

Automatic Documentation
=======================

`Sphinx <http://www.sphinx-doc.org/en/master/>`_ manages the source code of this
documentation. It is possible to generate it to produce a local mini WEB site to
read and navigate it.
To do this, type the following command: ::

    python setup.py build_sphinx

.. note::

    The documentation uses `furo <https://github.com/pradyunsg/furo>`_ as HTML
    style. This package must be available before running the above command. You
    can install it with corda-forge or pip.

Install
=======

To install this library, type the command ``python3 -m pip install .``.

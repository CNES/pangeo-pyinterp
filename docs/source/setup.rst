Setup with the existing python
===============================

We will present how to compile the code, install and run the various scripts
with `setuptools <https://setuptools.readthedocs.io/en/latest/>`_.

Requirements
############

Because the programs are written in Python, and some parts of the library in
C++, you must have Python 3, at least Python version 3.6, a C++ compiler and
`cmake <https://cmake.org/>`_ installed on your system to build the library.

.. note::

   The C++ compiler must support the ISO C++ 2017 standard

The compiling C++ requires the following development library:

    * `Boost C++ Libararies <https://www.boost.org/>`_
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

Build
#####

Once you have satisfied the requirements detailed above, to build the library,
type the command ``python3 setup.py build`` at the root of the project.

You can specify, among other things, the following options:

    * ``--boost-root`` to specify the Preferred Boost installation prefix.
    * ``--build-unittests``  to build the unit tests of the C++ extension.
    * ``--code-coverage`` to enable coverage reporting on the C++ extension.
    * ``--cxx-compiler`` to select the C++ compiler to use.
    * ``--debug`` to compile the C++ library in Debug mode.
    * ``--eigen-root`` to specify the Eigen3 include directory.
    * ``--gsl-root`` to specify the Preferred GSL installation prefix.
    * ``--reconfigure``  to force CMake to reconfigure the project.

Run the ``python setup.py build --help`` command to view all the options
available for building the library.

Test
####

Requirements
------------

Running tests require the following Python libraries:

    * `pytest <https://docs.pytest.org/en/latest/>`_
    * `numpy <https://www.numpy.org/>`_
    * `xarray <http://xarray.pydata.org/en/stable/>`_
    * `netCDF4 <https://unidata.github.io/netcdf4-python/>`_


Running tests
-------------

The distribution contains a set of test cases that can be processed with the
standard Python test framework. To run the full test suite,
use the following at the root of the project:

.. code-block:: bash

    python setup.py test

Generating the test coverage report
-----------------------------------

C++ extension
^^^^^^^^^^^^^

To generate the unit test coverage report on the C++ extension, perform the
following steps:

.. code-block:: bash

    python setup.py build --code-coverage --build-unittests
    python setup.py test --ext-coverage

The first command compiles the extension to generate a coverage mapping to allow
code coverage analysis. The second command performs the Python and C++ unit
tests, analyze the coverage of the C++ code and generates the associated HTML
report with `lcov <http://ltp.sourceforge.net/coverage/lcov.php>`_. The
generated report is available in the ``htmllcov`` directory located at the root
of the project.

.. note::

    It's not possible to generate this report on Windows.

Python code
^^^^^^^^^^^

To generate the unit test coverage report on the Python code, perform the
following step:

.. code-block:: bash

      python setup.py test --pytest-args="--cov=pyinterp --cov-report=html"

The HTML report is available in the ``htmlcov`` directory located at the root of
the project.

Automatic Documentation
#######################

The source code of this documentation is managed by
`sphinx <http://www.sphinx-doc.org/en/master/>`_. It is possible to
generate it in order to produce a local mini WEB site to read and navigate it.
To do this, type the following command: ::

    python setup.py build_sphinx

Install
#######

To install just type the command ``python3 setup.py``. You can specify an
alternate installation path, with:

.. code-block:: bash

    python setup.py install --prefix=/opt/local

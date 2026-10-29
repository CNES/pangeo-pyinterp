Building from source
********************

This page is for contributors and for platforms without a pre-built package.
The C++ extension is configured and compiled by `CMake <https://cmake.org/>`_,
driven by ``setup.py``. To simply use the library, install it with :doc:`pip`
or :doc:`conda` instead.

.. _quick-start:

Quick start
===========

The simplest way to get every build dependency is a conda environment. The
files used by the continuous integration are in ``.github/environments``:

.. tab:: Linux

   .. code-block:: bash

      git clone --recursive https://github.com/CNES/pangeo-pyinterp.git
      cd pangeo-pyinterp
      mamba env create -n pyinterp-dev -f .github/environments/linux.yml
      mamba activate pyinterp-dev

.. tab:: macOS

   .. code-block:: bash

      git clone --recursive https://github.com/CNES/pangeo-pyinterp.git
      cd pangeo-pyinterp
      mamba env create -n pyinterp-dev -f .github/environments/osx.yml
      mamba activate pyinterp-dev

.. tab:: Windows

   .. code-block:: bash

      git clone --recursive https://github.com/CNES/pangeo-pyinterp.git
      cd pangeo-pyinterp
      mamba env create -n pyinterp-dev -f .github/environments/win.yml
      mamba activate pyinterp-dev

   Run the build commands from a Visual Studio developer prompt (see
   :ref:`build-options` for the generator).

Then compile the extension and run the tests:

.. code-block:: bash

   python setup.py build_ext   # configure and compile the C++ extension
   python setup.py build       # assemble the package in build/lib.*
   pytest -v -ra               # run the Python tests

.. _requirements:

Requirements
============

* Python 3.12 or later.
* A C++ compiler supporting C++23, for example GCC 12 or later, a recent Clang,
  or a recent MSVC.
* CMake 3.19 or later. A ``pip`` build fetches it automatically.
* `Boost <https://www.boost.org/>`_ 1.90 or later (headers only). Most Linux
  distributions ship an older version; the conda-forge packages
  ``libboost-devel`` or ``libboost-headers`` are recent enough.
* The Python packages ``numpy`` and ``dask`` at runtime, ``xarray`` for the
  xarray backend and to run the tests, and ``setuptools`` 64 or later and
  ``setuptools-scm`` 8 or later to build.

`Eigen <https://eigen.tuxfamily.org/>`_, `nanobind
<https://github.com/wjakob/nanobind>`_ and `pocketfft
<https://github.com/mreineck/pocketfft>`_ are bundled as git submodules. If the
repository was cloned without ``--recursive``, fetch them with:

.. code-block:: bash

   git submodule update --init --recursive

Optional dependencies:

* `GoogleTest <https://github.com/google/googletest>`_ (``gtest``) to build the
  C++ unit tests. They are skipped when it is not found.
* A BLAS library: MKL, Apple Accelerate or OpenBLAS. Without one, Eigen's own
  implementation is used.
* `ccache <https://ccache.dev/>`_ and `Ninja <https://ninja-build.org/>`_ to
  speed up the build (see :ref:`speed-up`).

Building and installing
=======================

Compile the extension with:

.. code-block:: bash

   python setup.py build_ext

The CMake build tree is kept in ``build/temp.<platform>-<python>``, and the
package is assembled in ``build/lib.<platform>-<python>`` by
``python setup.py build``. The tests are run from there.

To install the library in the active environment, use ``pip``:

.. code-block:: bash

   python -m pip install .

.. _build-options:

Build options
-------------

The ``build_ext`` command accepts the following options. Some of them can also
be passed to ``pip`` with ``-C`` (``--config-settings``):

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - ``build_ext`` option
     - ``pip -C`` key
     - Description
   * - ``--cmake-args=ARGS``
     - ``cmake-args``
     - Additional arguments for CMake, for instance ``-DPYINTERP_USE_PCH=OFF``.
       See :ref:`cmake-options`.
   * - ``--cxx-compiler=PATH``
     - ``cxx-compiler``
     - C++ compiler to use.
   * - ``--generator=NAME``
     - ``generator``
     - CMake generator, for instance ``Ninja``. On Windows, the default is the
       ``CMAKE_GEN`` environment variable, or ``Visual Studio 17 2022``.
   * - ``--fft=NAME``
     - ``fft``
     - FFT implementation: ``pocketfft`` (default) or ``mkl``. ``mkl`` requires
       MKL to be found as the BLAS library.
   * - ``--mkl``
     - ``mkl=yes``
     - Use MKL as the BLAS library. Sets ``MKLROOT`` when the MKL headers are
       found in the active Python prefix (for example the conda package
       ``mkl-devel``).
   * - ``--debug``
     -
     - Compile in ``Debug`` mode instead of ``Release``.
   * - ``--build-unittests``
     -
     - Also build the C++ unit tests (see :ref:`cpp-tests`).
   * - ``--enable-coverage``
     -
     - Instrument the C++ code for coverage (GCC or Clang, not on Windows).
   * - ``--export-compile-commands``
     -
     - Write ``compile_commands.json`` in the build tree, for clangd and other
       tools.
   * - ``--reconfigure``
     -
     - Run the CMake configuration again on an existing build tree.

Run ``python setup.py build_ext --help`` to see all the options.

.. important::

   An existing build tree is not configured again: options that change the
   CMake configuration (``--cmake-args``, ``--fft``, ``--cxx-compiler``, the
   compiler launcher...) only take effect on a fresh build tree or with
   ``--reconfigure``.

For example, to install the library with MKL and Ninja:

.. code-block:: bash

   python -m pip install . -Cmkl=yes -Cgenerator=Ninja

.. _cmake-options:

CMake options
-------------

The following CMake options have no dedicated ``setup.py`` flag. Pass them with
``--cmake-args`` (``-Ccmake-args`` for ``pip``), or directly to ``cmake`` when
building the C++ library without ``setup.py``:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``PYINTERP_USE_PCH``
     - ``ON``
     - Compile the Python extension with a precompiled header
       (``cxx/src/pybind/pch.hpp``). It is ignored when ``ENABLE_CLANG_TIDY``
       is ``ON``.
   * - ``ENABLE_CLANG_TIDY``
     - ``OFF``
     - Run clang-tidy on each compiled file. See :ref:`static-analysis`.
   * - ``ENABLE_COVERAGE``
     - ``OFF``
     - Instrument the code for coverage; set by ``--enable-coverage``.
   * - ``FFT_IMPLEMENTATION``
     - ``pocketfft``
     - FFT implementation; set by ``--fft``.
   * - ``PYINTERP_BUILD_PYTHON_BINDINGS``
     - ``ON`` [#top]_
     - Build the ``pyinterp.core`` Python extension.
   * - ``PYINTERP_BUILD_TESTS``
     - ``ON`` [#top]_
     - Build the C++ unit tests (requires GoogleTest).
   * - ``CMAKE_BUILD_TYPE``
     - ``Release`` [#type]_
     - Besides the standard types, ``ASan`` compiles with the address and
       undefined behavior sanitizers (GCC or Clang).

.. [#type] Set by ``setup.py`` (``Debug`` with ``--debug``,
   ``RelWithDebInfo`` with ``--enable-coverage``); a direct CMake invocation
   defaults to ``RelWithDebInfo``.
.. [#top] Only when pyinterp is the top-level project; ``OFF`` when it is a
   sub-project (see :ref:`cmake-subproject`).

The configuration summary printed by CMake shows the value of these options.

Environment variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Effect
   * - ``CMAKE_BUILD_PARALLEL_LEVEL``
     - Number of parallel compilation jobs. Defaults to the number of CPUs.
   * - ``CMAKE_CXX_COMPILER_LAUNCHER``
     - Compiler launcher, for example ``ccache``. Read at configuration time
       only.
   * - ``CMAKE_PREFIX_PATH``
     - Where CMake looks for Boost and the other dependencies. Defaults to the
       active conda environment (``CONDA_PREFIX``).
   * - ``CMAKE_ARGS``
     - Additional CMake arguments, placed before ``--cmake-args``.
   * - ``CMAKE_GEN``
     - Default CMake generator on Windows.
   * - ``MKLROOT``
     - Use MKL as the BLAS library.
   * - ``PYINTERP_VERSION``
     - Version of the build (see :ref:`project-version`).

.. _speed-up:

Speeding up the build
=====================

The extension is made of many heavily templated translation units, so a full
build takes a while. The following settings make it faster, especially when the
same tree is rebuilt often.

Precompiled header
   The Python extension is compiled with a precompiled header that holds the
   Boost.Geometry, Eigen, nanobind and standard library headers shared by all
   its source files. It is enabled by default; disable it with
   ``--cmake-args=-DPYINTERP_USE_PCH=OFF`` if your compiler has trouble with it.

Ninja
   The Ninja generator schedules the jobs better than Make and starts a no-op
   build faster: add ``--generator=Ninja``.

Parallel jobs
   Every CPU is used by default. Each job can use 1 to 2 GB of memory; if the
   machine runs out of memory, lower ``CMAKE_BUILD_PARALLEL_LEVEL``.

ccache
   `ccache <https://ccache.dev/>`_ caches the compiled objects, so a rebuild
   after switching branches or deleting the build directory only compiles the
   files that changed. Enable it on a fresh build tree:

   .. code-block:: bash

      export CMAKE_CXX_COMPILER_LAUNCHER=ccache
      export CCACHE_SLOPPINESS=pch_defines,time_macros,include_file_mtime,include_file_ctime
      python setup.py build_ext --reconfigure

   ``CCACHE_SLOPPINESS`` is needed for ccache to cache the files compiled with
   the precompiled header. Check the hit rate with ``ccache --show-stats``.

   .. note::

      For a coverage build, unset ``CCACHE_BASEDIR``: with relative paths, the
      coverage data cannot be matched to the sources.

When you edit the C++ code, run ``python setup.py build_ext`` and
``python setup.py build`` again: only the modified files are recompiled. If
``pyinterp.__version__`` and ``pyinterp.__core_version__`` differ, the compiled
extension is out of date.

Testing
=======

.. _python-tests:

Python tests
------------

The Python tests require `pytest <https://docs.pytest.org/>`_. They are
collected from the ``build`` directory, so build the package first:

.. code-block:: bash

   python setup.py build_ext
   python setup.py build
   pytest -v -ra

.. _cpp-tests:

C++ tests
---------

The C++ unit tests are configured in the same build tree when GoogleTest is
found; otherwise they are skipped. Build them together with the extension, then
run them with ``ctest``:

.. code-block:: bash

   python setup.py build_ext --build-unittests
   ctest --test-dir $(python scripts/show_temp_build_dir.py)/cxx/tests \
       --output-on-failure

The ``test_all`` target builds and runs them in one step:

.. code-block:: bash

   cmake --build $(python scripts/show_temp_build_dir.py) --target test_all

Coverage report
---------------

The script ``scripts/generate_coverage.sh`` builds the extension with coverage
enabled, runs the Python and C++ tests, and merges both reports into
``build/htmllcov``. It requires GCC or Clang and `LCOV
<https://github.com/linux-test-project/lcov>`_ 2.0 or later, and is not
supported on Windows.

.. code-block:: bash

   ./scripts/generate_coverage.sh

.. _static-analysis:

Static analysis and formatting
==============================

clang-tidy
----------

The clang-tidy checks are defined in the ``.clang-tidy`` file at the root of the
project. The same file is used by clang-tidy, by `clangd
<https://clangd.llvm.org/>`_ in your editor, and by the ``ENABLE_CLANG_TIDY``
option. clangd needs a compilation database, written by
``--export-compile-commands``.

To analyze the whole project, export the compilation database and run
``run-clang-tidy`` on it:

.. code-block:: bash

   python setup.py build_ext --export-compile-commands --reconfigure
   run-clang-tidy -p $(python scripts/show_temp_build_dir.py) \
       "$(pwd)/cxx/(include|src|tests)/"

The ``ENABLE_CLANG_TIDY`` option runs clang-tidy during the compilation
instead:

.. code-block:: bash

   python setup.py build_ext --reconfigure --cmake-args=-DENABLE_CLANG_TIDY=ON

.. warning::

   ``ENABLE_CLANG_TIDY`` runs clang-tidy with ``-fix``: the fixes are applied to
   the source files directly. It also disables the precompiled header, so the
   build is much slower.

Formatting
----------

The code style is checked by `pre-commit <https://pre-commit.com/>`_: ruff for
Python, clang-format for C++, cmake-format for CMake, and mypy for the type
annotations. Install the hooks once:

.. code-block:: bash

   pre-commit install

They then run on each commit; ``pre-commit run --all-files`` checks the whole
project.

.. _build-docs:

Building the documentation
==========================

The documentation is generated by `Sphinx <https://www.sphinx-doc.org/>`_ from
the package built in ``build/lib.*``, so compile the extension first. It
requires ``sphinx``, ``sphinx-book-theme``, ``sphinx-gallery``,
``sphinx-inline-tabs``, and ``matplotlib`` and ``cartopy`` for the gallery of
examples. For example, with a dedicated environment:

.. code-block:: bash

   mamba create -n pyinterp-doc -c conda-forge python=3.13 cxx-compiler cmake \
       ninja libboost-devel numpy xarray dask pandas numcodecs matplotlib \
       cartopy setuptools setuptools-scm sphinx sphinx-book-theme \
       sphinx-gallery sphinx-inline-tabs
   mamba activate pyinterp-doc
   python setup.py build_ext --generator=Ninja
   python setup.py build
   sphinx-build -b html docs/source docs/build

Open ``docs/build/index.html`` in a browser to read the result.

Reference
=========

.. _project-version:

Project version
---------------

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

.. _cmake-subproject:

Using the C++ library from CMake
--------------------------------

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

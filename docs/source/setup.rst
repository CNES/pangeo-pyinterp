Installing the library
======================

This chapter presents the steps to build the library, run the tests, generate
the test coverage report and finally produce this automatic documentation.

Pre-built binaries are available on both `PyPI <https://pypi.org/p/pyinterp>`_
and `conda-forge <https://conda-forge.org>`_, so most users do not need to
compile anything. ``pip install pyinterp`` will fetch a wheel for Linux
(manylinux_2_28 x86_64 / aarch64), macOS (arm64) or Windows (x86_64) when one
matches your interpreter; otherwise pip falls back to building from source,
which requires CMake, a C++20 compiler, Boost and Eigen. ``conda install -c
conda-forge pyinterp`` is an alternative that always installs binaries together
with their compiled dependencies. Building the library by hand is documented for
contributors and for platforms that are not covered by the published wheels.

.. toctree::
   :maxdepth: 1
   :hidden:

   setup/build
   setup/pip
   setup/conda

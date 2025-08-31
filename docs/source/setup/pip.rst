Setup with pip
**************

You can also install the library with ``pip``. Since no wheels are provided, pip
builds the library from source, so all dependencies must be installed beforehand

Prerequisites see :ref:`requirements` in :doc:`build`.

Basic install::

    pip install pyinterp

Upgrade to the latest release::

    pip install -U pyinterp

If the build fails, verify CMake, a C++20 compiler, Boost and Eigen are present,
then retry with increased verbosity::

    pip install -v pyinterp

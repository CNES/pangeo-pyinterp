Setup with pip
**************

``pyinterp`` is published on `PyPI <https://pypi.org/p/pyinterp>`_ as binary
wheels for the most common platforms, so a plain ``pip install`` is usually
all you need.

Pre-built wheels are available for:

* **Linux** — ``manylinux_2_28`` on ``x86_64`` and ``aarch64`` (glibc ≥ 2.28)
* **macOS** — ``arm64`` (Apple Silicon), deployment target 13.4
* **Windows** — ``x86_64``

Wheels are built for CPython 3.12, 3.13, 3.14 and 3.15, and for the
free-threaded builds 3.14t and 3.15t.

Install the latest release, or upgrade an existing installation:

.. code-block:: bash

   pip install -U pyinterp

The ``xarray`` extra also installs `xarray <https://xarray.dev/>`_, needed by
:mod:`pyinterp.backends.xarray`, the ``pyinterp`` accessor of data arrays and
:func:`pyinterp.geohash.to_xarray`:

.. code-block:: bash

   pip install -U "pyinterp[xarray]"

The ``dask`` extra installs `Dask <https://docs.dask.org/>`_, needed by the
:mod:`pyinterp.dask` helpers; the extras can be combined:

.. code-block:: bash

   pip install -U "pyinterp[xarray,dask]"

Pre-release builds (``X.Y.ZrcN``, ``X.Y.Z.devN``) are uploaded to TestPyPI; you
can try them with:

.. code-block:: bash

   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ pyinterp

If no wheel matches your platform (for example a non-glibc Linux or 32-bit
Windows), pip builds the library from source. This requires a C++23 compiler
and Boost 1.90 or later; see :ref:`requirements`. If the build fails, retry
with increased verbosity to see the compiler output:

.. code-block:: bash

   pip install -v pyinterp

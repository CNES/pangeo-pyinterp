Setup with pip
**************

``pyinterp`` is published on `PyPI <https://pypi.org/p/pyinterp>`_ as binary
wheels for the most common platforms, so a plain ``pip install`` is usually
all you need.

Pre-built wheels are available for:

* **Linux** — ``manylinux_2_28`` on ``x86_64`` and ``aarch64`` (glibc ≥ 2.28)
* **macOS** — ``arm64`` (Apple Silicon), deployment target 13.4
* **Windows** — ``x86_64``

Wheels are built for CPython 3.12, 3.13, 3.14 and the free-threaded
build 3.14t.

Basic install::

    pip install pyinterp

Upgrade to the latest release::

    pip install -U pyinterp

Pre-release builds (``X.Y.ZrcN``, ``X.Y.Z.devN``) are uploaded to TestPyPI; you
can try them with::

    pip install --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ pyinterp

If no wheel matches your interpreter (for example a non-glibc Linux, 32-bit
Windows or an unsupported Python version), pip falls back to building from
source. In that case, all build dependencies must be available beforehand —
see :ref:`requirements` in :doc:`build`. If the build fails, verify CMake,
a C++20 compiler, Boost and Eigen are present, then retry with increased
verbosity::

    pip install -v pyinterp

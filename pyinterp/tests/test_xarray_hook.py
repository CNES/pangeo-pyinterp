# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for the registration of the xarray accessors."""

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

import pyinterp


#: Directory holding the tested pyinterp package.
ROOT = pathlib.Path(pyinterp.__file__).parent.parent


def _run(script: str) -> str:
    """Run a script in a new interpreter, warnings raised as errors.

    Each script needs its own interpreter, since the order of the imports is
    what is tested.

    Returns:
        The standard output of the script.

    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), *filter(None, [env.get("PYTHONPATH")])]
    )
    process = subprocess.run(  # noqa: S603
        [sys.executable, "-W", "error", "-c", textwrap.dedent(script)],
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=env,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    return process.stdout.strip()


def test_pyinterp_does_not_import_xarray() -> None:
    """Importing pyinterp leaves xarray unloaded."""
    assert (
        _run(
            """
            import sys
            import pyinterp
            print("xarray" in sys.modules)
            """
        )
        == "False"
    )


def test_xarray_imported_after() -> None:
    """The accessors are registered once xarray is imported."""
    assert (
        _run(
            """
            import sys
            import pyinterp
            import xarray
            print(
                hasattr(xarray.DataArray, "pyinterp"),
                hasattr(xarray.Dataset, "pyinterp"),
                any(
                    type(item).__name__ == "_XarrayFinder"
                    for item in sys.meta_path
                ),
            )
            """
        )
        == "True True False"
    )


def test_xarray_imported_before() -> None:
    """The accessors are registered at once if xarray is loaded."""
    assert (
        _run(
            """
            import xarray
            import pyinterp
            print(hasattr(xarray.DataArray, "pyinterp"))
            """
        )
        == "True"
    )


def test_backend_imported_first() -> None:
    """Importing the backend, which imports xarray, registers the accessors."""
    assert (
        _run(
            """
            import pyinterp.backends.xarray
            import xarray
            print(hasattr(xarray.DataArray, "pyinterp"))
            """
        )
        == "True"
    )


def test_without_xarray() -> None:
    """Pyinterp works without xarray and says how to install it."""
    output = _run(
        """
        import sys
        sys.modules["xarray"] = None
        import numpy
        import pyinterp
        pyinterp.Axis(numpy.arange(3.0))
        try:
            import pyinterp.backends.xarray
        except ImportError as exc:
            print(exc)
        """
    )
    assert "pip install pyinterp[xarray]" in output


def test_registration_failure() -> None:
    """A failure to register the accessors does not break xarray."""
    output = _run(
        """
        import sys
        import warnings
        import pyinterp
        sys.modules["pyinterp.backends.xarray"] = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import xarray
        print(hasattr(xarray, "DataArray"))
        print(caught[0].category.__name__, caught[0].message)
        """
    )
    lines = output.splitlines()
    assert lines[0] == "True"
    assert lines[1].startswith("RuntimeWarning the pyinterp accessors")


@pytest.mark.parametrize("module", ["xarray", "xarray.core.dataarray"])
def test_submodule(module: str) -> None:
    """Importing a submodule of xarray first also registers the accessors."""
    assert (
        _run(
            f"""
            import pyinterp
            import {module}
            import xarray
            print(hasattr(xarray.DataArray, "pyinterp"))
            """
        )
        == "True"
    )

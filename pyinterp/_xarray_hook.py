# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Register the xarray accessors without importing xarray.

xarray takes a long time to import, so :py:mod:`pyinterp` does not import it.
The ``pyinterp`` accessors of :py:mod:`pyinterp.backends.xarray` are
nevertheless available as soon as there is a data array to use them on: they
are registered at once if xarray is already imported, and otherwise right
after xarray is imported, by a finder that watches for it.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import types
    from collections.abc import Sequence
    from importlib.machinery import ModuleSpec


def _register_accessors() -> None:
    """Import the xarray backend, which registers the accessors.

    A failure is reported as a warning: it must not break the import of
    xarray in the code of the user.
    """
    try:
        importlib.import_module("pyinterp.backends.xarray")
    except Exception as exc:
        warnings.warn(
            f"the pyinterp accessors of xarray could not be registered: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


class _XarrayFinder(importlib.abc.MetaPathFinder):
    """Register the accessors once xarray has been imported.

    The finder does not find anything itself: when xarray is about to be
    imported, it removes itself from :py:data:`sys.meta_path`, lets the other
    finders locate xarray and wraps the execution of the module so that the
    accessors are registered once xarray is initialized.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> ModuleSpec | None:
        """Wrap the loader of xarray; other modules are left to other finders.

        Returns:
            The specification of xarray found by the other finders, or None.

        """
        del path, target
        if fullname != "xarray":
            return None
        _uninstall()
        spec = importlib.util.find_spec(fullname)
        loader = spec.loader if spec is not None else None
        if loader is None:
            return spec
        exec_module = loader.exec_module

        def wrapper(module: types.ModuleType) -> None:
            # Restore the method of the loader before anything can fail.
            del loader.exec_module
            exec_module(module)
            _register_accessors()

        try:
            loader.exec_module = wrapper  # type: ignore[method-assign]
        except AttributeError:
            # The loader does not accept new attributes: import xarray as
            # usual, without registering the accessors.
            pass
        return spec


def _uninstall() -> None:
    """Remove the finders watching for xarray."""
    sys.meta_path[:] = [
        item for item in sys.meta_path if not isinstance(item, _XarrayFinder)
    ]


def install() -> None:
    """Register the accessors now, or once xarray is imported."""
    # A module set to None in sys.modules is one whose import is blocked.
    if sys.modules.get("xarray") is not None:
        _register_accessors()
    elif not any(isinstance(item, _XarrayFinder) for item in sys.meta_path):
        sys.meta_path.insert(0, _XarrayFinder())

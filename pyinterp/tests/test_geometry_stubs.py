# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Guard that the generated ``pyinterp.geometry`` stub tree is in sync."""

from __future__ import annotations

import pathlib
import sys

import pytest


REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO / "scripts"

if SCRIPTS_DIR.is_dir():
    sys.path.insert(0, str(SCRIPTS_DIR))
    import generate_geometry_stubs  # type: ignore[import-not-found]

    sys.path.pop(0)
else:
    # Running from an installed wheel: the generator script is not shipped.
    generate_geometry_stubs = None  # type: ignore[assignment]


@pytest.mark.skipif(
    generate_geometry_stubs is None,
    reason="Generator script not available (installed wheel layout).",
)
def test_geometry_stubs_in_sync() -> None:
    """``pyinterp/geometry/`` must mirror ``pyinterp/core/geometry/``.

    If this fails, run ``python scripts/generate_geometry_stubs.py`` and
    commit the result.
    """
    assert generate_geometry_stubs.check(), (
        "pyinterp/geometry/ stubs are out of sync with "
        "pyinterp/core/geometry/. Run "
        "`python scripts/generate_geometry_stubs.py` and commit the result."
    )

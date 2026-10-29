# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests of the dask helpers when dask is not installed."""

import sys

import numpy as np
import pytest

import pyinterp.dask


@pytest.mark.parametrize(
    ("helper", "nargs"),
    [
        ("descriptive_statistics", 1),
        ("tdigest", 1),
        ("binning1d", 3),
        ("binning2d", 4),
        ("histogram2d", 4),
    ],
)
def test_without_dask(
    monkeypatch: pytest.MonkeyPatch, helper: str, nargs: int
) -> None:
    """The helpers say how to install dask when it is missing."""
    # A module set to None in sys.modules cannot be imported.
    monkeypatch.setitem(sys.modules, "dask", None)
    monkeypatch.setitem(sys.modules, "dask.array", None)
    # dask is checked before the arguments are.
    with pytest.raises(ImportError, match=r"pyinterp\[dask\]"):
        getattr(pyinterp.dask, helper)(*[np.zeros(3)] * nargs)

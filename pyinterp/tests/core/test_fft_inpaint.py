# Copyright (c) 2025 CNES
"""Tests for FFT inpaint routines in core.fill.

These tests check that the inpainting functions fill NaN values and
preserve the input dtype for float32 and float64 variants.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyinterp import core
import pyinterp.tests as tests

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike


def _run_inpaint_and_check(func: Callable, dtype: DTypeLike) -> None:
    ds = tests.load_grid2d()
    arr = np.copy(ds.mss.values[::5, ::5]).astype(dtype)

    # Mask of values to be inpainted
    missing = np.isnan(arr)
    assert missing.any(), 'Test dataset must contain missing values'

    # Run the inpainting (keep iterations modest to keep test fast)
    result = func(
        arr,
        first_guess=core.fill.FirstGuess.ZonalAverage,
        is_circle=True,
        max_iterations=10,
        epsilon=1e-4,
        sigma=10,
        num_threads=0,
    )

    # Function should return a tuple (iterations, residual)
    assert isinstance(result, tuple) and len(result) == 2
    iterations, residual = result
    assert isinstance(iterations, int)
    assert isinstance(residual, float)

    # All previously-missing values must be filled with finite numbers
    assert np.all(np.isfinite(arr[missing]))

    # dtype must be preserved
    assert arr.dtype == np.dtype(dtype)


def test_fft_inpaint_float64() -> None:
    """Test FFT inpaint for float64 data."""
    _run_inpaint_and_check(core.fill.fft_inpaint_float64, np.float64)


def test_fft_inpaint_float32() -> None:
    """Test FFT inpaint for float32 data."""
    _run_inpaint_and_check(core.fill.fft_inpaint_float32, np.float32)

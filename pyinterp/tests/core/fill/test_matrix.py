# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for matrix fill function."""

from __future__ import annotations

import numpy as np
import pytest

from ....core import fill


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matrix(dtype: np.dtype) -> None:
    """Test matrix fill function."""
    # Row interpolation: interior NaNs are linearly interpolated
    m = np.array(
        [
            [0.0, np.nan, np.nan, 3.0, 4.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=dtype,
    )

    fill.matrix(m, np.nan)

    # first row should be interpolated to [0,1,2,3,4]
    expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=dtype)
    np.testing.assert_allclose(m[0], expected)
    np.testing.assert_allclose(m[1], expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matrix_single_valid_point(dtype: np.dtype) -> None:
    """Test matrix fill function with single valid point."""
    # Single valid point should propagate its value to the whole row
    m = np.array([[np.nan, 2.0, np.nan]], dtype=dtype)

    fill.matrix(m, np.nan)

    expected = np.array([2.0, 2.0, 2.0], dtype=dtype)
    np.testing.assert_allclose(m[0], expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_vector(dtype: np.dtype) -> None:
    """Test vector fill function."""
    a = np.array([1.0, np.nan, 3.0], dtype=dtype)

    fill.vector(a, np.nan)

    expected = np.array([1.0, 2.0, 3.0], dtype=dtype)
    np.testing.assert_allclose(a, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matrix_with_sentinel_fillvalue(dtype: np.dtype) -> None:
    """Test matrix fill function with sentinel fill value."""
    # Use a sentinel fill value (non-NaN) and ensure equality masking works
    sentinel = -9999.0
    m = np.array([[0.0, sentinel, 2.0]], dtype=dtype)

    fill.matrix(m, sentinel)

    # interior sentinel should be interpolated between 0 and 2 => 1
    expected = np.array([0.0, 1.0, 2.0], dtype=dtype)
    np.testing.assert_allclose(m[0], expected)

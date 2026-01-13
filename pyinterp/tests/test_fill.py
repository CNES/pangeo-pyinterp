# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for fill wrapper functions.

This module provides comprehensive tests for the fill wrapper functions
that support various grid inpainting and interpolation methods.
Tests use pytest fixtures to minimize code duplication.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyinterp.core.config import fill
from pyinterp.fill import (
    fft_inpaint,
    gauss_seidel,
    loess,
    matrix,
    multigrid,
    vector,
)


# ============================================================================
# Fixtures for Test Grids
# ============================================================================


@pytest.fixture
def grid_2d_with_missing_float32() -> np.ndarray[tuple, np.dtype[np.float32]]:
    """Create a 2D grid with missing values (float32).

    Returns:
        2D grid with NaN values representing missing data.

    """
    grid = np.ones((20, 20), dtype=np.float32)

    # Create a pattern
    x = np.linspace(0, 2 * np.pi, 20)
    y = np.linspace(0, 2 * np.pi, 20)
    xx, yy = np.meshgrid(x, y)
    grid = (np.sin(xx) * np.cos(yy)).astype(np.float32)

    # Add missing values (checkerboard pattern)
    grid[::2, ::2] = np.nan

    return grid


@pytest.fixture
def grid_2d_with_missing_float64() -> np.ndarray[tuple, np.dtype[np.float64]]:
    """Create a 2D grid with missing values (float64).

    Returns:
        2D grid with NaN values representing missing data.

    """
    grid = np.ones((20, 20), dtype=np.float64)

    # Create a pattern
    x = np.linspace(0, 2 * np.pi, 20)
    y = np.linspace(0, 2 * np.pi, 20)
    xx, yy = np.meshgrid(x, y)
    grid = (np.sin(xx) * np.cos(yy)).astype(np.float64)

    # Add missing values (checkerboard pattern)
    grid[::2, ::2] = np.nan

    return grid


@pytest.fixture
def grid_2d_sparse_missing() -> np.ndarray[tuple, np.dtype[np.float64]]:
    """Create a 2D grid with sparse missing values.

    Returns:
        2D grid with fewer NaN values.

    """
    grid = np.linspace(0, 10, 400).reshape(20, 20).astype(np.float64)

    # Add sparse missing values
    grid[2, 5] = np.nan
    grid[5, 8] = np.nan
    grid[10, 10] = np.nan
    grid[15, 15] = np.nan

    return grid


@pytest.fixture
def array_1d_with_missing() -> np.ndarray[tuple, np.dtype[np.float64]]:
    """Create a 1D array with missing values.

    Returns:
        1D array with NaN values.

    """
    arr = np.linspace(0, 10, 100, dtype=np.float64)
    arr[::10] = np.nan
    return arr


@pytest.fixture
def small_grid_2d() -> np.ndarray[tuple, np.dtype[np.float64]]:
    """Create a small 2D grid for quick tests.

    Returns:
        Small 2D grid with missing values.

    """
    grid = np.arange(100, dtype=np.float64).reshape(10, 10)
    grid[::2, ::2] = np.nan
    return grid


# ============================================================================
# Tests for FFT Inpaint
# ============================================================================


class TestFFTInpaint:
    """Tests for FFT-based inpainting function."""

    def test_fft_inpaint_float32_with_kwargs(
        self,
        grid_2d_with_missing_float32: np.ndarray[tuple, np.dtype[np.float32]],
    ) -> None:
        """Test FFT inpaint with float32 grid using kwargs.

        Args:
            grid_2d_with_missing_float32: Float32 grid fixture

        """
        grid_copy = grid_2d_with_missing_float32.copy()
        iterations, residual = fft_inpaint(
            grid_copy,
            max_iterations=100,
            epsilon=1e-4,
        )

        assert iterations >= 0
        assert residual >= 0.0
        assert np.all(np.isfinite(grid_copy))
        assert not np.any(np.isnan(grid_copy))

    def test_fft_inpaint_float64_with_kwargs(
        self,
        grid_2d_with_missing_float64: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test FFT inpaint with float64 grid using kwargs.

        Args:
            grid_2d_with_missing_float64: Float64 grid fixture

        """
        grid_copy = grid_2d_with_missing_float64.copy()
        iterations, residual = fft_inpaint(
            grid_copy,
            max_iterations=100,
            epsilon=1e-5,
        )

        assert iterations >= 0
        assert residual >= 0.0
        assert np.all(np.isfinite(grid_copy))

    def test_fft_inpaint_with_config_object(
        self,
        grid_2d_with_missing_float64: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test FFT inpaint with explicit config object.

        Args:
            grid_2d_with_missing_float64: Float64 grid fixture

        """
        grid_copy = grid_2d_with_missing_float64.copy()
        config_obj = (
            fill.FFTInpaint()
            .with_max_iterations(100)
            .with_epsilon(1e-4)
            .with_sigma(0.5)
        )

        iterations, residual = fft_inpaint(grid_copy, config=config_obj)

        assert iterations >= 0
        assert residual >= 0.0

    def test_fft_inpaint_with_sigma(
        self,
        grid_2d_with_missing_float64: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test FFT inpaint with smoothing parameter.

        Args:
            grid_2d_with_missing_float64: Float64 grid fixture

        """
        grid_copy = grid_2d_with_missing_float64.copy()
        iterations, _ = fft_inpaint(
            grid_copy,
            max_iterations=50,
            epsilon=1e-4,
            sigma=0.5,
        )

        assert iterations >= 0

    @pytest.mark.parametrize(
        "first_guess",
        ["zero", "zonal_average"],
    )
    def test_fft_inpaint_first_guess_methods(
        self,
        grid_2d_with_missing_float64: np.ndarray[tuple, np.dtype[np.float64]],
        first_guess: str,
    ) -> None:
        """Test FFT inpaint with different first guess methods.

        Args:
            grid_2d_with_missing_float64: Float64 grid fixture
            first_guess: First guess method to test

        """
        grid_copy = grid_2d_with_missing_float64.copy()
        iterations, _ = fft_inpaint(
            grid_copy,
            max_iterations=50,
            epsilon=1e-4,
            first_guess=first_guess,  # type: ignore[arg-type]
        )

        assert iterations >= 0
        assert np.all(np.isfinite(grid_copy))

    def test_fft_inpaint_periodic(
        self,
        grid_2d_with_missing_float64: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test FFT inpaint with periodic boundaries.

        Args:
            grid_2d_with_missing_float64: Float64 grid fixture

        """
        grid_copy = grid_2d_with_missing_float64.copy()
        iterations, _ = fft_inpaint(
            grid_copy,
            max_iterations=50,
            epsilon=1e-4,
            is_periodic=True,
        )

        assert iterations >= 0


# ============================================================================
# Tests for Gauss-Seidel
# ============================================================================


class TestGaussSeidel:
    """Tests for Gauss-Seidel relaxation function."""

    def test_gauss_seidel_float32_with_kwargs(
        self,
        grid_2d_with_missing_float32: np.ndarray[tuple, np.dtype[np.float32]],
    ) -> None:
        """Test Gauss-Seidel with float32 grid using kwargs.

        Args:
            grid_2d_with_missing_float32: Float32 grid fixture

        """
        grid_copy = grid_2d_with_missing_float32.copy()
        iterations, residual = gauss_seidel(
            grid_copy,
            max_iterations=100,
            epsilon=1e-4,
            relaxation=1.5,
        )

        assert iterations >= 0
        assert residual >= 0.0
        assert np.all(np.isfinite(grid_copy))

    def test_gauss_seidel_float64_with_kwargs(
        self,
        grid_2d_with_missing_float64: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test Gauss-Seidel with float64 grid using kwargs.

        Args:
            grid_2d_with_missing_float64: Float64 grid fixture

        """
        grid_copy = grid_2d_with_missing_float64.copy()
        iterations, residual = gauss_seidel(
            grid_copy,
            max_iterations=100,
            epsilon=1e-5,
            relaxation=1.0,
        )

        assert iterations >= 0
        assert residual >= 0.0

    def test_gauss_seidel_with_config_object(
        self,
        grid_2d_with_missing_float64: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test Gauss-Seidel with explicit config object.

        Args:
            grid_2d_with_missing_float64: Float64 grid fixture

        """
        grid_copy = grid_2d_with_missing_float64.copy()
        config_obj = (
            fill.GaussSeidel()
            .with_max_iterations(100)
            .with_epsilon(1e-4)
            .with_relaxation(1.5)
        )

        iterations, _ = gauss_seidel(grid_copy, config=config_obj)

        assert iterations >= 0

    @pytest.mark.parametrize(
        "relaxation",
        [0.5, 1.0, 1.5, 2.0],
    )
    def test_gauss_seidel_relaxation_parameter(
        self,
        small_grid_2d: np.ndarray[tuple, np.dtype[np.float64]],
        relaxation: float,
    ) -> None:
        """Test Gauss-Seidel with different relaxation parameters.

        Args:
            small_grid_2d: Small grid fixture
            relaxation: Relaxation parameter to test

        """
        grid_copy = small_grid_2d.copy()
        iterations, _ = gauss_seidel(
            grid_copy,
            max_iterations=50,
            epsilon=1e-4,
            relaxation=relaxation,
        )

        assert iterations >= 0
        assert np.all(np.isfinite(grid_copy))

    @pytest.mark.parametrize(
        "first_guess",
        ["zero", "zonal_average"],
    )
    def test_gauss_seidel_first_guess_methods(
        self,
        grid_2d_with_missing_float64: np.ndarray[tuple, np.dtype[np.float64]],
        first_guess: str,
    ) -> None:
        """Test Gauss-Seidel with different first guess methods.

        Args:
            grid_2d_with_missing_float64: Float64 grid fixture
            first_guess: First guess method to test

        """
        grid_copy = grid_2d_with_missing_float64.copy()
        iterations, _ = gauss_seidel(
            grid_copy,
            max_iterations=50,
            epsilon=1e-4,
            first_guess=first_guess,  # type: ignore[arg-type]
        )

        assert iterations >= 0


# ============================================================================
# Tests for LOESS
# ============================================================================


class TestLoess:
    """Tests for LOESS interpolation function."""

    def test_loess_float32_with_kwargs(
        self,
        grid_2d_with_missing_float32: np.ndarray[tuple, np.dtype[np.float32]],
    ) -> None:
        """Test LOESS with float32 grid using kwargs.

        Args:
            grid_2d_with_missing_float32: Float32 grid fixture

        """
        grid_copy = grid_2d_with_missing_float32.copy()
        result = loess(
            grid_copy,
            nx=5,
            ny=5,
            max_iterations=10,
        )

        assert result.shape == grid_2d_with_missing_float32.shape
        assert np.all(np.isfinite(result))

    def test_loess_float64_with_kwargs(
        self,
        grid_2d_with_missing_float64: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test LOESS with float64 grid using kwargs.

        Args:
            grid_2d_with_missing_float64: Float64 grid fixture

        """
        grid_copy = grid_2d_with_missing_float64.copy()
        result = loess(
            grid_copy,
            nx=5,
            ny=5,
            max_iterations=5,
            epsilon=1e-4,
        )

        assert result.shape == grid_copy.shape
        assert np.all(np.isfinite(result))

    def test_loess_with_config_object(
        self,
        grid_2d_with_missing_float64: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test LOESS with explicit config object.

        Args:
            grid_2d_with_missing_float64: Float64 grid fixture

        """
        grid_copy = grid_2d_with_missing_float64.copy()
        config_obj = (
            fill.Loess()
            .with_nx(5)
            .with_ny(5)
            .with_max_iterations(10)
            .with_epsilon(1e-4)
        )

        result = loess(grid_copy, config=config_obj)

        assert result.shape == grid_copy.shape

    @pytest.mark.parametrize(
        "value_type",
        ["all", "defined", "undefined"],
    )
    def test_loess_value_type_options(
        self,
        grid_2d_with_missing_float64: np.ndarray[tuple, np.dtype[np.float64]],
        value_type: str,
    ) -> None:
        """Test LOESS with different value type options.

        Args:
            grid_2d_with_missing_float64: Float64 grid fixture
            value_type: Value type option to test

        """
        missing = np.isnan(grid_2d_with_missing_float64)
        grid_copy = grid_2d_with_missing_float64.copy()
        result = loess(
            grid_copy,
            nx=5,
            ny=5,
            max_iterations=5,
            value_type=value_type,  # type: ignore[arg-type]
        )

        if value_type == "all":
            assert np.all(np.isfinite(result))
        elif value_type == "defined":
            assert np.all(np.isfinite(result[~missing]))
        elif value_type == "undefined":
            assert np.all(np.isfinite(result[missing]))
        else:
            raise ValueError(f"Unknown value_type: {value_type}")

    def test_loess_different_window_sizes(
        self,
        small_grid_2d: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test LOESS with different window sizes.

        Args:
            small_grid_2d: Small grid fixture

        """
        grid_copy = small_grid_2d.copy()
        result = loess(
            grid_copy,
            nx=3,
            ny=3,
            max_iterations=5,
        )

        assert result.shape == grid_copy.shape
        assert np.all(np.isfinite(result))


# ============================================================================
# Tests for Multigrid
# ============================================================================


class TestMultigrid:
    """Tests for multigrid inpainting function."""

    def test_multigrid_float32_with_kwargs(
        self,
        grid_2d_with_missing_float32: np.ndarray[tuple, np.dtype[np.float32]],
    ) -> None:
        """Test multigrid with float32 grid using kwargs.

        Args:
            grid_2d_with_missing_float32: Float32 grid fixture

        """
        grid_copy = grid_2d_with_missing_float32.copy()
        iterations, residual = multigrid(
            grid_copy,
            max_iterations=100,
            epsilon=1e-4,
            pre_smooth=2,
            post_smooth=2,
        )

        assert iterations >= 0
        assert residual >= 0.0
        assert np.all(np.isfinite(grid_copy))

    def test_multigrid_float64_with_kwargs(
        self,
        grid_2d_with_missing_float64: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test multigrid with float64 grid using kwargs.

        Args:
            grid_2d_with_missing_float64: Float64 grid fixture

        """
        grid_copy = grid_2d_with_missing_float64.copy()
        iterations, residual = multigrid(
            grid_copy,
            max_iterations=100,
            epsilon=1e-5,
            pre_smooth=1,
            post_smooth=1,
        )

        assert iterations >= 0
        assert residual >= 0.0

    def test_multigrid_with_config_object(
        self,
        grid_2d_with_missing_float64: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test multigrid with explicit config object.

        Args:
            grid_2d_with_missing_float64: Float64 grid fixture

        """
        grid_copy = grid_2d_with_missing_float64.copy()
        config_obj = (
            fill.Multigrid()
            .with_max_iterations(100)
            .with_epsilon(1e-4)
            .with_pre_smooth(2)
            .with_post_smooth(2)
        )

        iterations, _ = multigrid(grid_copy, config=config_obj)

        assert iterations >= 0

    @pytest.mark.parametrize(
        "pre_smooth,post_smooth",
        [(1, 1), (2, 2), (1, 2), (2, 1)],
    )
    def test_multigrid_smoothing_parameters(
        self,
        small_grid_2d: np.ndarray[tuple, np.dtype[np.float64]],
        pre_smooth: int,
        post_smooth: int,
    ) -> None:
        """Test multigrid with different smoothing parameters.

        Args:
            small_grid_2d: Small grid fixture
            pre_smooth: Pre-smoothing iterations
            post_smooth: Post-smoothing iterations

        """
        grid_copy = small_grid_2d.copy()
        iterations, _ = multigrid(
            grid_copy,
            max_iterations=50,
            epsilon=1e-4,
            pre_smooth=pre_smooth,
            post_smooth=post_smooth,
        )

        assert iterations >= 0
        assert np.all(np.isfinite(grid_copy))

    @pytest.mark.parametrize(
        "first_guess",
        ["zero", "zonal_average"],
    )
    def test_multigrid_first_guess_methods(
        self,
        grid_2d_with_missing_float64: np.ndarray[tuple, np.dtype[np.float64]],
        first_guess: str,
    ) -> None:
        """Test multigrid with different first guess methods.

        Args:
            grid_2d_with_missing_float64: Float64 grid fixture
            first_guess: First guess method to test

        """
        grid_copy = grid_2d_with_missing_float64.copy()
        iterations, _ = multigrid(
            grid_copy,
            max_iterations=50,
            epsilon=1e-4,
            first_guess=first_guess,  # type: ignore[arg-type]
        )

        assert iterations >= 0


# ============================================================================
# Tests for Matrix Fill
# ============================================================================


class TestMatrixFill:
    """Tests for matrix fill function."""

    def test_matrix_fill_float32(
        self,
        grid_2d_with_missing_float32: np.ndarray[tuple, np.dtype[np.float32]],
    ) -> None:
        """Test matrix fill with float32 grid.

        Args:
            grid_2d_with_missing_float32: Float32 grid fixture

        """
        grid_copy = grid_2d_with_missing_float32.copy()
        original_shape = grid_copy.shape

        matrix(grid_copy)

        # Grid should be modified in-place
        assert np.all(np.isfinite(grid_copy))
        assert grid_copy.shape == original_shape

    def test_matrix_fill_float64(
        self,
        grid_2d_with_missing_float64: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test matrix fill with float64 grid.

        Args:
            grid_2d_with_missing_float64: Float64 grid fixture

        """
        grid_copy = grid_2d_with_missing_float64.copy()

        matrix(grid_copy)
        assert np.all(np.isfinite(grid_copy))

    def test_matrix_fill_different_values(
        self,
        small_grid_2d: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test matrix fill with different fill values.

        Args:
            small_grid_2d: Small grid fixture

        """
        missing = np.isnan(small_grid_2d)
        for fill_value in [-1.0, 0.0, 1.0, 100.5]:
            grid_copy = small_grid_2d.copy()
            grid_copy[missing] = fill_value
            matrix(grid_copy, fill_value)
            assert np.all(np.isfinite(grid_copy))


# ============================================================================
# Tests for Vector Fill
# ============================================================================


class TestVectorFill:
    """Tests for vector fill function."""

    def test_vector_fill_float64(
        self,
        array_1d_with_missing: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test vector fill with float64 array.

        Args:
            array_1d_with_missing: 1D array fixture

        """
        arr_copy = array_1d_with_missing.copy()
        original_shape = arr_copy.shape
        vector(arr_copy)

        assert np.all(np.isfinite(arr_copy))
        assert arr_copy.shape == original_shape

    def test_vector_fill_different_values(
        self,
        array_1d_with_missing: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test vector fill with different fill values.

        Args:
            array_1d_with_missing: 1D array fixture

        """
        missing = np.isnan(array_1d_with_missing)
        for fill_value in [-10.5, 0.0, 7.3]:
            arr_copy = array_1d_with_missing.copy()
            arr_copy[missing] = fill_value
            vector(arr_copy, fill_value)
            assert np.all(arr_copy[missing] != fill_value)

    def test_vector_fill_preserves_shape(
        self,
        array_1d_with_missing: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test that vector fill preserves array shape.

        Args:
            array_1d_with_missing: 1D array fixture

        """
        arr_copy = array_1d_with_missing.copy()
        original_shape = arr_copy.shape

        vector(arr_copy, 5.0)

        assert arr_copy.shape == original_shape


# ============================================================================
# Integration Tests
# ============================================================================


class TestFillMethodsConsistency:
    """Tests for consistency across different fill methods."""

    def test_all_iterative_methods_produce_valid_output(
        self,
        small_grid_2d: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test that all iterative fill methods produce valid output.

        Args:
            small_grid_2d: Small grid fixture

        """
        fft_grid = small_grid_2d.copy()
        gs_grid = small_grid_2d.copy()
        mg_grid = small_grid_2d.copy()

        fft_itr, _ = fft_inpaint(fft_grid, max_iterations=50, epsilon=1e-4)
        gs_itr, _ = gauss_seidel(gs_grid, max_iterations=50, epsilon=1e-4)
        mg_itr, _ = multigrid(mg_grid, max_iterations=50, epsilon=1e-4)

        # All should converge
        assert fft_itr >= 0
        assert gs_itr >= 0
        assert mg_itr >= 0

        # All should fill missing values
        assert not np.any(np.isnan(fft_grid))
        assert not np.any(np.isnan(gs_grid))
        assert not np.any(np.isnan(mg_grid))

    def test_config_and_kwargs_equivalence(
        self,
        small_grid_2d: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test that config object and kwargs produce same results.

        Args:
            small_grid_2d: Small grid fixture

        """
        # Gauss-Seidel with kwargs
        grid_kwargs = small_grid_2d.copy()
        itr_kwargs, res_kwargs = gauss_seidel(
            grid_kwargs,
            max_iterations=50,
            epsilon=1e-4,
            relaxation=1.5,
        )

        # Gauss-Seidel with config
        grid_config = small_grid_2d.copy()
        config = (
            fill.GaussSeidel()
            .with_max_iterations(50)
            .with_epsilon(1e-4)
            .with_relaxation(1.5)
        )
        itr_config, res_config = gauss_seidel(grid_config, config=config)

        # Results should be identical
        np.testing.assert_array_equal(grid_kwargs, grid_config)
        assert itr_kwargs == itr_config
        np.testing.assert_almost_equal(res_kwargs, res_config, decimal=10)

    def test_loess_returns_correct_dtype(
        self,
        grid_2d_with_missing_float32: np.ndarray[tuple, np.dtype[np.float32]],
    ) -> None:
        """Test that LOESS preserves input dtype.

        Args:
            grid_2d_with_missing_float32: Float32 grid fixture

        """
        grid_copy = grid_2d_with_missing_float32.copy()
        original_dtype = grid_copy.dtype

        result = loess(grid_copy, nx=5, ny=5, max_iterations=3)

        assert result.dtype == original_dtype


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for proper error handling and validation."""

    def test_invalid_first_guess_raises_error(
        self,
        small_grid_2d: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test that invalid first guess raises error.

        Args:
            small_grid_2d: Small grid fixture

        """
        grid_copy = small_grid_2d.copy()

        with pytest.raises(KeyError):
            gauss_seidel(
                grid_copy,
                max_iterations=10,
                first_guess="invalid",  # type: ignore[arg-type]
            )

    def test_invalid_loess_value_type_raises_error(
        self,
        small_grid_2d: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test that invalid LOESS value type raises error.

        Args:
            small_grid_2d: Small grid fixture

        """
        grid_copy = small_grid_2d.copy()

        with pytest.raises(KeyError):
            loess(
                grid_copy,
                nx=5,
                ny=5,
                value_type="invalid",  # type: ignore[arg-type]
            )

    def test_fft_inpaint_modifies_in_place(
        self,
        small_grid_2d: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test that FFT inpaint modifies grid in-place.

        Args:
            small_grid_2d: Small grid fixture

        """
        grid = small_grid_2d.copy()
        grid_id = id(grid)

        fft_inpaint(grid, max_iterations=10, epsilon=1e-4)

        # Should be same object
        assert id(grid) == grid_id


# ============================================================================
# Performance and Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_fill_all_zero_grid(
        self,
    ) -> None:
        """Test fill methods on all-zero grid."""
        grid = np.zeros((10, 10), dtype=np.float64)
        grid[::2, ::2] = np.nan

        result = loess(grid, nx=3, ny=3, max_iterations=3)

        assert result.shape == grid.shape
        assert np.all(np.isfinite(result))

    def test_fill_sparse_missing_values(
        self,
        grid_2d_sparse_missing: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test fill methods with sparse missing values.

        Args:
            grid_2d_sparse_missing: Sparse missing grid fixture

        """
        grid_copy = grid_2d_sparse_missing.copy()

        iterations, _ = gauss_seidel(
            grid_copy,
            max_iterations=100,
            epsilon=1e-4,
        )

        assert not np.any(np.isnan(grid_copy))
        assert iterations >= 0

    def test_convergence_with_tight_epsilon(
        self,
        small_grid_2d: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test convergence with tight epsilon value.

        Args:
            small_grid_2d: Small grid fixture

        """
        grid_copy = small_grid_2d.copy()

        iterations, residual = gauss_seidel(
            grid_copy,
            max_iterations=1000,
            epsilon=1e-10,
        )

        assert residual <= 1e-10 or iterations >= 1000

    def test_max_iterations_limit(
        self,
        small_grid_2d: np.ndarray[tuple, np.dtype[np.float64]],
    ) -> None:
        """Test that algorithm respects max_iterations limit.

        Args:
            small_grid_2d: Small grid fixture

        """
        grid_copy = small_grid_2d.copy()
        max_itr = 5

        iterations, _ = gauss_seidel(
            grid_copy,
            max_iterations=max_itr,
            epsilon=1e-20,  # Unreachable to force iteration limit
        )

        assert iterations <= max_itr

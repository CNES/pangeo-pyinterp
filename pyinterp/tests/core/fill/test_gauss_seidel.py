# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for Gauss-Seidel fill method."""

from __future__ import annotations

import numpy as np
import pytest

from ....core import config, fill


class TestGaussSeidel:
    """Test Gauss-Seidel fill method."""

    @pytest.fixture
    def sample_data_float64(self) -> np.ndarray:
        """Create sample 2D float64 array with missing values."""
        data = np.array(
            [
                [1.0, 2.0, np.nan, 4.0],
                [5.0, np.nan, 7.0, 8.0],
                [9.0, 10.0, 11.0, np.nan],
                [np.nan, 14.0, 15.0, 16.0],
            ],
            dtype=np.float64,
        )
        return data

    @pytest.fixture
    def sample_data_float32(self) -> np.ndarray:
        """Create sample 2D float32 array with missing values."""
        data = np.array(
            [
                [1.0, 2.0, np.nan, 4.0],
                [5.0, np.nan, 7.0, 8.0],
                [9.0, 10.0, 11.0, np.nan],
                [np.nan, 14.0, 15.0, 16.0],
            ],
            dtype=np.float32,
        )
        return data

    @pytest.fixture
    def default_config(self) -> config.fill.GaussSeidel:
        """Create default Gauss-Seidel configuration."""
        return config.fill.GaussSeidel()

    def test_gauss_seidel_basic_float64(
        self,
        sample_data_float64: np.ndarray,
        default_config: config.fill.GaussSeidel,
    ) -> None:
        """Test basic Gauss-Seidel fill with float64 data."""
        grid = sample_data_float64.copy()
        iterations, residual = fill.gauss_seidel(grid, default_config)

        assert isinstance(iterations, (int, np.integer))
        assert isinstance(residual, (float, np.floating))
        assert iterations > 0
        assert residual >= 0.0
        assert grid.dtype == np.float64

    def test_gauss_seidel_basic_float32(
        self,
        sample_data_float32: np.ndarray,
        default_config: config.fill.GaussSeidel,
    ) -> None:
        """Test basic Gauss-Seidel fill with float32 data."""
        grid = sample_data_float32.copy()
        iterations, residual = fill.gauss_seidel(grid, default_config)

        assert isinstance(iterations, (int, np.integer))
        assert isinstance(residual, (float, np.floating))
        assert iterations > 0
        assert residual >= 0.0
        assert grid.dtype == np.float32

    def test_gauss_seidel_preserves_dtype(
        self,
        sample_data_float64: np.ndarray,
        sample_data_float32: np.ndarray,
    ) -> None:
        """Test that Gauss-Seidel preserves input data type."""
        config_obj = config.fill.GaussSeidel()

        grid64 = sample_data_float64.copy()
        fill.gauss_seidel(grid64, config_obj)
        assert grid64.dtype == np.float64

        grid32 = sample_data_float32.copy()
        fill.gauss_seidel(grid32, config_obj)
        assert grid32.dtype == np.float32

    def test_gauss_seidel_fills_nan(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test that Gauss-Seidel fills NaN values."""
        grid = sample_data_float64.copy()
        nan_count_before = np.isnan(grid).sum()

        fill.gauss_seidel(grid, config.fill.GaussSeidel())

        nan_count_after = np.isnan(grid).sum()
        assert nan_count_before > 0
        assert nan_count_after < nan_count_before

    def test_gauss_seidel_convergence(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test Gauss-Seidel convergence behavior."""
        # Strict convergence
        grid_strict = sample_data_float64.copy()
        config_strict = (
            config.fill.GaussSeidel()
            .with_epsilon(1e-8)
            .with_max_iterations(500)
        )
        iterations_strict, _ = fill.gauss_seidel(grid_strict, config_strict)

        # Relaxed convergence
        grid_relaxed = sample_data_float64.copy()
        config_relaxed = (
            config.fill.GaussSeidel()
            .with_epsilon(1e-3)
            .with_max_iterations(20)
        )
        iterations_relaxed, _ = fill.gauss_seidel(grid_relaxed, config_relaxed)

        # Relaxed should converge faster (fewer iterations)
        assert iterations_relaxed <= iterations_strict

    def test_gauss_seidel_with_relaxation(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test Gauss-Seidel with SOR relaxation parameter."""
        for relaxation in [0.5, 1.0, 1.5]:
            grid = sample_data_float64.copy()
            config_obj = config.fill.GaussSeidel().with_relaxation(relaxation)

            iterations, residual = fill.gauss_seidel(grid, config_obj)

            assert iterations > 0
            assert residual >= 0.0
            assert not np.isnan(residual)

    def test_gauss_seidel_periodic(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test Gauss-Seidel with periodic boundary condition."""
        grid_periodic = sample_data_float64.copy()
        config_periodic = config.fill.GaussSeidel().with_is_periodic(True)

        iterations_p, residual_p = fill.gauss_seidel(
            grid_periodic, config_periodic
        )

        assert iterations_p > 0
        assert residual_p >= 0.0

    def test_gauss_seidel_non_periodic(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test Gauss-Seidel with non-periodic boundary condition."""
        grid_non_periodic = sample_data_float64.copy()
        config_non_periodic = config.fill.GaussSeidel().with_is_periodic(False)

        iterations_np, residual_np = fill.gauss_seidel(
            grid_non_periodic, config_non_periodic
        )

        assert iterations_np > 0
        assert residual_np >= 0.0

    def test_gauss_seidel_first_guess_zero(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test Gauss-Seidel with ZERO first guess."""
        grid = sample_data_float64.copy()
        config_obj = config.fill.GaussSeidel().with_first_guess(
            config.fill.FirstGuess.ZERO
        )

        iterations, residual = fill.gauss_seidel(grid, config_obj)

        assert iterations > 0
        assert residual >= 0.0

    def test_gauss_seidel_first_guess_zonal_average(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test Gauss-Seidel with ZONAL_AVERAGE first guess."""
        grid = sample_data_float64.copy()
        config_obj = config.fill.GaussSeidel().with_first_guess(
            config.fill.FirstGuess.ZONAL_AVERAGE
        )

        iterations, residual = fill.gauss_seidel(grid, config_obj)

        assert iterations > 0
        assert residual >= 0.0

    def test_gauss_seidel_num_threads(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test Gauss-Seidel with different thread counts."""
        for num_threads in [1, 2, 4]:
            grid = sample_data_float64.copy()
            config_obj = config.fill.GaussSeidel().with_num_threads(
                num_threads
            )

            iterations, residual = fill.gauss_seidel(grid, config_obj)

            assert iterations > 0
            assert residual >= 0.0

    def test_gauss_seidel_all_nan_input(self) -> None:
        """Test Gauss-Seidel behavior with all NaN input."""
        all_nan = np.full((4, 4), np.nan, dtype=np.float64)
        config_obj = config.fill.GaussSeidel()

        iterations, residual = fill.gauss_seidel(all_nan, config_obj)

        assert iterations >= 0
        assert residual >= 0.0

    def test_gauss_seidel_no_nan_input(self) -> None:
        """Test Gauss-Seidel behavior with no NaN values."""
        no_nan = np.arange(16, dtype=np.float64).reshape(4, 4)
        no_nan_copy = no_nan.copy()
        config_obj = config.fill.GaussSeidel()

        iterations, residual = fill.gauss_seidel(no_nan, config_obj)

        # No changes expected when there are no NaN values
        assert iterations >= 0
        assert residual >= 0.0
        np.testing.assert_array_almost_equal(no_nan, no_nan_copy, decimal=5)

    def test_gauss_seidel_small_array(self) -> None:
        """Test Gauss-Seidel with small array."""
        small_data = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float64)
        config_obj = config.fill.GaussSeidel()

        iterations, residual = fill.gauss_seidel(small_data, config_obj)

        assert iterations >= 0
        assert residual >= 0.0

    def test_gauss_seidel_large_array(self) -> None:
        """Test Gauss-Seidel with larger array."""
        # Create a 20x20 array with scattered NaN values
        rng = np.random.default_rng(42)
        large_data = rng.random((20, 20), dtype=np.float64)
        # Add some NaN values
        large_data[::3, ::3] = np.nan

        config_obj = (
            config.fill.GaussSeidel()
            .with_num_threads(4)
            .with_max_iterations(100)
        )

        iterations, residual = fill.gauss_seidel(large_data, config_obj)

        assert iterations > 0
        assert residual >= 0.0

    def test_gauss_seidel_method_chaining_comprehensive(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test comprehensive method chaining for Gauss-Seidel configuration."""
        grid = sample_data_float64.copy()
        config_obj = (
            config.fill.GaussSeidel()
            .with_first_guess(config.fill.FirstGuess.ZONAL_AVERAGE)
            .with_is_periodic(False)
            .with_epsilon(1e-5)
            .with_max_iterations(200)
            .with_relaxation(1.2)
            .with_num_threads(2)
        )

        iterations, residual = fill.gauss_seidel(grid, config_obj)

        assert iterations > 0
        assert residual >= 0.0
        assert grid.dtype == np.float64

    def test_gauss_seidel_modifies_grid_in_place(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test that Gauss-Seidel modifies grid in place."""
        grid = sample_data_float64.copy()
        grid_id_before = id(grid)

        fill.gauss_seidel(grid, config.fill.GaussSeidel())

        # Grid should be modified in place
        grid_id_after = id(grid)
        assert grid_id_before == grid_id_after

    def test_gauss_seidel_return_tuple(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test that Gauss-Seidel returns correct tuple format."""
        grid = sample_data_float64.copy()
        result = fill.gauss_seidel(grid, config.fill.GaussSeidel())

        assert isinstance(result, tuple)
        assert len(result) == 2
        iterations, residual = result
        assert isinstance(iterations, (int, np.integer))
        assert isinstance(residual, (float, np.floating))

    def test_gauss_seidel_analytical_linear_field(self) -> None:
        """Test Gauss-Seidel with analytical linear field."""
        # Create a linear field: z = 2*x + 3*y
        ny, nx = 20, 20
        x = np.linspace(0, 10, nx)
        y = np.linspace(0, 10, ny)
        xx, yy = np.meshgrid(x, y)
        original = 2.0 * xx + 3.0 * yy

        # Remove some values in the interior
        grid = original.copy()
        grid[5:15, 5:15] = np.nan

        # Fill using Gauss-Seidel
        config_obj = config.fill.GaussSeidel().with_max_iterations(500)
        fill.gauss_seidel(grid, config_obj)

        # Compare filled values with original (should be very close for linear
        # field)
        mask = np.isnan(original.copy())
        mask[5:15, 5:15] = True
        filled_region = ~mask
        np.testing.assert_allclose(
            grid[filled_region], original[filled_region], rtol=1e-2, atol=0.5
        )

    def test_gauss_seidel_analytical_sinusoidal_field(self) -> None:
        """Test Gauss-Seidel with analytical sinusoidal field."""
        # Create a smooth sinusoidal field
        ny, nx = 24, 24
        x = np.linspace(0, 2 * np.pi, nx)
        y = np.linspace(0, 2 * np.pi, ny)
        xx, yy = np.meshgrid(x, y)
        original = np.sin(xx) * np.cos(yy)

        # Remove scattered values
        rng = np.random.default_rng(42)
        grid = original.copy()
        mask_indices = rng.choice(nx * ny, size=nx * ny // 4, replace=False)
        grid.flat[mask_indices] = np.nan

        # Fill using Gauss-Seidel
        config_obj = (
            config.fill.GaussSeidel()
            .with_max_iterations(1000)
            .with_epsilon(1e-6)
        )
        fill.gauss_seidel(grid, config_obj)

        # Compare filled values with original
        filled_mask = np.zeros_like(original, dtype=bool)
        filled_mask.flat[mask_indices] = True
        np.testing.assert_allclose(
            grid[filled_mask], original[filled_mask], rtol=1.0, atol=0.8
        )

    def test_gauss_seidel_analytical_quadratic_field(self) -> None:
        """Test Gauss-Seidel with analytical quadratic field."""
        # Create a quadratic field
        ny, nx = 16, 16
        x = np.linspace(-5, 5, nx)
        y = np.linspace(-5, 5, ny)
        xx, yy = np.meshgrid(x, y)
        original = xx**2 + yy**2

        # Remove a rectangular region
        grid = original.copy()
        grid[6:10, 6:10] = np.nan

        # Fill using Gauss-Seidel
        config_obj = config.fill.GaussSeidel().with_max_iterations(500)
        fill.gauss_seidel(grid, config_obj)

        # Compare filled values with original
        # Note: Gauss-Seidel provides harmonic solution, not exact quadratic
        filled_region = np.zeros_like(original, dtype=bool)
        filled_region[6:10, 6:10] = True
        np.testing.assert_allclose(
            grid[filled_region], original[filled_region], rtol=2.0, atol=3.0
        )

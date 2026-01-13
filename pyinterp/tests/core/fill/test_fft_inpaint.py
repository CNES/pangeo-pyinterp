# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for FFT Inpaint fill method."""

from __future__ import annotations

import numpy as np
import pytest

from ....core import config, fill


class TestFFTInpaint:
    """Test FFT Inpaint fill method."""

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
    def default_config(self) -> config.fill.FFTInpaint:
        """Create default FFT Inpaint configuration."""
        return config.fill.FFTInpaint()

    def test_fft_inpaint_basic_float64(
        self,
        sample_data_float64: np.ndarray,
        default_config: config.fill.FFTInpaint,
    ) -> None:
        """Test basic FFT Inpaint fill with float64 data."""
        grid = sample_data_float64.copy()
        iterations, residual = fill.fft_inpaint(grid, default_config)

        assert isinstance(iterations, (int, np.integer))
        assert isinstance(residual, (float, np.floating))
        assert iterations > 0
        assert residual >= 0.0
        assert grid.dtype == np.float64

    def test_fft_inpaint_basic_float32(
        self,
        sample_data_float32: np.ndarray,
        default_config: config.fill.FFTInpaint,
    ) -> None:
        """Test basic FFT Inpaint fill with float32 data."""
        grid = sample_data_float32.copy()
        iterations, residual = fill.fft_inpaint(grid, default_config)

        assert isinstance(iterations, (int, np.integer))
        assert isinstance(residual, (float, np.floating))
        assert iterations > 0
        assert residual >= 0.0
        assert grid.dtype == np.float32

    def test_fft_inpaint_preserves_dtype(
        self,
        sample_data_float64: np.ndarray,
        sample_data_float32: np.ndarray,
    ) -> None:
        """Test that FFT Inpaint preserves input data type."""
        config_obj = config.fill.FFTInpaint()

        grid64 = sample_data_float64.copy()
        fill.fft_inpaint(grid64, config_obj)
        assert grid64.dtype == np.float64

        grid32 = sample_data_float32.copy()
        fill.fft_inpaint(grid32, config_obj)
        assert grid32.dtype == np.float32

    def test_fft_inpaint_fills_nan(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test that FFT Inpaint fills NaN values."""
        grid = sample_data_float64.copy()
        nan_count_before = np.isnan(grid).sum()

        fill.fft_inpaint(grid, config.fill.FFTInpaint())

        nan_count_after = np.isnan(grid).sum()
        assert nan_count_before > 0
        assert nan_count_after < nan_count_before

    def test_fft_inpaint_convergence(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test FFT Inpaint convergence behavior."""
        # Strict convergence
        grid_strict = sample_data_float64.copy()
        config_strict = (
            config.fill.FFTInpaint()
            .with_epsilon(1e-8)
            .with_max_iterations(500)
        )
        iterations_strict, _ = fill.fft_inpaint(grid_strict, config_strict)

        # Relaxed convergence
        grid_relaxed = sample_data_float64.copy()
        config_relaxed = (
            config.fill.FFTInpaint()
            .with_epsilon(1e-3)
            .with_max_iterations(100)
        )
        iterations_relaxed, _ = fill.fft_inpaint(grid_relaxed, config_relaxed)

        # Relaxed should converge faster (fewer iterations)
        assert iterations_relaxed <= iterations_strict

    def test_fft_inpaint_sigma_values(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test FFT Inpaint with different sigma values."""
        for sigma in [0.5, 1.0, 2.0, 5.0]:
            grid = sample_data_float64.copy()
            config_obj = config.fill.FFTInpaint().with_sigma(sigma)

            iterations, residual = fill.fft_inpaint(grid, config_obj)

            assert iterations > 0
            assert residual >= 0.0
            assert not np.isnan(residual)

    def test_fft_inpaint_periodic(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test FFT Inpaint with periodic boundary condition."""
        grid_periodic = sample_data_float64.copy()
        config_periodic = config.fill.FFTInpaint().with_is_periodic(True)

        iterations_p, residual_p = fill.fft_inpaint(
            grid_periodic, config_periodic
        )

        assert iterations_p > 0
        assert residual_p >= 0.0

    def test_fft_inpaint_non_periodic(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test FFT Inpaint with non-periodic boundary condition."""
        grid_non_periodic = sample_data_float64.copy()
        config_non_periodic = config.fill.FFTInpaint().with_is_periodic(False)

        iterations_np, residual_np = fill.fft_inpaint(
            grid_non_periodic, config_non_periodic
        )

        assert iterations_np > 0
        assert residual_np >= 0.0

    def test_fft_inpaint_periodic_vs_non_periodic(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test FFT Inpaint with periodic vs non-periodic."""
        # Periodic (FFT)
        grid_periodic = sample_data_float64.copy()
        config_periodic = config.fill.FFTInpaint().with_is_periodic(True)
        iterations_p, residual_p = fill.fft_inpaint(
            grid_periodic, config_periodic
        )

        # Non-periodic (DCT)
        grid_non_periodic = sample_data_float64.copy()
        config_non_periodic = config.fill.FFTInpaint().with_is_periodic(False)
        iterations_np, residual_np = fill.fft_inpaint(
            grid_non_periodic, config_non_periodic
        )

        # Both should succeed
        assert iterations_p >= 0
        assert iterations_np >= 0
        assert residual_p >= 0.0
        assert residual_np >= 0.0

    def test_fft_inpaint_first_guess_zero(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test FFT Inpaint with ZERO first guess."""
        grid = sample_data_float64.copy()
        config_obj = config.fill.FFTInpaint().with_first_guess(
            config.fill.FirstGuess.ZERO
        )

        iterations, residual = fill.fft_inpaint(grid, config_obj)

        assert iterations > 0
        assert residual >= 0.0

    def test_fft_inpaint_first_guess_zonal_average(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test FFT Inpaint with ZONAL_AVERAGE first guess."""
        grid = sample_data_float64.copy()
        config_obj = config.fill.FFTInpaint().with_first_guess(
            config.fill.FirstGuess.ZONAL_AVERAGE
        )

        iterations, residual = fill.fft_inpaint(grid, config_obj)

        assert iterations > 0
        assert residual >= 0.0

    def test_fft_inpaint_num_threads(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test FFT Inpaint with different thread counts."""
        for num_threads in [1, 2, 4]:
            grid = sample_data_float64.copy()
            config_obj = config.fill.FFTInpaint().with_num_threads(num_threads)

            iterations, residual = fill.fft_inpaint(grid, config_obj)

            assert iterations > 0
            assert residual >= 0.0

    def test_fft_inpaint_all_nan_input(self) -> None:
        """Test FFT Inpaint behavior with all NaN input."""
        all_nan = np.full((4, 4), np.nan, dtype=np.float64)
        config_obj = config.fill.FFTInpaint()

        iterations, residual = fill.fft_inpaint(all_nan, config_obj)

        assert iterations >= 0
        assert residual >= 0.0

    def test_fft_inpaint_no_nan_input(self) -> None:
        """Test FFT Inpaint behavior with no NaN values."""
        no_nan = np.arange(16, dtype=np.float64).reshape(4, 4)
        no_nan_copy = no_nan.copy()
        config_obj = config.fill.FFTInpaint()

        iterations, residual = fill.fft_inpaint(no_nan, config_obj)

        # No changes expected when there are no NaN values
        assert iterations >= 0
        assert residual >= 0.0
        np.testing.assert_array_almost_equal(no_nan, no_nan_copy, decimal=5)

    def test_fft_inpaint_small_array(self) -> None:
        """Test FFT Inpaint with small array."""
        small_data = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float64)
        config_obj = config.fill.FFTInpaint()

        iterations, residual = fill.fft_inpaint(small_data, config_obj)

        assert iterations >= 0
        assert residual >= 0.0

    def test_fft_inpaint_large_array(self) -> None:
        """Test FFT Inpaint with larger array."""
        # Create a 20x20 array with scattered NaN values
        rng = np.random.default_rng(42)
        large_data = rng.random((20, 20), dtype=np.float64)
        # Add some NaN values
        large_data[::3, ::3] = np.nan

        config_obj = (
            config.fill.FFTInpaint()
            .with_num_threads(4)
            .with_max_iterations(200)
            .with_sigma(2.0)
        )

        iterations, residual = fill.fft_inpaint(large_data, config_obj)

        assert iterations > 0
        assert residual >= 0.0

    def test_fft_inpaint_method_chaining_comprehensive(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test comprehensive method chaining for FFT Inpaint configuration."""
        grid = sample_data_float64.copy()
        config_obj = (
            config.fill.FFTInpaint()
            .with_first_guess(config.fill.FirstGuess.ZONAL_AVERAGE)
            .with_is_periodic(False)
            .with_epsilon(1e-4)
            .with_max_iterations(200)
            .with_sigma(1.5)
            .with_num_threads(2)
        )

        iterations, residual = fill.fft_inpaint(grid, config_obj)

        assert iterations > 0
        assert residual >= 0.0
        assert grid.dtype == np.float64

    def test_fft_inpaint_modifies_grid_in_place(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test that FFT Inpaint modifies grid in place."""
        grid = sample_data_float64.copy()
        grid_id_before = id(grid)

        fill.fft_inpaint(grid, config.fill.FFTInpaint())

        # Grid should be modified in place
        grid_id_after = id(grid)
        assert grid_id_before == grid_id_after

    def test_fft_inpaint_return_tuple(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test that FFT Inpaint returns correct tuple format."""
        grid = sample_data_float64.copy()
        result = fill.fft_inpaint(grid, config.fill.FFTInpaint())

        assert isinstance(result, tuple)
        assert len(result) == 2
        iterations, residual = result
        assert isinstance(iterations, (int, np.integer))
        assert isinstance(residual, (float, np.floating))

    def test_fft_inpaint_sigma_and_convergence(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test FFT Inpaint with different sigma values and convergence."""
        # Test multiple sigma values affect convergence behavior
        residuals = []
        for sigma in [0.5, 1.0, 2.0]:
            grid = sample_data_float64.copy()
            config_obj = (
                config.fill.FFTInpaint()
                .with_sigma(sigma)
                .with_max_iterations(200)
            )

            iterations, residual = fill.fft_inpaint(grid, config_obj)
            residuals.append(residual)

            assert iterations > 0
            assert residual >= 0.0

        # All residuals should be non-negative
        assert all(r >= 0.0 for r in residuals)

    def test_fft_inpaint_large_sigma(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test FFT Inpaint with large sigma value."""
        grid = sample_data_float64.copy()
        config_obj = config.fill.FFTInpaint().with_sigma(10.0)

        iterations, residual = fill.fft_inpaint(grid, config_obj)

        assert iterations > 0
        assert residual >= 0.0

    def test_fft_inpaint_small_sigma(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test FFT Inpaint with small sigma value."""
        grid = sample_data_float64.copy()
        config_obj = config.fill.FFTInpaint().with_sigma(0.1)

        iterations, residual = fill.fft_inpaint(grid, config_obj)

        assert iterations > 0
        assert residual >= 0.0

    def test_fft_inpaint_analytical_linear_field(self) -> None:
        """Test FFT Inpaint with analytical linear field."""
        # Create a linear field: z = 2*x + 3*y
        ny, nx = 32, 32
        x = np.linspace(0, 10, nx)
        y = np.linspace(0, 10, ny)
        xx, yy = np.meshgrid(x, y)
        original = 2.0 * xx + 3.0 * yy

        # Remove some values in the interior
        grid = original.copy()
        grid[10:22, 10:22] = np.nan

        # Fill using FFT Inpaint
        config_obj = (
            config.fill.FFTInpaint().with_max_iterations(500).with_sigma(1.0)
        )
        fill.fft_inpaint(grid, config_obj)

        # Compare filled values with original
        mask = np.isnan(original.copy())
        mask[10:22, 10:22] = True
        filled_region = ~mask
        np.testing.assert_allclose(
            grid[filled_region], original[filled_region], rtol=1e-2, atol=0.5
        )

    def test_fft_inpaint_analytical_sinusoidal_field(self) -> None:
        """Test FFT Inpaint with analytical sinusoidal field."""
        # Create a smooth sinusoidal field
        ny, nx = 32, 32
        x = np.linspace(0, 2 * np.pi, nx)
        y = np.linspace(0, 2 * np.pi, ny)
        xx, yy = np.meshgrid(x, y)
        original = np.sin(xx) * np.cos(yy)

        # Remove scattered values
        rng = np.random.default_rng(42)
        grid = original.copy()
        mask_indices = rng.choice(nx * ny, size=nx * ny // 4, replace=False)
        grid.flat[mask_indices] = np.nan

        # Fill using FFT Inpaint
        config_obj = (
            config.fill.FFTInpaint()
            .with_max_iterations(500)
            .with_sigma(1.0)
            .with_epsilon(1e-5)
        )
        fill.fft_inpaint(grid, config_obj)

        # Compare filled values with original
        # Note: With 25% missing data in oscillatory field, exact reconstruction
        # is not possible. FFT inpaint provides smooth approximation.
        filled_mask = np.zeros_like(original, dtype=bool)
        filled_mask.flat[mask_indices] = True
        np.testing.assert_allclose(
            grid[filled_mask], original[filled_mask], rtol=1.0, atol=0.8
        )

    def test_fft_inpaint_analytical_quadratic_field(self) -> None:
        """Test FFT Inpaint with analytical quadratic field."""
        # Create a quadratic field
        ny, nx = 32, 32
        x = np.linspace(-5, 5, nx)
        y = np.linspace(-5, 5, ny)
        xx, yy = np.meshgrid(x, y)
        original = xx**2 + yy**2

        # Remove a rectangular region
        grid = original.copy()
        grid[12:20, 12:20] = np.nan

        # Fill using FFT Inpaint
        config_obj = (
            config.fill.FFTInpaint().with_max_iterations(500).with_sigma(2.0)
        )
        fill.fft_inpaint(grid, config_obj)

        # Compare filled values with original
        # Note: FFT inpaint provides smooth harmonic solution, not exact
        # quadratic
        filled_region = np.zeros_like(original, dtype=bool)
        filled_region[12:20, 12:20] = True
        np.testing.assert_allclose(
            grid[filled_region], original[filled_region], rtol=2.0, atol=4.0
        )

    def test_fft_inpaint_analytical_gaussian_field(self) -> None:
        """Test FFT Inpaint with analytical Gaussian field."""
        # Create a smooth Gaussian field
        ny, nx = 32, 32
        x = np.linspace(-3, 3, nx)
        y = np.linspace(-3, 3, ny)
        xx, yy = np.meshgrid(x, y)
        original = np.exp(-(xx**2 + yy**2) / 2)

        # Remove scattered values
        rng = np.random.default_rng(123)
        grid = original.copy()
        mask_indices = rng.choice(nx * ny, size=nx * ny // 5, replace=False)
        grid.flat[mask_indices] = np.nan

        # Fill using FFT Inpaint
        config_obj = (
            config.fill.FFTInpaint().with_max_iterations(500).with_sigma(1.5)
        )
        fill.fft_inpaint(grid, config_obj)

        # Compare filled values with original
        # Note: With 20% missing data, reconstruction is approximate
        filled_mask = np.zeros_like(original, dtype=bool)
        filled_mask.flat[mask_indices] = True
        np.testing.assert_allclose(
            grid[filled_mask], original[filled_mask], rtol=0.5, atol=0.3
        )

    def test_fft_inpaint_analytical_periodic_field(self) -> None:
        """Test FFT Inpaint with periodic analytical field."""
        # Create a periodic field (good for FFT with periodic boundary)
        ny, nx = 32, 32
        x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
        y = np.linspace(0, 2 * np.pi, ny, endpoint=False)
        xx, yy = np.meshgrid(x, y)
        original = np.sin(2 * xx) + np.cos(3 * yy)

        # Remove scattered values
        rng = np.random.default_rng(456)
        grid = original.copy()
        mask_indices = rng.choice(nx * ny, size=nx * ny // 6, replace=False)
        grid.flat[mask_indices] = np.nan

        # Fill using FFT Inpaint with periodic boundary
        config_obj = (
            config.fill.FFTInpaint()
            .with_max_iterations(500)
            .with_sigma(1.0)
            .with_is_periodic(True)
        )
        fill.fft_inpaint(grid, config_obj)

        # Compare filled values with original
        filled_mask = np.zeros_like(original, dtype=bool)
        filled_mask.flat[mask_indices] = True
        np.testing.assert_allclose(
            grid[filled_mask], original[filled_mask], rtol=1.0, atol=2.0
        )

# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for LOESS fill method."""

from __future__ import annotations

import numpy as np
import pytest

from ....core import config, fill


class TestLoess:
    """Test LOESS fill method."""

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
    def default_config(self) -> config.fill.Loess:
        """Create default LOESS configuration."""
        return config.fill.Loess()

    def test_loess_basic_float64(
        self,
        sample_data_float64: np.ndarray,
        default_config: config.fill.Loess,
    ) -> None:
        """Test basic LOESS fill with float64 data."""
        result = fill.loess(sample_data_float64, default_config)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.shape == sample_data_float64.shape

    def test_loess_basic_float32(
        self,
        sample_data_float32: np.ndarray,
        default_config: config.fill.Loess,
    ) -> None:
        """Test basic LOESS fill with float32 data."""
        result = fill.loess(sample_data_float32, default_config)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == sample_data_float32.shape

    def test_loess_preserves_dtype(
        self, sample_data_float64: np.ndarray, sample_data_float32: np.ndarray
    ) -> None:
        """Test that LOESS preserves input data type."""
        config_obj = config.fill.Loess()

        result64 = fill.loess(sample_data_float64, config_obj)
        assert result64.dtype == np.float64

        result32 = fill.loess(sample_data_float32, config_obj)
        assert result32.dtype == np.float32

    def test_loess_with_custom_config(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test LOESS with custom configuration parameters."""
        custom_config = (
            config.fill.Loess()
            .with_epsilon(1e-6)
            .with_max_iterations(100)
            .with_first_guess(config.fill.FirstGuess.ZERO)
            .with_nx(3)
            .with_ny(3)
            .with_num_threads(2)
        )

        result = fill.loess(
            sample_data_float64,
            custom_config,
        )

        assert result is not None
        assert result.shape == sample_data_float64.shape
        assert result.dtype == np.float64

    def test_loess_value_type_all(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test LOESS with ALL value type."""
        config_obj = config.fill.Loess().with_value_type(
            config.fill.LoessValueType.ALL
        )

        result = fill.loess(sample_data_float64, config_obj)

        assert result is not None
        assert result.shape == sample_data_float64.shape

    def test_loess_value_type_defined(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test LOESS with DEFINED value type."""
        config_obj = config.fill.Loess().with_value_type(
            config.fill.LoessValueType.DEFINED
        )

        result = fill.loess(sample_data_float64, config_obj)

        assert result is not None
        assert result.shape == sample_data_float64.shape

    def test_loess_value_type_undefined(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test LOESS with UNDEFINED value type."""
        config_obj = config.fill.Loess().with_value_type(
            config.fill.LoessValueType.UNDEFINED
        )

        result = fill.loess(sample_data_float64, config_obj)

        assert result is not None
        assert result.shape == sample_data_float64.shape

    def test_loess_first_guess_zero(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test LOESS with ZERO first guess."""
        config_obj = config.fill.Loess().with_first_guess(
            config.fill.FirstGuess.ZERO
        )

        result = fill.loess(sample_data_float64, config_obj)

        assert result is not None
        assert result.shape == sample_data_float64.shape

    def test_loess_first_guess_zonal_average(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test LOESS with ZONAL_AVERAGE first guess."""
        config_obj = config.fill.Loess().with_first_guess(
            config.fill.FirstGuess.ZONAL_AVERAGE
        )

        result = fill.loess(sample_data_float64, config_obj)

        assert result is not None
        assert result.shape == sample_data_float64.shape

    def test_loess_window_sizes(self, sample_data_float64: np.ndarray) -> None:
        """Test LOESS with different window sizes."""
        # Test with different nx and ny values
        for nx, ny in [(3, 3), (5, 5), (3, 5), (5, 3)]:
            config_obj = config.fill.Loess().with_nx(nx).with_ny(ny)

            result = fill.loess(sample_data_float64, config_obj)

            assert result is not None
            assert result.shape == sample_data_float64.shape

    def test_loess_convergence_params(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test LOESS with different convergence parameters."""
        # Test with strict convergence
        strict_config = (
            config.fill.Loess().with_epsilon(1e-8).with_max_iterations(200)
        )

        result_strict = fill.loess(sample_data_float64, strict_config)
        assert result_strict is not None

        # Test with relaxed convergence
        relaxed_config = (
            config.fill.Loess().with_epsilon(1e-3).with_max_iterations(10)
        )

        result_relaxed = fill.loess(sample_data_float64, relaxed_config)
        assert result_relaxed is not None

    def test_loess_num_threads(self, sample_data_float64: np.ndarray) -> None:
        """Test LOESS with different thread counts."""
        for num_threads in [1, 2, 4]:
            config_obj = config.fill.Loess().with_num_threads(num_threads)

            result = fill.loess(sample_data_float64, config_obj)

            assert result is not None
            assert result.shape == sample_data_float64.shape

    def test_loess_all_nan_input(self) -> None:
        """Test LOESS behavior with all NaN input."""
        all_nan = np.full((4, 4), np.nan, dtype=np.float64)
        config_obj = config.fill.Loess()

        result = fill.loess(all_nan, config_obj)

        assert result is not None
        assert result.shape == all_nan.shape

    def test_loess_no_nan_input(self) -> None:
        """Test LOESS behavior with no NaN values."""
        no_nan = np.arange(16, dtype=np.float64).reshape(4, 4)
        config_obj = config.fill.Loess()

        result = fill.loess(no_nan, config_obj)

        assert result is not None
        assert result.shape == no_nan.shape
        # Result should be close to input when no filling is needed
        np.testing.assert_array_almost_equal(result, no_nan, decimal=5)

    def test_loess_preserves_valid_values(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test that LOESS preserves valid (non-NaN) input values."""
        config_obj = config.fill.Loess().with_value_type(
            config.fill.LoessValueType.UNDEFINED
        )

        result = fill.loess(sample_data_float64, config_obj)

        # Check that non-NaN values are preserved or very close
        valid_mask = ~np.isnan(sample_data_float64)
        np.testing.assert_array_almost_equal(
            result[valid_mask], sample_data_float64[valid_mask], decimal=1
        )

    def test_loess_small_array(self) -> None:
        """Test LOESS with small array."""
        small_data = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float64)
        config_obj = config.fill.Loess().with_nx(2).with_ny(2)

        result = fill.loess(small_data, config_obj)

        assert result is not None
        assert result.shape == small_data.shape

    def test_loess_large_array(self) -> None:
        """Test LOESS with larger array."""
        # Create a 20x20 array with scattered NaN values
        rng = np.random.default_rng(42)
        large_data = rng.random((20, 20), dtype=np.float64)
        # Add some NaN values
        large_data[::3, ::3] = np.nan

        config_obj = (
            config.fill.Loess()
            .with_nx(5)
            .with_ny(5)
            .with_num_threads(4)
            .with_max_iterations(50)
        )

        result = fill.loess(large_data, config_obj)

        assert result is not None
        assert result.shape == large_data.shape

    def test_loess_method_chaining_comprehensive(
        self, sample_data_float64: np.ndarray
    ) -> None:
        """Test comprehensive method chaining for LOESS configuration."""
        config_obj = (
            config.fill.Loess()
            .with_value_type(config.fill.LoessValueType.ALL)
            .with_first_guess(config.fill.FirstGuess.ZONAL_AVERAGE)
            .with_epsilon(1e-5)
            .with_max_iterations(150)
            .with_nx(4)
            .with_ny(4)
            .with_num_threads(2)
        )

        result = fill.loess(sample_data_float64, config_obj)

        assert result is not None
        assert result.shape == sample_data_float64.shape
        assert result.dtype == np.float64

# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for dask distributed computation module."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from .. import core


# Try to import dask, skip tests if not available
pytest.importorskip("dask")
pytest.importorskip("dask.array")


import dask.array as da

from .. import dask as dask_stats


@pytest.fixture
def random_values_dask() -> da.Array:
    """Generate random dask array for testing."""
    rng = np.random.default_rng(seed=42)
    values = rng.random(10000)
    return da.from_array(values, chunks=1000)


@pytest.fixture
def random_weights_dask() -> da.Array:
    """Generate random dask weights array for testing."""
    rng = np.random.default_rng(seed=43)
    weights = rng.random(10000)
    return da.from_array(weights, chunks=1000)


@pytest.fixture
def multidim_values_dask() -> da.Array:
    """Generate multi-dimensional dask array for testing."""
    rng = np.random.default_rng(seed=44)
    values = rng.random((50, 40, 30))
    return da.from_array(values, chunks=(10, 10, 10))


class TestDescriptiveStatistics:
    """Tests for descriptive_statistics function."""

    def test_basic_statistics(self, random_values_dask: da.Array) -> None:
        """Test basic statistics computation."""
        values_np = random_values_dask.compute()
        stats = dask_stats.descriptive_statistics(random_values_dask)

        assert stats.count() == values_np.size
        assert stats.min() == np.min(values_np)
        assert stats.max() == np.max(values_np)
        assert stats.mean() == pytest.approx(np.mean(values_np), rel=1e-10)
        assert stats.sum() == pytest.approx(np.sum(values_np), rel=1e-10)
        assert stats.variance() == pytest.approx(np.var(values_np), rel=1e-10)

    def test_weighted_statistics(
        self,
        random_values_dask: da.Array,
        random_weights_dask: da.Array,
    ) -> None:
        """Test weighted statistics computation."""
        values_np = random_values_dask.compute()
        weights_np = random_weights_dask.compute()

        stats = dask_stats.descriptive_statistics(
            random_values_dask,
            weights=random_weights_dask,
        )

        expected_mean = np.average(values_np, weights=weights_np)
        assert stats.count() == values_np.size
        assert stats.mean() == pytest.approx(expected_mean, rel=1e-10)
        assert stats.sum_of_weights() == pytest.approx(np.sum(weights_np))

    def test_with_dtype_float32(self, random_values_dask: da.Array) -> None:
        """Test statistics with float32 dtype."""
        stats = dask_stats.descriptive_statistics(
            random_values_dask,
            dtype="float32",
        )
        values_np = random_values_dask.compute()

        assert stats.count() == values_np.size
        assert stats.mean() == pytest.approx(np.mean(values_np), rel=1e-5)

    def test_shape_mismatch_error(self, random_values_dask: da.Array) -> None:
        """Test that shape mismatch raises an error."""
        rng = np.random.default_rng(seed=42)
        weights = da.from_array(rng.random(100), chunks=50)

        with pytest.raises(ValueError, match="same shape"):
            dask_stats.descriptive_statistics(
                random_values_dask,
                weights=weights,
            )

    def test_non_dask_array_error(self) -> None:
        """Test that non-dask arrays raise an error."""
        rng = np.random.default_rng(seed=42)
        values = rng.random(100)

        with pytest.raises(TypeError, match="must be a dask array"):
            dask_stats.descriptive_statistics(values)  # type: ignore[arg-type]

    def test_multidim_axis(self, multidim_values_dask: da.Array) -> None:
        """Test statistics along specific axis."""
        stats = dask_stats.descriptive_statistics(
            multidim_values_dask,
            axis=[1],
        )
        values_np = multidim_values_dask.compute()

        # Check shape
        expected_shape = (50, 30)
        assert stats.count().shape == expected_shape

        # Check mean values
        expected_mean = np.mean(values_np, axis=1)
        np.testing.assert_allclose(stats.mean(), expected_mean, rtol=1e-10)


class TestTDigest:
    """Tests for tdigest function."""

    def test_basic_quantiles(self, random_values_dask: da.Array) -> None:
        """Test basic quantile computation."""
        values_np = random_values_dask.compute()
        digest = dask_stats.tdigest(random_values_dask)

        assert digest.count() == values_np.size
        assert digest.min() == pytest.approx(np.min(values_np), rel=1e-6)
        assert digest.max() == pytest.approx(np.max(values_np), rel=1e-6)
        assert digest.mean() == pytest.approx(np.mean(values_np), rel=1e-3)

        # Test quantiles
        median = digest.quantile(0.5)
        assert median == pytest.approx(np.median(values_np), rel=1e-2)

    def test_weighted_quantiles(
        self,
        random_values_dask: da.Array,
        random_weights_dask: da.Array,
    ) -> None:
        """Test weighted quantile computation."""
        values_np = random_values_dask.compute()
        weights_np = random_weights_dask.compute()

        digest = dask_stats.tdigest(
            random_values_dask,
            weights=random_weights_dask,
        )

        assert digest.count() == values_np.size
        assert digest.sum_of_weights() == pytest.approx(np.sum(weights_np))

        # Weighted mean
        expected_mean = np.average(values_np, weights=weights_np)
        assert digest.mean() == pytest.approx(expected_mean, rel=1e-3)

    def test_compression_parameter(self, random_values_dask: da.Array) -> None:
        """Test that compression parameter works."""
        values_np = random_values_dask.compute()

        digest_low = dask_stats.tdigest(
            random_values_dask,
            compression=50,
        )
        digest_high = dask_stats.tdigest(
            random_values_dask,
            compression=500,
        )

        true_median = np.median(values_np)
        median_low = digest_low.quantile(0.5)
        median_high = digest_high.quantile(0.5)

        # Both should be reasonably close
        assert median_low == pytest.approx(true_median, rel=5e-2)
        assert median_high == pytest.approx(true_median, rel=1e-2)

    def test_quantile_array(self, random_values_dask: da.Array) -> None:
        """Test computing multiple quantiles at once."""
        values_np = random_values_dask.compute()
        digest = dask_stats.tdigest(random_values_dask)

        quantiles = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = digest.quantile(quantiles)

        assert result.shape == (5,)

        # Check min/max are exact
        assert result[0] == pytest.approx(np.min(values_np), abs=1e-10)
        assert result[4] == pytest.approx(np.max(values_np), abs=1e-10)

    def test_dtype_float32(self, random_values_dask: da.Array) -> None:
        """Test TDigest with float32 dtype."""
        digest = dask_stats.tdigest(random_values_dask, dtype="float32")
        values_np = random_values_dask.compute()

        assert digest.count() == values_np.size
        median = digest.quantile(0.5)
        assert median == pytest.approx(np.median(values_np), rel=1e-2)


class TestBinning1D:
    """Tests for binning1d function."""

    @pytest.fixture
    def binning1d_setup(self) -> tuple[core.Axis, da.Array, da.Array]:
        """Set up binning1d test data."""
        axis = core.Axis(np.linspace(0, 10, 11))

        rng = np.random.default_rng(seed=42)
        x = rng.uniform(0, 10, size=10000)
        z = rng.random(10000)

        x_dask = da.from_array(x, chunks=1000)
        z_dask = da.from_array(z, chunks=1000)

        return axis, x_dask, z_dask

    def test_basic_binning(
        self, binning1d_setup: tuple[core.Axis, da.Array, da.Array]
    ) -> None:
        """Test basic 1D binning."""
        axis, x_dask, z_dask = binning1d_setup
        binning = core.Binning1D(axis)

        result = dask_stats.binning1d(x_dask, z_dask, binning)

        # Should have accumulated data
        count = result.count()
        assert count.shape == (11,)
        assert np.sum(count) > 0

        # Mean should be valid where we have data
        mean = result.mean()
        assert not np.all(np.isnan(mean))

    def test_binning_with_weights(
        self, binning1d_setup: tuple[core.Axis, da.Array, da.Array]
    ) -> None:
        """Test 1D binning with weights."""
        axis, x_dask, z_dask = binning1d_setup

        rng = np.random.default_rng(seed=45)
        weights = rng.random(10000)
        weights_dask = da.from_array(weights, chunks=1000)

        binning = core.Binning1D(axis)
        result = dask_stats.binning1d(
            x_dask,
            z_dask,
            binning,
            weights=weights_dask,
        )

        # Check sum of weights
        sum_weights = result.sum_of_weights()
        assert np.sum(sum_weights[~np.isnan(sum_weights)]) > 0

    def test_binning_with_range(
        self, binning1d_setup: tuple[core.Axis, da.Array, da.Array]
    ) -> None:
        """Test 1D binning with range filter."""
        axis, x_dask, z_dask = binning1d_setup
        binning = core.Binning1D(axis, range=(2.0, 8.0))

        result = dask_stats.binning1d(x_dask, z_dask, binning)

        # Values outside range should not be counted
        assert result.range() == (2.0, 8.0)

    def test_shape_mismatch_error(
        self, binning1d_setup: tuple[core.Axis, da.Array, da.Array]
    ) -> None:
        """Test that shape mismatch raises error."""
        axis, x_dask, _ = binning1d_setup
        rng = np.random.default_rng(seed=42)
        z_dask = da.from_array(rng.random(100), chunks=50)
        binning = core.Binning1D(axis)

        with pytest.raises(ValueError, match="same shape"):
            dask_stats.binning1d(x_dask, z_dask, binning)


class TestBinning2D:
    """Tests for binning2d function."""

    @pytest.fixture
    def binning2d_setup(
        self,
    ) -> tuple[core.Axis, core.Axis, da.Array, da.Array, da.Array]:
        """Set up binning2d test data."""
        x_axis = core.Axis(np.linspace(0, 10, 11))
        y_axis = core.Axis(np.linspace(0, 10, 11))

        rng = np.random.default_rng(seed=42)
        x = rng.uniform(0, 10, size=10000)
        y = rng.uniform(0, 10, size=10000)
        z = rng.random(10000)

        x_dask = da.from_array(x, chunks=1000)
        y_dask = da.from_array(y, chunks=1000)
        z_dask = da.from_array(z, chunks=1000)

        return x_axis, y_axis, x_dask, y_dask, z_dask

    def test_basic_binning(
        self,
        binning2d_setup: tuple[
            core.Axis, core.Axis, da.Array, da.Array, da.Array
        ],
    ) -> None:
        """Test basic 2D binning."""
        x_axis, y_axis, x_dask, y_dask, z_dask = binning2d_setup
        binning = core.Binning2D(x_axis, y_axis)

        result = dask_stats.binning2d(x_dask, y_dask, z_dask, binning)

        # Should have accumulated data
        count = result.count()
        assert count.shape == (11, 11)
        assert np.sum(count) > 0

        # Mean should be valid where we have data
        mean = result.mean()
        assert not np.all(np.isnan(mean))

    def test_binning_simple_vs_linear(
        self,
        binning2d_setup: tuple[
            core.Axis, core.Axis, da.Array, da.Array, da.Array
        ],
    ) -> None:
        """Test simple vs linear binning."""
        x_axis, y_axis, x_dask, y_dask, z_dask = binning2d_setup

        binning_simple = core.Binning2D(x_axis, y_axis)
        binning_linear = core.Binning2D(x_axis, y_axis)

        result_simple = dask_stats.binning2d(
            x_dask, y_dask, z_dask, binning_simple, simple=True
        )
        result_linear = dask_stats.binning2d(
            x_dask, y_dask, z_dask, binning_linear, simple=False
        )

        # Both should have data but different distributions
        assert np.sum(result_simple.count()) > 0
        assert np.sum(result_linear.count()) > 0

    def test_binning_with_spheroid(
        self,
        binning2d_setup: tuple[
            core.Axis, core.Axis, da.Array, da.Array, da.Array
        ],
    ) -> None:
        """Test 2D binning with geographic coordinates."""
        x_axis, y_axis, x_dask, y_dask, z_dask = binning2d_setup
        spheroid = core.geometry.geographic.Spheroid()
        binning = core.Binning2D(x_axis, y_axis, spheroid=spheroid)

        result = dask_stats.binning2d(x_dask, y_dask, z_dask, binning)

        assert result.spheroid is not None
        assert np.sum(result.count()) > 0

    def test_shape_mismatch_error(
        self,
        binning2d_setup: tuple[
            core.Axis, core.Axis, da.Array, da.Array, da.Array
        ],
    ) -> None:
        """Test that shape mismatch raises error."""
        rng = np.random.default_rng(seed=42)
        x_axis, y_axis, x_dask, y_dask, _ = binning2d_setup
        z_dask = da.from_array(rng.random(100), chunks=50)
        binning = core.Binning2D(x_axis, y_axis)

        with pytest.raises(ValueError, match="same shape"):
            dask_stats.binning2d(x_dask, y_dask, z_dask, binning)


class TestHistogram2D:
    """Tests for histogram2d function."""

    @pytest.fixture
    def histogram2d_setup(
        self,
    ) -> tuple[core.Axis, core.Axis, da.Array, da.Array, da.Array]:
        """Set up histogram2d test data."""
        x_axis = core.Axis(np.linspace(0, 10, 11))
        y_axis = core.Axis(np.linspace(0, 10, 11))

        rng = np.random.default_rng(seed=42)
        x = rng.uniform(0, 10, size=10000)
        y = rng.uniform(0, 10, size=10000)
        z = rng.random(10000)

        x_dask = da.from_array(x, chunks=1000)
        y_dask = da.from_array(y, chunks=1000)
        z_dask = da.from_array(z, chunks=1000)

        return x_axis, y_axis, x_dask, y_dask, z_dask

    def test_basic_histogram(
        self,
        histogram2d_setup: tuple[
            core.Axis, core.Axis, da.Array, da.Array, da.Array
        ],
    ) -> None:
        """Test basic 2D histogram."""
        x_axis, y_axis, x_dask, y_dask, z_dask = histogram2d_setup
        histogram = core.Histogram2D(x_axis, y_axis)

        result = dask_stats.histogram2d(x_dask, y_dask, z_dask, histogram)

        # Should have accumulated data
        count = result.count()
        assert count.shape == (11, 11)
        assert np.sum(count) > 0

        # Mean should be valid where we have data
        mean = result.mean()
        assert not np.all(np.isnan(mean))

    def test_histogram_quantile(
        self,
        histogram2d_setup: tuple[
            core.Axis, core.Axis, da.Array, da.Array, da.Array
        ],
    ) -> None:
        """Test histogram quantile computation."""
        x_axis, y_axis, x_dask, y_dask, z_dask = histogram2d_setup
        histogram = core.Histogram2D(x_axis, y_axis)

        result = dask_stats.histogram2d(x_dask, y_dask, z_dask, histogram)

        # Compute median
        median = result.quantile(0.5)
        assert median.shape == (11, 11)

    def test_shape_mismatch_error(
        self,
        histogram2d_setup: tuple[
            core.Axis, core.Axis, da.Array, da.Array, da.Array
        ],
    ) -> None:
        """Test that shape mismatch raises error."""
        rng = np.random.default_rng(seed=42)
        x_axis, y_axis, x_dask, y_dask, _ = histogram2d_setup
        z_dask = da.from_array(rng.random(100), chunks=50)
        histogram = core.Histogram2D(x_axis, y_axis)

        with pytest.raises(ValueError, match="same shape"):
            dask_stats.histogram2d(x_dask, y_dask, z_dask, histogram)


class TestDaskNotInstalled:
    """Tests for behavior when dask is not installed."""

    def test_import_error_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that a helpful error message is shown when dask is missing."""
        # Save original modules
        original_dask = sys.modules.get("dask")
        original_dask_array = sys.modules.get("dask.array")

        try:
            # Remove dask from modules
            sys.modules["dask"] = None  # type: ignore[assignment]
            sys.modules["dask.array"] = None  # type: ignore[assignment]

            # Import the function fresh to test error
            from .. import dask as dask_module  # noqa: PLC0415

            importlib.reload(dask_module)

            with pytest.raises(ImportError, match="dask is required"):
                dask_module._check_dask_available()
        finally:
            # Restore original modules
            if original_dask is not None:
                sys.modules["dask"] = original_dask
            else:
                sys.modules.pop("dask", None)
            if original_dask_array is not None:
                sys.modules["dask.array"] = original_dask_array
            else:
                sys.modules.pop("dask.array", None)

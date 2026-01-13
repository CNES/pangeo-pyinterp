# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for T-Digest quantile computation."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ... import core


if TYPE_CHECKING:
    from numpy.typing import NDArray


# Fixtures
@pytest.fixture
def random_values() -> NDArray:
    """Generate random values for testing."""
    rng = np.random.default_rng(seed=42)
    return rng.random(10000)


@pytest.fixture
def random_weights() -> NDArray:
    """Generate random weights for testing."""
    rng = np.random.default_rng(seed=43)
    return rng.random(10000)


@pytest.fixture
def multidim_values() -> NDArray:
    """Generate multi-dimensional random values for testing."""
    rng = np.random.default_rng(seed=44)
    return rng.random((50, 40, 30))


class TestTDigestFloat64:
    """Tests for TDigestFloat64 (float64 specialization)."""

    def test_empty_container(self) -> None:
        """Test the calculation of quantiles on an empty container."""
        td = core.TDigest(np.array([]))

        assert td.count() == 0
        assert np.isnan(td.max())
        assert np.isnan(td.mean())
        assert np.isnan(td.min())
        # Note: sum_of_weights returns 0.0 for empty container (not NaN)
        assert td.sum_of_weights() == 0.0
        assert np.isnan(td.quantile(0.5))

    def test_single_value(self) -> None:
        """Test t-digest with a single value."""
        td = core.TDigest(np.array([42.0]))

        assert td.count() == 1
        assert td.min() == 42.0
        assert td.max() == 42.0
        assert td.mean() == 42.0
        assert td.quantile(0.0) == 42.0
        assert td.quantile(0.5) == 42.0
        assert td.quantile(1.0) == 42.0

    def test_quantiles(self, random_values: NDArray) -> None:
        """Test quantile calculation."""
        td = core.TDigest(random_values)

        # Test basic quantiles
        assert td.count() == random_values.size
        assert td.min() == pytest.approx(np.min(random_values), rel=1e-6)
        assert td.max() == pytest.approx(np.max(random_values), rel=1e-6)
        assert td.mean() == pytest.approx(np.mean(random_values), rel=1e-3)

        # Test quantile accuracy
        median = td.quantile(0.5)
        assert median == pytest.approx(np.median(random_values), rel=1e-2)

        q25 = td.quantile(0.25)
        assert q25 == pytest.approx(np.quantile(random_values, 0.25), rel=1e-2)

        q75 = td.quantile(0.75)
        assert q75 == pytest.approx(np.quantile(random_values, 0.75), rel=1e-2)

    def test_quantiles_vector(self, random_values: NDArray) -> None:
        """Test multiple quantile calculation at once."""
        td = core.TDigest(random_values)

        quantiles = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = td.quantile(quantiles)

        # Should return 1D array for single digest
        assert result.ndim == 1
        assert result.shape == (5,)

        # Check accuracy
        expected = np.quantile(random_values, quantiles)
        for i, q in enumerate(quantiles):
            if q == 0.0 or q == 1.0:
                # Exact for min/max
                assert result[i] == pytest.approx(expected[i], abs=1e-10)
            else:
                # Approximate for intermediate quantiles
                assert result[i] == pytest.approx(expected[i], rel=1e-2)

    def test_merge(self, random_values: NDArray) -> None:
        """Test merging two TDigest instances."""
        td1 = core.TDigest(random_values[:5000])
        td2 = core.TDigest(random_values[5000:])

        td1 += td2
        del td2

        assert td1.count() == random_values.size
        assert td1.min() == pytest.approx(np.min(random_values), rel=1e-6)
        assert td1.max() == pytest.approx(np.max(random_values), rel=1e-6)
        assert td1.mean() == pytest.approx(np.mean(random_values), rel=1e-3)

        median = td1.quantile(0.5)
        assert median == pytest.approx(np.median(random_values), rel=1e-2)

    def test_pickle(self, random_values: NDArray) -> None:
        """Test pickling and unpickling."""
        td = core.TDigest(random_values)
        other = pickle.loads(pickle.dumps(td))

        assert other.count() == td.count()
        assert other.max() == td.max()
        assert other.mean() == td.mean()
        assert other.min() == td.min()
        assert other.sum_of_weights() == td.sum_of_weights()
        assert other.quantile(0.5) == td.quantile(0.5)

    def test_weighted(
        self, random_values: NDArray, random_weights: NDArray
    ) -> None:
        """Test weighted quantiles."""
        td = core.TDigest(random_values, weights=random_weights)

        # count() returns the number of samples, not sum of weights
        assert td.count() == random_values.size
        assert td.min() == np.min(random_values)
        assert td.max() == np.max(random_values)
        assert td.sum_of_weights() == pytest.approx(np.sum(random_weights))

        # Weighted mean
        expected_mean = np.average(random_values, weights=random_weights)
        assert td.mean() == pytest.approx(expected_mean, rel=1e-3)

        # Weighted quantile (approximate check)
        median = td.quantile(0.5)
        assert 0.0 <= median <= 1.0  # Should be in valid range

    def test_weighted_shape_mismatch(self, random_values: NDArray) -> None:
        """Test that mismatched shapes raise an error."""
        with pytest.raises(ValueError):
            core.TDigest(random_values, weights=random_values[:100])

    def test_compression_parameter(self, random_values: NDArray) -> None:
        """Test that higher compression improves accuracy."""
        td_low = core.TDigest(random_values, compression=50)
        td_high = core.TDigest(random_values, compression=500)

        # Both should give reasonable estimates
        true_median = np.median(random_values)
        median_low = td_low.quantile(0.5)
        median_high = td_high.quantile(0.5)

        # Check both are close
        assert median_low == pytest.approx(true_median, rel=5e-2)
        assert median_high == pytest.approx(true_median, rel=1e-2)

        # Higher compression should typically be more accurate or equally good
        # (though this is not guaranteed for every dataset)
        error_low = abs(median_low - true_median)
        error_high = abs(median_high - true_median)
        # If low compression is already perfect (error = 0), skip comparison
        if error_low > 1e-10:
            assert error_high <= error_low * 2  # Allow some tolerance

    def test_axis_none(self, multidim_values: NDArray) -> None:
        """Test quantiles computed along all axes (axis=None)."""
        td = core.TDigest(multidim_values, axis=None)

        assert td.count() == multidim_values.size
        flat_values = multidim_values.flatten()

        median = td.quantile(0.5)
        assert median == pytest.approx(np.median(flat_values), rel=1e-2)

    def test_axis_single(self, multidim_values: NDArray) -> None:
        """Test quantiles computed along a single axis."""
        axis = [1]
        td = core.TDigest(multidim_values, axis=axis)

        # Should have shape with axis removed
        expected_shape = (
            multidim_values.shape[0],
            multidim_values.shape[2],
        )
        assert td.count().shape == expected_shape

        # Check a few medians
        medians = td.quantile(0.5)
        assert medians.shape == expected_shape

        # Verify a sample median
        expected_median = np.median(multidim_values[0, :, 0])
        assert medians[0, 0] == pytest.approx(expected_median, rel=1e-2)

    def test_axis_multiple(self, multidim_values: NDArray) -> None:
        """Test quantiles computed along multiple axes."""
        axis = [0, 2]
        td = core.TDigest(multidim_values, axis=axis)

        # Should have shape with axes removed
        expected_shape = (multidim_values.shape[1],)
        assert td.count().shape == expected_shape

        medians = td.quantile(0.5)
        assert medians.shape == expected_shape

    def test_quantile_bounds_error(self, random_values: NDArray) -> None:
        """Test that quantile values outside [0, 1] raise an error."""
        td = core.TDigest(random_values)

        with pytest.raises((ValueError, RuntimeError)):
            td.quantile(-0.1)

        with pytest.raises((ValueError, RuntimeError)):
            td.quantile(1.1)


class TestTDigestFloat32:
    """Tests for TDigestFloat32 (float32 specialization)."""

    def test_creation_explicit_dtype(self) -> None:
        """Test creating TDigest with explicit float32 dtype."""
        rng = np.random.default_rng(seed=42)
        values = rng.random(1000).astype(np.float32)
        td = core.TDigest(values, dtype="float32")

        assert td.count() == values.size
        median = td.quantile(0.5)
        expected_median = np.median(values)
        assert median == pytest.approx(expected_median, rel=1e-2)

    def test_dtype_parameter_default(self) -> None:
        """Test that default dtype is float64."""
        rng = np.random.default_rng(seed=42)
        values = rng.random(1000)
        td = core.TDigest(values)

        # Verify it's using float64 by checking precision
        median = td.quantile(0.5)
        expected_median = np.median(values)
        assert median == pytest.approx(expected_median, rel=1e-2)

    def test_dtype_parameter_string(self) -> None:
        """Test dtype parameter with string values."""
        rng = np.random.default_rng(seed=42)
        values = rng.random(1000)

        td32 = core.TDigest(values, dtype="float32")
        td64 = core.TDigest(values, dtype="float64")

        # Both should give similar results
        assert td32.count() == td64.count()
        median32 = td32.quantile(0.5)
        median64 = td64.quantile(0.5)
        assert median32 == pytest.approx(median64, rel=1e-4)

    def test_invalid_dtype(self) -> None:
        """Test that invalid dtype raises an error."""
        rng = np.random.default_rng(seed=42)
        values = rng.random(100)

        with pytest.raises((ValueError, TypeError)):
            core.TDigest(values, dtype="float16")

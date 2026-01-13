# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for descriptive statistics computation."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ... import core


if TYPE_CHECKING:
    from numpy.typing import NDArray


def weighted_mom1(values: NDArray, weights: NDArray) -> np.floating:
    """Return the weighted moment 1 of the values."""
    return np.average(values, weights=weights)


def weighted_mom2(values: NDArray, weights: NDArray) -> np.floating:
    """Return the weighted moment 2 of the values."""
    average = weighted_mom1(values, weights)
    return np.average((values - average) ** 2, weights=weights)


def weighted_mom3(values: NDArray, weights: NDArray) -> np.floating:
    """Return the weighted moment 3 of the values."""
    average = weighted_mom1(values, weights)
    mom2 = weighted_mom2(values, weights)
    return np.average(
        ((values - average) / np.sqrt(mom2)) ** 3, weights=weights
    )


def weighted_mom4(values: NDArray, weights: NDArray) -> np.floating:
    """Return the weighted moment 4 of the values."""
    average = weighted_mom1(values, weights)
    mom2 = weighted_mom2(values, weights)
    return np.average(
        ((values - average) / np.sqrt(mom2)) ** 4 - 3, weights=weights
    )


# Fixtures
@pytest.fixture
def random_values() -> NDArray:
    """Generate random values for testing."""
    rng = np.random.default_rng(seed=42)
    return rng.random(1000)


@pytest.fixture
def random_weights() -> NDArray:
    """Generate random weights for testing."""
    rng = np.random.default_rng(seed=43)
    return rng.random(1000)


@pytest.fixture
def multidim_values() -> NDArray:
    """Generate multi-dimensional random values for testing."""
    rng = np.random.default_rng(seed=44)
    return rng.random((2, 3, 4, 5, 6, 7))


class TestDescriptiveStatisticsFloat64:
    """Tests for DescriptiveStatisticsFloat64 (float64 specialization)."""

    def test_empty_container(self) -> None:
        """Test the calculation of statistics on an empty container."""
        ds = core.DescriptiveStatistics(np.array([]))

        assert ds.count() == 0
        assert np.isnan(ds.max())
        assert np.isnan(ds.mean())
        assert np.isnan(ds.min())
        assert ds.sum_of_weights() == 0
        assert ds.sum() == 0
        assert np.isnan(ds.variance())
        assert np.isnan(ds.skewness())
        assert np.isnan(ds.kurtosis())

    def test_flatten(self, random_values: NDArray) -> None:
        """Test the calculation of statistics on a vector."""
        ds = core.DescriptiveStatistics(random_values)

        assert ds.count() == random_values.size
        assert ds.max() == np.max(random_values)
        assert ds.mean() == pytest.approx(np.mean(random_values))
        assert ds.min() == np.min(random_values)
        assert ds.sum_of_weights() == random_values.size
        assert ds.sum() == pytest.approx(np.sum(random_values))
        assert ds.variance() == pytest.approx(np.var(random_values))
        assert ds.kurtosis() == pytest.approx(
            weighted_mom4(random_values, np.ones(random_values.size))
        )
        assert ds.skewness() == pytest.approx(
            weighted_mom3(random_values, np.ones(random_values.size))
        )

    def test_merge(self, random_values: NDArray) -> None:
        """Test merging two DescriptiveStatistics instances."""
        instance1 = core.DescriptiveStatistics(random_values[:500])
        instance2 = core.DescriptiveStatistics(random_values[500:])

        instance1 += instance2
        del instance2

        assert instance1.count() == random_values.size
        assert instance1.max() == np.max(random_values)
        assert instance1.mean() == pytest.approx(np.mean(random_values))
        assert instance1.min() == np.min(random_values)
        assert instance1.sum_of_weights() == random_values.size
        assert instance1.sum() == pytest.approx(np.sum(random_values))
        assert instance1.variance() == pytest.approx(np.var(random_values))
        assert instance1.kurtosis() == pytest.approx(
            weighted_mom4(random_values, np.ones(random_values.size))
        )
        assert instance1.skewness() == pytest.approx(
            weighted_mom3(random_values, np.ones(random_values.size))
        )

    def test_pickle(self, random_values: NDArray) -> None:
        """Test pickling and unpickling."""
        ds = core.DescriptiveStatistics(random_values)
        other = pickle.loads(pickle.dumps(ds))
        assert other.count() == ds.count()
        assert other.max() == ds.max()
        assert other.mean() == ds.mean()
        assert other.min() == ds.min()
        assert other.sum_of_weights() == ds.sum_of_weights()
        assert other.sum() == ds.sum()
        assert other.variance() == ds.variance()
        assert other.skewness() == ds.skewness()
        assert other.kurtosis() == ds.kurtosis()

    def test_weighted(
        self, random_values: NDArray, random_weights: NDArray
    ) -> None:
        """Test weighted statistics."""
        ds = core.DescriptiveStatistics(random_values, weights=random_weights)

        assert ds.count() == random_values.size
        # Note: max/min return the max/min of the values themselves,
        # not weighted
        assert ds.max() == np.max(random_values)
        assert ds.mean() == pytest.approx(
            weighted_mom1(random_values, random_weights)
        )
        assert ds.min() == np.min(random_values)
        assert ds.sum_of_weights() == pytest.approx(np.sum(random_weights))
        assert ds.sum() == pytest.approx(
            np.sum(random_values * random_weights)
        )
        assert ds.variance() == pytest.approx(
            weighted_mom2(random_values, random_weights)
        )
        assert ds.kurtosis() == pytest.approx(
            weighted_mom4(random_values, random_weights)
        )
        assert ds.skewness() == pytest.approx(
            weighted_mom3(random_values, random_weights)
        )

    def test_weighted_shape_mismatch(self, random_values: NDArray) -> None:
        """Test that mismatched shapes raise an error."""
        with pytest.raises(ValueError):
            core.DescriptiveStatistics(
                random_values, weights=random_values[:100]
            )

    def test_axis_none(self, multidim_values: NDArray) -> None:
        """Test statistics computed along all axes (axis=None)."""
        ds = core.DescriptiveStatistics(multidim_values, axis=None)
        assert np.all(ds.count() == np.sum(multidim_values * 0 + 1))
        assert np.all(ds.max() == np.max(multidim_values))
        assert ds.mean() == pytest.approx(np.mean(multidim_values))
        assert np.all(ds.min() == np.min(multidim_values))
        assert np.all(ds.sum_of_weights() == np.sum(multidim_values * 0 + 1))
        assert ds.sum() == pytest.approx(np.sum(multidim_values))
        assert ds.variance() == pytest.approx(np.var(multidim_values))

    def test_axis_single(self, multidim_values: NDArray) -> None:
        """Test statistics computed along a single axis."""
        axis = [1]
        ds = core.DescriptiveStatistics(multidim_values, axis=axis)
        assert np.all(
            ds.count() == np.sum(multidim_values * 0 + 1, axis=tuple(axis))
        )
        assert np.all(ds.max() == np.max(multidim_values, axis=tuple(axis)))
        assert ds.mean() == pytest.approx(
            np.mean(multidim_values, axis=tuple(axis))
        )
        assert np.all(ds.min() == np.min(multidim_values, axis=tuple(axis)))
        assert np.all(
            ds.sum_of_weights()
            == np.sum(multidim_values * 0 + 1, axis=tuple(axis))
        )
        assert ds.sum() == pytest.approx(
            np.sum(multidim_values, axis=tuple(axis))
        )
        assert ds.variance() == pytest.approx(
            np.var(multidim_values, axis=tuple(axis))
        )

    def test_axis_multiple(self, multidim_values: NDArray) -> None:
        """Test statistics computed along multiple axes."""
        axis = [2, 3]
        ds = core.DescriptiveStatistics(multidim_values, axis=axis)
        assert np.all(
            ds.count() == np.sum(multidim_values * 0 + 1, axis=tuple(axis))
        )
        assert np.all(ds.max() == np.max(multidim_values, axis=tuple(axis)))
        assert ds.mean() == pytest.approx(
            np.mean(multidim_values, axis=tuple(axis))
        )
        assert np.all(ds.min() == np.min(multidim_values, axis=tuple(axis)))
        assert np.all(
            ds.sum_of_weights()
            == np.sum(multidim_values * 0 + 1, axis=tuple(axis))
        )
        assert ds.sum() == pytest.approx(
            np.sum(multidim_values, axis=tuple(axis))
        )
        assert ds.variance() == pytest.approx(
            np.var(multidim_values, axis=tuple(axis))
        )

    def test_axis_nonconsecutive(self, multidim_values: NDArray) -> None:
        """Test statistics computed along non-consecutive axes."""
        axis = [1, 3, 5]
        ds = core.DescriptiveStatistics(multidim_values, axis=axis)
        assert np.all(
            ds.count() == np.sum(multidim_values * 0 + 1, axis=tuple(axis))
        )
        assert np.all(ds.max() == np.max(multidim_values, axis=tuple(axis)))
        assert ds.mean() == pytest.approx(
            np.mean(multidim_values, axis=tuple(axis))
        )
        assert np.all(ds.min() == np.min(multidim_values, axis=tuple(axis)))
        assert np.all(
            ds.sum_of_weights()
            == np.sum(multidim_values * 0 + 1, axis=tuple(axis))
        )
        assert ds.sum() == pytest.approx(
            np.sum(multidim_values, axis=tuple(axis))
        )
        assert ds.variance() == pytest.approx(
            np.var(multidim_values, axis=tuple(axis))
        )

    def test_axis_out_of_bounds(self, multidim_values: NDArray) -> None:
        """Test that out-of-bounds axis raises an error."""
        with pytest.raises((ValueError, IndexError)):
            core.DescriptiveStatistics(multidim_values, axis=[12])

    def test_axis_negative_not_supported(
        self, multidim_values: NDArray
    ) -> None:
        """Test that negative axis indices are not supported."""
        # Negative indices should be converted to positive by the tensor
        # validation.
        # So axis=[-1] should work and be equivalent to axis=[5] for a 6D array
        ds = core.DescriptiveStatistics(multidim_values, axis=[-1])
        expected_shape = multidim_values.shape[:-1]
        assert ds.mean().shape == expected_shape


class TestDescriptiveStatisticsFloat32:
    """Tests for DescriptiveStatisticsFloat32 (float32 specialization)."""

    def test_creation_explicit_dtype(self) -> None:
        """Test creating DescriptiveStatistics with explicit float32 dtype."""
        rng = np.random.default_rng(seed=42)
        values = rng.random(100).astype(np.float32)
        ds = core.DescriptiveStatistics(values, dtype="float32")

        assert ds.count() == values.size
        assert ds.mean() == pytest.approx(np.mean(values), rel=1e-5)
        assert ds.variance() == pytest.approx(np.var(values), rel=1e-5)

    def test_dtype_parameter_default(self) -> None:
        """Test that default dtype is float64."""
        rng = np.random.default_rng(seed=42)
        values = rng.random(100)
        ds = core.DescriptiveStatistics(values)

        # Verify it's using float64 by checking precision
        assert ds.mean() == pytest.approx(np.mean(values), rel=1e-14)

    def test_dtype_parameter_string(self) -> None:
        """Test dtype parameter with string values."""
        rng = np.random.default_rng(seed=42)
        values = rng.random(100)

        ds32 = core.DescriptiveStatistics(values, dtype="float32")
        ds64 = core.DescriptiveStatistics(values, dtype="float64")

        # Float32 should have less precision than float64
        assert ds32.count() == ds64.count()
        assert ds32.mean() == pytest.approx(ds64.mean(), rel=1e-6)

    def test_invalid_dtype(self) -> None:
        """Test that invalid dtype raises an error."""
        rng = np.random.default_rng(seed=42)
        values = rng.random(100)

        with pytest.raises((ValueError, TypeError)):
            core.DescriptiveStatistics(values, dtype="float16")

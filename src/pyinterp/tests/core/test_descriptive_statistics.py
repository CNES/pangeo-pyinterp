# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pickle

import numpy as np
import pytest

from ... import core


def weighted_mom1(values, weights):
    """Return the weighted moment 1 of the values."""
    return np.average(values, weights=weights)


def weighted_mom2(values, weights):
    """Return the weighted moment 2 of the values."""
    average = weighted_mom1(values, weights)
    return np.average((values - average)**2, weights=weights)


def weighted_mom3(values, weights):
    """Return the weighted moment 3 of the values."""
    average = weighted_mom1(values, weights)
    mom2 = weighted_mom2(values, weights)
    return np.average(((values - average) / np.sqrt(mom2))**3, weights=weights)


def weighted_mom4(values, weights):
    """Return the weighted moment 4 of the values."""
    average = weighted_mom1(values, weights)
    mom2 = weighted_mom2(values, weights)
    return np.average(((values - average) / np.sqrt(mom2))**4 - 3,
                      weights=weights)


def test_empty_container():
    """Test the calculation of statistics on an empty container."""
    ds = core.DescriptiveStatisticsFloat64(np.array([]))

    assert ds.count() == 0
    assert np.isnan(ds.max())
    assert np.isnan(ds.mean())
    assert np.isnan(ds.min())
    assert ds.sum_of_weights() == 0
    assert ds.sum() == 0
    assert np.isnan(ds.variance())
    assert np.isnan(ds.skewness())
    assert np.isnan(ds.kurtosis())


def test_flatten():
    """Test the calculation of statistics on a vector."""
    values = np.random.random_sample(1000)
    ds = core.DescriptiveStatisticsFloat64(values)

    assert ds.count() == values.size
    assert ds.max() == np.max(values)
    assert ds.mean() == pytest.approx(np.mean(values))
    assert ds.min() == np.min(values)
    assert ds.sum_of_weights() == values.size
    assert ds.sum() == pytest.approx(np.sum(values))
    assert ds.variance() == pytest.approx(np.var(values))
    assert ds.kurtosis() == pytest.approx(
        weighted_mom4(values, np.ones(values.size)))
    assert ds.skewness() == pytest.approx(
        weighted_mom3(values, np.ones(values.size)))


def test_merge():
    """Test merging."""
    values = np.random.random_sample(1000)
    instance1 = core.DescriptiveStatisticsFloat64(values[:500])
    instance2 = core.DescriptiveStatisticsFloat64(values[500:])

    instance1 += instance2
    del instance2

    assert instance1.count() == values.size
    assert instance1.max() == np.max(values)
    assert instance1.mean() == pytest.approx(np.mean(values))
    assert instance1.min() == np.min(values)
    assert instance1.sum_of_weights() == values.size
    assert instance1.sum() == pytest.approx(np.sum(values))
    assert instance1.variance() == pytest.approx(np.var(values))
    assert instance1.kurtosis() == pytest.approx(
        weighted_mom4(values, np.ones(values.size)))
    assert instance1.skewness() == pytest.approx(
        weighted_mom3(values, np.ones(values.size)))


def test_pickle():
    """Test pickling."""
    values = np.random.random_sample(1000)
    ds = core.DescriptiveStatisticsFloat64(values)
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


def test_weighted():
    """Test weighted statistics."""
    values = np.random.random_sample(1000)
    weights = np.random.random_sample(1000)
    ds = core.DescriptiveStatisticsFloat64(values, weights=weights)

    assert ds.count() == values.size
    assert ds.max() == np.max(values * weights)
    assert ds.mean() == pytest.approx(weighted_mom1(values, weights))
    assert ds.min() == np.min(values * weights)
    assert ds.sum_of_weights() == pytest.approx(np.sum(weights))
    assert ds.sum() == pytest.approx(np.sum(values * weights))
    assert ds.variance() == pytest.approx(weighted_mom2(values, weights))
    assert ds.kurtosis() == pytest.approx(weighted_mom4(values, weights))
    assert ds.skewness() == pytest.approx(weighted_mom3(values, weights))

    with pytest.raises(ValueError):
        core.DescriptiveStatisticsFloat64(values, weights=weights[:100])


def test_axis():
    """Test axes along which the statistics are computed."""
    values = np.random.random_sample((2, 3, 4, 5, 6, 7))

    def check_axis(values, axis):
        ds = core.DescriptiveStatisticsFloat64(values, axis=axis)
        assert np.all(ds.count() == np.sum(values * 0 + 1, axis=axis))
        assert np.all(ds.max() == np.max(values, axis=axis))
        assert ds.mean() == pytest.approx(np.mean(values, axis=axis))
        assert np.all(ds.min() == np.min(values, axis=axis))
        assert np.all(ds.sum_of_weights() == np.sum(values * 0 + 1, axis=axis))
        assert ds.sum() == pytest.approx(np.sum(values, axis=axis))
        assert ds.variance() == pytest.approx(np.var(values, axis=axis))

    check_axis(values, None)
    check_axis(values, (1, ))
    check_axis(values, (2, 3))
    check_axis(values, (1, 3, 5))

    with pytest.raises(ValueError):
        core.DescriptiveStatisticsFloat64(values, axis=(12, ))  # type: ignore
    with pytest.raises(ValueError):
        core.DescriptiveStatisticsFloat64(values, axis=(-1, ))  # type: ignore

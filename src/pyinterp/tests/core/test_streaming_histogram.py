# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pickle

import numpy as np
import pytest

from ... import core
from .test_descriptive_statistics import (
    weighted_mom1,
    weighted_mom2,
    weighted_mom3,
    weighted_mom4,
)


def weighted_quantile(data, weights, perc):
    """Calculate weighted quantile."""
    ix = np.argsort(data)
    data = data[ix]
    weights = weights[ix]
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    return np.interp(perc, cdf, data)


def test_empty_container():
    """Test the calculation of statistics on an empty container."""
    ds = core.StreamingHistogramFloat64(np.array([]))

    assert ds.count() == 0
    assert np.isnan(ds.max())
    assert np.isnan(ds.mean())
    assert np.isnan(ds.min())
    assert np.isnan(ds.quantile())
    assert ds.sum_of_weights() == 0
    assert np.isnan(ds.variance())
    assert np.isnan(ds.skewness())
    assert np.isnan(ds.kurtosis())


def test_flatten():
    """Test the calculation of statistics on a vector."""
    values = np.random.random_sample(1000)
    ds = core.StreamingHistogramFloat64(values, bin_count=1000)

    assert ds.count() == values.size
    assert ds.size() == values.size
    assert ds.max() == np.max(values)
    assert ds.mean() == pytest.approx(np.mean(values))
    assert ds.min() == np.min(values)
    assert ds.sum_of_weights() == values.size
    assert ds.variance() == pytest.approx(np.var(values))
    assert ds.kurtosis() == pytest.approx(
        weighted_mom4(values, np.ones(values.size)))
    assert ds.skewness() == pytest.approx(
        weighted_mom3(values, np.ones(values.size)))

    ds.resize(100)
    assert ds.count() == values.size
    assert ds.size() == 100
    assert ds.max() == np.max(values)
    assert ds.mean() == pytest.approx(np.mean(values))
    assert ds.min() == np.min(values)
    assert ds.sum_of_weights() == values.size
    assert ds.variance() == pytest.approx(np.var(values), abs=1e-5)
    assert ds.kurtosis() == pytest.approx(weighted_mom4(
        values, np.ones(values.size)),
                                          abs=1e-3)
    assert ds.skewness() == pytest.approx(weighted_mom3(
        values, np.ones(values.size)),
                                          abs=1e-3)


def test_merge():
    """Test merging."""
    values = np.random.random_sample(1000)
    instance1 = core.StreamingHistogramFloat64(values[:500], bin_count=1000)
    instance2 = core.StreamingHistogramFloat64(values[500:], bin_count=1000)

    instance1 += instance2
    del instance2

    assert instance1.count() == values.size
    assert instance1.size() == 1000
    assert instance1.max() == np.max(values)
    assert instance1.mean() == pytest.approx(np.mean(values))
    assert instance1.min() == np.min(values)
    assert instance1.sum_of_weights() == values.size
    assert instance1.variance() == pytest.approx(np.var(values))
    assert instance1.kurtosis() == pytest.approx(
        weighted_mom4(values, np.ones(values.size)))
    assert instance1.skewness() == pytest.approx(
        weighted_mom3(values, np.ones(values.size)))


def test_pickle():
    """Test pickling."""
    values = np.random.random_sample(1000)
    ds = core.StreamingHistogramFloat64(values)
    other = pickle.loads(pickle.dumps(ds))
    assert other.count() == ds.count()
    assert other.kurtosis() == ds.kurtosis()
    assert other.max() == ds.max()
    assert other.mean() == ds.mean()
    assert other.min() == ds.min()
    assert other.size() == ds.size()
    assert other.skewness() == ds.skewness()
    assert other.sum_of_weights() == ds.sum_of_weights()
    assert other.variance() == ds.variance()


def test_weighted():
    """Test weighted statistics."""
    values = np.random.random_sample(1000)
    weights = np.random.random_sample(1000)
    ds = core.StreamingHistogramFloat64(values,
                                        weights=weights,
                                        bin_count=1000)

    assert ds.count() == values.size
    assert ds.size() == 1000
    assert ds.max() == np.max(values * weights)
    assert ds.mean() == pytest.approx(weighted_mom1(values, weights))
    assert ds.min() == np.min(values * weights)
    assert ds.sum_of_weights() == pytest.approx(np.sum(weights))
    assert ds.variance() == pytest.approx(weighted_mom2(values, weights))
    assert ds.kurtosis() == pytest.approx(weighted_mom4(values, weights))
    assert ds.skewness() == pytest.approx(weighted_mom3(values, weights))
    assert ds.quantile() == pytest.approx(
        weighted_quantile(values, weights, 0.5))

    with pytest.raises(ValueError):
        core.DescriptiveStatisticsFloat64(values, weights=weights[:100])

    ds.resize(100)
    assert ds.count() == values.size
    assert ds.size() == 100
    assert ds.max() == np.max(values * weights)
    assert ds.mean() == pytest.approx(weighted_mom1(values, weights))
    assert ds.min() == np.min(values * weights)
    assert ds.sum_of_weights() == pytest.approx(np.sum(weights))
    assert ds.variance() == pytest.approx(weighted_mom2(values, weights),
                                          abs=1e-2)
    assert ds.kurtosis() == pytest.approx(weighted_mom4(values, weights),
                                          abs=1e-2)
    assert ds.skewness() == pytest.approx(weighted_mom3(values, weights),
                                          abs=1e-2)
    assert ds.quantile() == pytest.approx(weighted_quantile(
        values, weights, 0.5),
                                          abs=1e-2)


def test_axis():
    """Test axes along which the statistics are computed."""
    values = np.random.random_sample((2, 3, 4, 5, 6, 7))

    def check_axis(values, axis):
        ds = core.StreamingHistogramFloat64(values,
                                            axis=axis,
                                            bin_count=values.size)
        assert np.all(ds.count() == np.sum(values * 0 + 1, axis=axis))
        assert np.all(ds.max() == np.max(values, axis=axis))
        assert ds.mean() == pytest.approx(np.mean(values, axis=axis))
        assert np.all(ds.min() == np.min(values, axis=axis))
        assert np.all(ds.sum_of_weights() == np.sum(values * 0 + 1, axis=axis))
        assert ds.variance() == pytest.approx(np.var(values, axis=axis))
        size = ds.size().max()
        assert np.squeeze(
            ds.bins()).shape == values.sum(axis=axis).shape + (size, )

    check_axis(values, None)
    check_axis(values, (1, ))
    check_axis(values, (2, 3))
    check_axis(values, (1, 3, 5))

    with pytest.raises(ValueError):
        core.DescriptiveStatisticsFloat64(values, axis=(12, ))  # type: ignore
    with pytest.raises(ValueError):
        core.DescriptiveStatisticsFloat64(values, axis=(-1, ))  # type: ignore

# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pickle
#
import dask.array as da
import numpy as np
import pytest
import pyinterp
#
from .core.test_descriptive_statistics import weighted_mom3, weighted_mom4


@pytest.mark.parametrize("dtype, error", [(np.float32, 1e-4),
                                          (np.float64, 1e-6)])
def test_descriptive_statistics_1d(dtype, error):
    """Test the computation of descriptive statistics for a 1D array."""
    values = np.random.random_sample((10000, )).astype(dtype)
    ds = pyinterp.DescriptiveStatistics(values, dtype=dtype)

    def check_stats(ds, values):
        assert isinstance(ds, pyinterp.DescriptiveStatistics)
        assert ds.count() == values.size
        assert ds.max() == np.max(values)
        assert ds.mean() == pytest.approx(np.mean(values), rel=error)
        assert ds.min() == np.min(values)
        assert ds.sum_of_weights() == values.size
        assert ds.sum() == pytest.approx(np.sum(values), rel=error)
        assert ds.variance() == pytest.approx(np.var(values), rel=error)
        assert ds.kurtosis() == pytest.approx(weighted_mom4(
            values, np.ones(values.size)),
                                              rel=error)
        assert ds.skewness() == pytest.approx(weighted_mom3(
            values, np.ones(values.size)),
                                              rel=error)

    check_stats(ds, values)

    other = pickle.loads(pickle.dumps(ds))
    check_stats(other, values)

    ds = pyinterp.DescriptiveStatistics(values, weights=np.ones(values.size))
    check_stats(ds, values)

    ds = pyinterp.DescriptiveStatistics(da.asarray(values))
    check_stats(ds, values)


def test_axis():
    """Test the computation of descriptive statistics for a reduced tensor."""
    values = np.random.random_sample((2, 3, 4, 5, 6, 7))

    def check_axis(values, axis, delayed=False):
        ds = pyinterp.DescriptiveStatistics(
            da.asarray(values) if delayed else values, axis=axis)
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

    check_axis(values, None, delayed=True)
    check_axis(values, (1, ), delayed=True)
    check_axis(values, (2, 3), delayed=True)
    check_axis(values, (1, 3, 5), delayed=True)

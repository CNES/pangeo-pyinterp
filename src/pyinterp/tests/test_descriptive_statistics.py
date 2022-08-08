# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pickle

import dask.array as da
import numpy as np
import pytest

from . import load_grid2d, load_grid3d, load_grid4d
from .. import DescriptiveStatistics
from .core.test_descriptive_statistics import weighted_mom3, weighted_mom4


def check_stats(ds, values, dtype, error):
    """Check the statistics of a DescriptiveStatistics object."""
    assert isinstance(ds, DescriptiveStatistics)
    assert ds.count() == values.size
    assert ds.max() == np.max(values)
    assert ds.mean() == pytest.approx(np.mean(values), rel=error, abs=error)
    assert ds.min() == np.min(values)
    assert ds.sum_of_weights() == values.size
    assert ds.sum() == pytest.approx(np.sum(values), rel=error, abs=error)
    assert ds.var() == pytest.approx(np.var(values), rel=error, abs=error)
    assert ds.std() == pytest.approx(np.std(values), rel=error, abs=error)
    kurtosis = weighted_mom4(values, np.ones(values.size, dtype=dtype))
    assert ds.kurtosis() == pytest.approx(kurtosis, abs=error)
    skewness = weighted_mom3(values, np.ones(values.size, dtype=dtype))
    assert ds.skewness() == pytest.approx(skewness, rel=error, abs=error)


@pytest.mark.parametrize('dtype,error', [(np.float32, 1e-4),
                                         (np.float64, 1e-6)])
def test_descriptive_statistics_1d(dtype, error):
    """Test the computation of descriptive statistics for a 1D array."""
    values = np.random.random_sample((10000, )).astype(dtype)
    ds = DescriptiveStatistics(values, dtype=dtype)

    check_stats(ds, values, dtype, error)

    other = pickle.loads(pickle.dumps(ds))
    check_stats(other, values, dtype, error)

    ds = DescriptiveStatistics(values, weights=np.ones(values.size))
    check_stats(ds, values, dtype, error)

    ds = DescriptiveStatistics(da.from_array(values, chunks=(1000, )))
    check_stats(ds, values, dtype, error)

    assert isinstance(str(ds), str)


@pytest.mark.parametrize('dtype,error', [(np.float32, 1e-4),
                                         (np.float64, 1e-6)])
def test_descriptive_statistics_iadd(dtype, error):
    v0 = np.random.random_sample((5000, )).astype(dtype)
    ds = DescriptiveStatistics(v0, dtype=dtype)
    v1 = np.random.random_sample((5000, )).astype(dtype)
    ds += DescriptiveStatistics(v1, dtype=dtype)
    check_stats(ds, np.concatenate((v0, v1)), dtype, error)

    with pytest.raises(TypeError):
        ds += v1

    with pytest.raises(TypeError):
        ds2 = DescriptiveStatistics(v0,
                                    dtype=np.float32 if dtype == np.float64
                                    else np.float64)  # type: ignore
        ds += ds2


@pytest.mark.parametrize('dtype,error', [(np.float32, 1e-4),
                                         (np.float64, 1e-6)])
def test_descriptive_statistics_add(dtype, error):
    v0 = np.random.random_sample((5000, )).astype(dtype)
    ds = DescriptiveStatistics(v0, dtype=dtype)
    v1 = np.random.random_sample((5000, )).astype(dtype)
    ds = ds + DescriptiveStatistics(v1, dtype=dtype)
    check_stats(ds, np.concatenate((v0, v1)), dtype, error)

    with pytest.raises(TypeError):
        ds + v1

    with pytest.raises(TypeError):
        ds2 = DescriptiveStatistics(v0,
                                    dtype=np.float32 if dtype == np.float64
                                    else np.float64)  # type: ignore
        ds + ds2


def test_array():
    """Test the computation of descriptive statistics for a tensor."""
    values = np.random.random_sample((2, 20, 30))
    ds = DescriptiveStatistics(values, axis=(0, ))

    array = ds.array()
    assert array.shape == (20, 30)

    assert np.all(ds.count() == array['count'])
    assert np.all(ds.max() == array['max'])
    assert np.all(ds.mean() == array['mean'])
    assert np.all(ds.min() == array['min'])
    assert np.all(ds.sum() == array['sum'])
    assert np.all(ds.sum_of_weights() == array['sum_of_weights'])
    assert np.all(ds.var() == array['var'])
    assert np.all(ds.kurtosis() == array['kurtosis'])
    assert np.all(ds.skewness() == array['skewness'])

    with pytest.raises(ValueError):
        ds = DescriptiveStatistics(values, axis=(0, ), dtype=np.dtype('S1'))


def test_axis():
    """Test the computation of descriptive statistics for a reduced tensor."""
    values = np.random.random_sample((2, 3, 4, 5, 6, 7))

    def check_axis(values, axis, delayed=False):
        ds = DescriptiveStatistics(da.asarray(values) if delayed else values,
                                   axis=axis)
        assert np.all(ds.count() == np.sum(values * 0 + 1, axis=axis))
        assert np.all(ds.max() == np.max(values, axis=axis))
        assert ds.mean() == pytest.approx(np.mean(values, axis=axis))
        assert np.all(ds.min() == np.min(values, axis=axis))
        assert np.all(ds.sum_of_weights() == np.sum(values * 0 + 1, axis=axis))
        assert ds.sum() == pytest.approx(np.sum(values, axis=axis))
        assert ds.var() == pytest.approx(np.var(values, axis=axis))

    check_axis(values, None)
    check_axis(values, 1)
    check_axis(values, (2, 3))
    check_axis(values, (1, 3, 5))

    check_axis(values, None, delayed=True)
    check_axis(values, (1, ), delayed=True)
    check_axis(values, (2, 3), delayed=True)
    check_axis(values, (1, 3, 5), delayed=True)


def test_grid():
    """Test the computation of descriptive statistics for a grid."""
    data = load_grid2d().mss
    ds = DescriptiveStatistics(data)
    assert ds.mean()[0] == pytest.approx(data.mean())

    data = load_grid3d().tcw
    ds = DescriptiveStatistics(data, axis=(0, ))
    assert ds.mean() == pytest.approx(data.mean(axis=0))

    data = load_grid4d().pressure
    ds = DescriptiveStatistics(data, axis=(0, 1))
    assert ds.mean() == pytest.approx(data.mean(axis=(0, 1)))

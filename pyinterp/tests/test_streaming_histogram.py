# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for the StreamingHistogram class."""
import pickle

import dask.array as da
import numpy as np
import pytest

from . import load_grid2d, load_grid3d, load_grid4d
from .. import StreamingHistogram
from ..typing import NDArray
from .core.test_descriptive_statistics import weighted_mom3, weighted_mom4


def check_stats(
    histogram: StreamingHistogram,
    values: NDArray,
    dtype: np.dtype,
    error: float,
) -> None:
    """Check the statistics of a streaming histogram."""
    assert isinstance(histogram, StreamingHistogram)
    assert histogram.count() == values.size
    assert histogram.size() == values.size
    assert histogram.max() == np.max(values)
    assert histogram.mean() == pytest.approx(np.mean(values),
                                             rel=error,
                                             abs=error)
    assert histogram.min() == np.min(values)
    assert histogram.sum_of_weights() == values.size
    assert histogram.quantile() == pytest.approx(np.quantile(values, 0.5),
                                                 rel=error,
                                                 abs=error)
    assert histogram.var() == pytest.approx(np.var(values),
                                            rel=error,
                                            abs=error)
    assert histogram.std() == pytest.approx(np.std(values),
                                            rel=error,
                                            abs=error)
    kurtosis = weighted_mom4(values, np.ones(values.size, dtype=dtype))
    assert histogram.kurtosis() == pytest.approx(kurtosis, abs=error)
    skewness = weighted_mom3(values, np.ones(values.size, dtype=dtype))
    assert histogram.skewness() == pytest.approx(skewness,
                                                 rel=error,
                                                 abs=error)


@pytest.mark.parametrize(('dtype', 'error'), [(np.float32, 1e-4),
                                              (np.float64, 1e-6)])
def test_streaming_histogram_1d(dtype: np.dtype, error: float) -> None:
    """Test the computation of streaming histogram for a 1D array."""
    rng = np.random.default_rng()
    values = rng.random((10000, )).astype(dtype)
    histogram = StreamingHistogram(values, dtype=dtype, bin_count=values.size)

    check_stats(histogram, values, dtype, error)
    assert np.all(histogram.bins()['value'] == np.sort(values))
    assert np.all(histogram.bins()['weight'] == np.ones_like(values))

    other = pickle.loads(pickle.dumps(histogram))
    check_stats(other, values, dtype, error)

    histogram = StreamingHistogram(values,
                                   weights=np.ones(values.size),
                                   bin_count=values.size)
    check_stats(histogram, values, dtype, error)

    histogram = StreamingHistogram(da.from_array(values, chunks=(1000, )),
                                   bin_count=values.size)
    check_stats(histogram, values, dtype, error)

    assert isinstance(str(histogram), str)


def test_streaming_histogram_iadd() -> None:
    """Test the addition of two StreamingHistogram objects in place."""
    size = 5000 * 2
    rng = np.random.default_rng()
    v0 = rng.random((size // 2, ))
    ds = StreamingHistogram(v0, dtype=np.dtype(np.float64), bin_count=size)
    v1 = rng.random((size // 2, ))
    ds += StreamingHistogram(v1, dtype=np.dtype(np.float64), bin_count=size)
    check_stats(ds, np.concatenate((v0, v1)), np.dtype(np.float64), 1e-6)

    with pytest.raises(TypeError):
        ds += v1

    with pytest.raises(TypeError):
        ds2 = StreamingHistogram(v0.astype('float32'),
                                 dtype=np.dtype(np.float32),
                                 bin_count=size)
        ds += ds2

    with pytest.raises(ValueError):
        StreamingHistogram(v0, dtype=np.dtype(np.int32), bin_count=size)


def test_axis() -> None:
    """Test the computation of streaming histogram for a reduced tensor."""
    rng = np.random.default_rng()
    values = rng.random((2, 3, 4, 5, 6, 7))

    def check_axis(
        values: NDArray,
        axis: int | tuple[int, ...] | None,
        delayed: bool = False,
    ) -> None:
        histogram = StreamingHistogram(
            da.asarray(values) if delayed else values,
            axis=axis,
            bin_count=values.size)
        assert np.all(histogram.count() == np.sum(values * 0 + 1, axis=axis))
        assert np.all(histogram.max() == np.max(values, axis=axis))
        assert histogram.mean() == pytest.approx(np.mean(values, axis=axis))
        assert np.all(histogram.min() == np.min(values, axis=axis))
        assert np.all(
            histogram.sum_of_weights() == np.sum(values * 0 + 1, axis=axis))
        assert histogram.quantile() == pytest.approx(
            np.quantile(values, 0.5, axis=axis))
        assert histogram.var() == pytest.approx(np.var(values, axis=axis))

    check_axis(values, None)
    check_axis(values, 1)
    check_axis(values, (2, 3))
    check_axis(values, (1, 3, 5))

    check_axis(values, None, delayed=True)
    check_axis(values, (1, ), delayed=True)
    check_axis(values, (2, 3), delayed=True)
    check_axis(values, (1, 3, 5), delayed=True)


def test_grid() -> None:
    """Test the computation of streaming histogram for a grid."""
    data = load_grid2d().mss
    histogram = StreamingHistogram(data)
    assert histogram.mean()[0] == pytest.approx(data.mean(), abs=1e-4)

    data = load_grid3d().tcw
    histogram = StreamingHistogram(data, axis=(0, ))
    assert histogram.mean() == pytest.approx(data.mean(axis=0), abs=1e-4)

    data = load_grid4d().pressure
    histogram = StreamingHistogram(data, axis=(0, 1))
    assert histogram.mean() == pytest.approx(data.mean(axis=(0, 1)), abs=1e-4)

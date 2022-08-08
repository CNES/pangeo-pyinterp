# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import datetime
import pickle

import numpy as np
import pytest

from .. import TemporalAxis


def new_axis(timedelta: bool = False):
    start = datetime.datetime(2000, 1, 1)
    if timedelta:
        values = np.array(
            [datetime.timedelta(seconds=index) for index in range(86400)],
            dtype='timedelta64[us]')
    else:
        values = np.array([
            start + datetime.timedelta(seconds=index) for index in range(86400)
        ],
                          dtype='datetime64[us]')
    return values, TemporalAxis(values)


def test_datetime64_constructor():
    values, axis = new_axis()
    assert isinstance(str(axis), str)
    assert str(axis) == ("""<pyinterp.core.TemporalAxis>
  min_value: 2000-01-01T00:00:00.000000
  max_value: 2000-01-01T23:59:59.000000
  step     : 1000000 microseconds""")
    assert axis.dtype() == np.dtype('datetime64[us]')
    assert np.any(values != axis.safe_cast(values.astype('datetime64[h]')))
    assert axis.increment() == np.timedelta64(1000000, 'us')
    assert axis.front() == np.datetime64('2000-01-01')
    assert axis.back() == np.datetime64('2000-01-01T23:59:59')
    assert axis[0] == np.datetime64('2000-01-01')
    assert axis.min_value() == np.datetime64('2000-01-01')
    assert axis.max_value() == np.datetime64('2000-01-01T23:59:59')
    assert np.all(
        axis.find_index(np.array(['2000-01-01', '2000-02-01'],
                                 dtype='datetime64'),
                        bounded=False) == [0, -1])
    assert np.all(
        axis.find_index(np.array(['2000-01-01', '2000-02-01'],
                                 dtype='datetime64'),
                        bounded=True) == [0, 86399])
    assert np.all(axis[10:20] == values[10:20])
    axis.flip(inplace=True)
    assert axis.increment() == np.timedelta64(-1000000, 'us')
    assert axis.back() == np.datetime64('2000-01-01')
    assert axis.front() == np.datetime64('2000-01-01T23:59:59')
    assert axis.min_value() == np.datetime64('2000-01-01')
    assert axis.max_value() == np.datetime64('2000-01-01T23:59:59')
    assert np.all(
        axis.find_index(np.array(['2000-01-01', '2000-02-01'],
                                 dtype='datetime64'),
                        bounded=False) == [86399, -1])

    assert np.all(
        axis.find_indexes(
            np.array(['2000-01-01', '2000-02-01'], dtype='datetime64')) ==
        [[86398, 86399], [-1, -1]])

    axis = TemporalAxis(values.astype('datetime64[s]'))
    with pytest.warns(UserWarning):
        axis.safe_cast(values)


def test_timedelta64_constructor():
    values, axis = new_axis(timedelta=True)
    assert isinstance(str(axis), str)
    assert str(axis) == ("""<pyinterp.core.TemporalAxis>
  min_value: 0 microseconds
  max_value: 86399000000 microseconds
  step     : 1000000 microseconds""")
    assert np.any(values != axis.safe_cast(values.astype('timedelta64[h]')))
    assert axis.increment() == np.timedelta64(1000000, 'us')
    assert axis.front() == np.timedelta64(0)
    assert axis.back() == np.timedelta64('86399', 's')
    assert axis[0] == np.timedelta64(0)
    assert axis.min_value() == np.timedelta64(0)
    assert axis.max_value() == np.timedelta64(86399, 's')
    assert np.all(
        axis.find_index(np.array([0, 86400], dtype='timedelta64[s]'),
                        bounded=False) == [0, -1])
    assert np.all(
        axis.find_index(np.array([0, 86400], dtype='timedelta64[s]'),
                        bounded=True) == [0, 86399])
    assert np.all(axis[10:20] == values[10:20])
    axis.flip(inplace=True)
    assert axis.increment() == np.timedelta64(-1000000, 'us')
    assert axis.back() == np.timedelta64(0)
    assert axis.front() == np.timedelta64(86399, 's')
    assert axis.min_value() == np.timedelta64(0)
    assert axis.max_value() == np.timedelta64(86399, 's')
    assert np.all(
        axis.find_index(np.array([0, 86400], dtype='timedelta64[s]'),
                        bounded=False) == [86399, -1])
    assert np.all(
        axis.find_indexes(np.array([0, 86400], dtype='timedelta64[s]')) ==
        [[86398, 86399], [-1, -1]])

    axis = TemporalAxis(values.astype('timedelta64[s]'))
    with pytest.warns(UserWarning):
        axis.safe_cast(values)


def test_temporal_axis_degraded():
    with pytest.raises(ValueError):
        TemporalAxis(np.arange(10))

    axis = TemporalAxis(
        np.array(['2000-01-01', '2000-02-01'], dtype='datetime64[s]'))
    with pytest.raises(ValueError):
        axis.safe_cast(np.arange(2))

    assert axis.safe_cast(
        np.array(['2000-01-01', '2000-02-01'],
                 dtype='datetime64[D]')).dtype == np.dtype('datetime64[s]')


def test_pickle():
    values, axis = new_axis()
    other = pickle.loads(pickle.dumps(axis))
    assert axis == other
    assert id(axis) != id(other)
    assert np.all(values[:] == axis[:])
    assert np.all(values[:] == other[:])

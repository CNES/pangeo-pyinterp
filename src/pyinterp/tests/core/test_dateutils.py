# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
from typing import List, Tuple
import datetime
import random

import numpy as np
import pytest

from ... import core


def make_date(samples=10000,
              resolution='us') -> Tuple[List[datetime.datetime], np.ndarray]:
    epoch = datetime.datetime(1970, 1, 1)
    delta = datetime.datetime.now() - datetime.datetime(1970, 1, 1)

    pydates = [epoch + random.random() * delta for _ in range(samples)]
    pydates += [epoch - random.random() * delta for _ in range(samples)]
    npdates = np.array(pydates).astype(f'datetime64[{resolution}]')

    return pydates, npdates


@pytest.mark.parametrize('resolution', ['ms', 'us', 'ns'])
def test_date(resolution):
    pydates, npdates = make_date(resolution=resolution)
    yms = core.dateutils.date(npdates)

    for ix, item in enumerate(yms):
        expected = pydates[ix]
        assert item['year'] == expected.year
        assert item['month'] == expected.month
        assert item['day'] == expected.day


@pytest.mark.parametrize('resolution', ['ms', 'us', 'ns'])
def test_datetime(resolution):
    expected, npdates = make_date(resolution=resolution)
    pydates = core.dateutils.datetime(npdates)

    for ix, item in enumerate(pydates):
        if resolution == 'ms':
            expected[ix] = expected[ix].replace(
                microsecond=expected[ix].microsecond // 1000 * 1000)
        assert item == expected[ix]


@pytest.mark.parametrize('resolution', ['ms', 'us', 'ns'])
def test_timedelta_since_january(resolution):
    pydates, npdates = make_date(resolution=resolution)
    days = core.dateutils.timedelta_since_january(npdates)

    for ix, item in enumerate(days):
        expected = pydates[ix].utctimetuple().tm_yday
        yday = item.astype('timedelta64[D]').astype('int')
        assert yday + 1 == expected
        microseconds = int(item.astype('timedelta64[us]').astype('int64'))
        dt = datetime.timedelta(microseconds=microseconds)
        minute, second = divmod(dt.seconds, 60)
        hour, minute = divmod(minute, 60)
        assert hour == pydates[ix].hour
        assert minute == pydates[ix].minute
        assert second == pydates[ix].second
        if resolution == 'ms':
            pydates[ix] = pydates[ix].replace(
                microsecond=pydates[ix].microsecond // 1000 * 1000)
        assert dt.microseconds == pydates[ix].microsecond


@pytest.mark.parametrize('resolution', ['ms', 'us', 'ns'])
def test_isocalendar(resolution):
    pydates, npdates = make_date(resolution=resolution)
    isocalendar = core.dateutils.isocalendar(npdates)

    for ix, item in enumerate(isocalendar):
        year, week, weekday = pydates[ix].isocalendar()
        assert item['year'] == year
        assert item['week'] == week
        assert item['weekday'] == weekday


@pytest.mark.parametrize('resolution', ['ms', 'us', 'ns'])
def test_time(resolution):
    pydates, npdates = make_date(resolution=resolution)
    hms = core.dateutils.time(npdates)

    for ix, item in enumerate(hms):
        expected = pydates[ix]
        assert item['hour'] == expected.hour
        assert item['minute'] == expected.minute
        assert item['second'] == expected.second


@pytest.mark.parametrize('resolution', ['ms', 'us', 'ns'])
def test_weekday(resolution):
    pydates, npdates = make_date(resolution=resolution)
    weekday = core.dateutils.weekday(npdates)

    for ix, item in enumerate(weekday):
        _, _, weekday = pydates[ix].isocalendar()
        assert item == weekday % 7


def test_wrong_units():
    _, npdates = make_date(10)

    with pytest.raises(ValueError):
        core.dateutils.date(npdates.astype('datetime64[h]'))

    with pytest.raises(ValueError):
        core.dateutils.date(npdates.reshape(5, 2))


@pytest.mark.parametrize(
    'resolution',
    ['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'as'])
def test_format_date(resolution):
    _, npdates = make_date(60, resolution=resolution)
    for item in npdates:
        value = item.astype('int64')
        calculated = core.dateutils.datetime64_to_str(
            value, f'datetime64[{resolution}]')
        expected = str(item)
        assert calculated == expected

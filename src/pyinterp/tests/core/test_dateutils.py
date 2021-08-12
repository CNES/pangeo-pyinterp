# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
from typing import List, Tuple
import datetime
import random
import numpy as np
import pytest
import pyinterp.core as core


def make_date(samples=10000) -> Tuple[List[datetime.datetime], np.ndarray]:
    epoch = datetime.datetime(1970, 1, 1)
    delta = datetime.datetime.now() - datetime.datetime(1970, 1, 1)

    pydates = [epoch + random.random() * delta for _ in range(samples)]
    npdates = np.array(pydates).astype("datetime64[ns]")

    return pydates, npdates


def test_date():
    pydates, npdates = make_date()
    yms = core.dateutils.date(npdates)

    for ix, item in enumerate(yms):
        expected = pydates[ix]
        assert item['year'] == expected.year
        assert item['month'] == expected.month
        assert item['day'] == expected.day


def test_datetime():
    expected, npdates = make_date()
    pydates = core.dateutils.datetime(npdates)

    for ix, item in enumerate(pydates):
        assert item == expected[ix]


def test_timedelta_since_january():
    pydates, npdates = make_date()
    days = core.dateutils.timedelta_since_january(npdates)

    for ix, item in enumerate(days):
        expected = pydates[ix].utctimetuple().tm_yday
        yday = item.astype("timedelta64[D]").astype("int")
        assert yday + 1 == expected
        microseconds = int(item.astype("timedelta64[us]").astype("int64"))
        dt = datetime.timedelta(microseconds=microseconds)
        minute, second = divmod(dt.seconds, 60)
        hour, minute = divmod(minute, 60)
        assert hour == pydates[ix].hour
        assert minute == pydates[ix].minute
        assert second == pydates[ix].second
        assert dt.microseconds == pydates[ix].microsecond


def test_isocalendar():
    pydates, npdates = make_date()
    isocalendar = core.dateutils.isocalendar(npdates)

    for ix, item in enumerate(isocalendar):
        year, week, weekday = pydates[ix].isocalendar()
        assert item['year'] == year
        assert item['week'] == week
        assert item['weekday'] == weekday


def test_time():
    pydates, npdates = make_date()
    hms = core.dateutils.time(npdates)

    for ix, item in enumerate(hms):
        expected = pydates[ix]
        assert item['hour'] == expected.hour
        assert item['minute'] == expected.minute
        assert item['second'] == expected.second


def test_weekday():
    pydates, npdates = make_date()
    weekday = core.dateutils.weekday(npdates)

    for ix, item in enumerate(weekday):
        _, _, weekday = pydates[ix].isocalendar()
        assert item == weekday % 7


def test_wrong_units():
    _, npdates = make_date(10)

    with pytest.raises(ValueError):
        core.dateutils.date(npdates.astype("datetime64[h]"))

    with pytest.raises(ValueError):
        core.dateutils.date(npdates.reshape(5, 2))

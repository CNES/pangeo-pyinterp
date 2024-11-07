import pickle

import numpy
import pytest

from ..period import Period, PeriodList


def datetime64(value: int) -> numpy.datetime64:
    """Converts an integer to a datetime64."""
    return numpy.datetime64(value, 'D')


def timedelta64(value: int) -> numpy.timedelta64:
    """Converts an integer to a timedelta64."""
    return numpy.timedelta64(value, 'D')


def make_period(start: int, end: int, within: bool = False) -> Period:
    """Creates a period from two integers."""
    return Period(datetime64(start), datetime64(end), within=within)


def period1(within=False):
    """Creates the period [1, 10)"""
    return make_period(1, 10, within=within)


def period2():
    """Creates the period [5, 30)"""
    return make_period(5, 30)


def period3():
    """Creates the period [35, 81)"""
    return make_period(35, 81)


def test_interface():
    """Tests the interface of the Period class."""
    p1 = period1()
    assert p1.begin == datetime64(1)
    assert p1.last == datetime64(9)
    assert p1.end() == datetime64(10)
    assert p1.duration() == timedelta64(9)
    assert not p1.is_null()
    assert len(p1) == 9

    assert str(p1) == '[1970-01-02, 1970-01-11)'
    # With numpy 2.0.0rc1, the representation of a datetime64 is different
    # from the one of numpy 1.21.2: "numpy" becomes "np".
    if numpy.version.version >= '2.0.0':
        assert repr(p1) == ("Period(begin=np.datetime64('1970-01-02'), "
                            "end=np.datetime64('1970-01-10'), within=True)")
    else:
        assert repr(p1) == ("Period(begin=numpy.datetime64('1970-01-02'), "
                            "end=numpy.datetime64('1970-01-10'), within=True)")

    assert pickle.loads(pickle.dumps(p1)) == p1

    p1 = period1(within=True)
    assert p1.begin == datetime64(1)
    assert p1.last == datetime64(10)
    assert p1.end() == datetime64(11)
    assert p1.duration() == timedelta64(10)
    assert not p1.is_null()
    assert len(p1) == 10

    p2 = period2()
    assert p2.begin == datetime64(5)
    assert p2.last == datetime64(29)
    assert p2.end() == datetime64(30)
    assert p2.duration() == timedelta64(25)
    assert not p2.is_null()
    assert len(p2) == 25


def test_cmp():
    """Tests the comparison operators of the Period class."""
    p1 = period1()
    p2 = period2()
    p3 = period3()

    assert p1 == period1()
    assert p1 != p2
    assert p1 < p3
    assert p3 > p2


def test_relation():
    """Tests the relation operators of the Period class."""
    p1 = period1()
    p2 = period2()
    p3 = period3()

    assert p2.contains(datetime64(20))
    assert not p2.contains(datetime64(2))

    assert p1.contains(make_period(2, 8))
    assert not p1.contains(p3)

    assert p1.intersects(p2)
    assert p2.intersects(p1)

    assert p1.is_adjacent(make_period(-5, 1))
    assert p1.is_adjacent(make_period(10, 20))
    assert not p1.is_adjacent(p3)

    assert p1.is_before(datetime64(15))
    assert p3.is_after(datetime64(15))

    assert p1.intersection(p2) == make_period(5, 10)
    assert p1.intersection(p3).is_null()

    assert p1.merge(p2) == make_period(1, 30)
    assert p1.merge(p3).is_null()


def test_zero_length_period():
    """Tests the behavior of a zero-length period."""
    zero_len = make_period(3, 3)
    assert len(zero_len) == 0
    assert make_period(1, 1) == make_period(1, 1)
    assert make_period(3, 3) == zero_len

    # zero_length period always returns false for is_before & is_after
    assert not zero_len.is_before(datetime64(5))
    assert not zero_len.is_after(datetime64(5))
    assert not zero_len.is_before(datetime64(-5))
    assert not zero_len.is_after(datetime64(-5))

    assert zero_len.is_null()
    assert not zero_len.contains(datetime64(20))
    # a null_period cannot contain any points
    assert not zero_len.contains(datetime64(3))
    assert not zero_len.contains(make_period(5, 8))

    p1 = period1()
    assert p1.contains(zero_len)
    assert zero_len.intersects(p1)
    assert p1.intersects(zero_len)
    assert zero_len.is_adjacent(make_period(-10, 3))
    assert make_period(-10, 3).is_adjacent(zero_len)
    assert zero_len.intersection(p1) == zero_len


def test_invalid_period():
    """Tests the behavior of a null period."""
    null_per = make_period(5, 1)
    with pytest.raises(ValueError):
        assert len(null_per) == 0

    assert not null_per.is_before(datetime64(7))
    assert not null_per.is_after(datetime64(7))
    assert not null_per.is_before(datetime64(-5))
    assert not null_per.is_after(datetime64(-5))

    assert null_per.is_null()
    assert not null_per.contains(datetime64(20))
    assert not null_per.contains(datetime64(3))
    assert not null_per.contains(make_period(7, 9))
    p1 = period1()
    assert p1.contains(null_per)
    assert null_per.intersects(p1)
    assert p1.intersects(null_per)
    assert null_per.is_adjacent(make_period(-10, 5))
    assert null_per.is_adjacent(make_period(1, 10))


def test_invalid():
    """Tests the behavior of invalid periods."""
    p1x = make_period(0, -2)
    assert p1x.begin == datetime64(0)
    assert p1x.last == datetime64(-3)
    assert p1x.end() == datetime64(-2)
    assert p1x.duration() == timedelta64(-2)
    assert p1x.is_null()

    p1x = make_period(0, -1)
    assert p1x.begin == datetime64(0)
    assert p1x.last == datetime64(-2)
    assert p1x.end() == datetime64(-1)
    assert p1x.duration() == timedelta64(-1)
    assert p1x.is_null()

    p1x = make_period(0, 0)
    assert p1x.begin == datetime64(0)
    assert p1x.last == datetime64(-1)
    assert p1x.end() == datetime64(0)
    assert p1x.duration() == timedelta64(0)
    assert p1x.is_null()

    p1x = make_period(0, 1)
    assert p1x.begin == datetime64(0)
    assert p1x.last == datetime64(0)
    assert p1x.end() == datetime64(1)
    assert p1x.duration() == timedelta64(1)
    assert not p1x.is_null()

    p1x = make_period(0, 2)
    assert p1x.begin == datetime64(0)
    assert p1x.last == datetime64(1)
    assert p1x.end() == datetime64(2)
    assert p1x.duration() == timedelta64(2)
    assert not p1x.is_null()

    p1x = make_period(0, -1)
    assert p1x.begin == datetime64(0)
    assert p1x.last == datetime64(-2)
    assert p1x.end() == datetime64(-1)
    assert p1x.duration() == timedelta64(-1)
    assert p1x.is_null()

    p1x = make_period(0, -2)
    assert p1x.begin == datetime64(0)
    assert p1x.last == datetime64(-3)
    assert p1x.end() == datetime64(-2)
    assert p1x.duration() == timedelta64(-2)
    assert p1x.is_null()

    p1x = make_period(0, 0)
    assert p1x.begin == datetime64(0)
    assert p1x.last == datetime64(-1)
    assert p1x.end() == datetime64(0)
    assert p1x.duration() == timedelta64(0)
    assert p1x.is_null()

    p1x = make_period(0, 1)
    assert p1x.begin == datetime64(0)
    assert p1x.last == datetime64(0)
    assert p1x.end() == datetime64(1)
    assert p1x.duration() == timedelta64(1)
    assert not p1x.is_null()

    p1x = make_period(0, 2)
    assert p1x.begin == datetime64(0)
    assert p1x.last == datetime64(1)
    assert p1x.end() == datetime64(2)
    assert p1x.duration() == timedelta64(2)
    assert not p1x.is_null()

    p1x = make_period(1, 1)
    p2x = make_period(1, 2)
    p3x = make_period(1, 3)
    assert p1x.duration() == timedelta64(0)
    assert p2x.duration() == timedelta64(1)
    assert p3x.duration() == timedelta64(2)
    assert p1x.is_null()
    assert not p2x.is_null()


def test_period_as_resolution():
    """Tests the conversion of the period to a period with a different
    resolution."""
    begin = numpy.datetime64('2019-01-01')
    end = numpy.datetime64('2019-01-03')
    period = Period(begin, end, within=True)
    assert period.as_resolution('M') == Period(
        numpy.datetime64('2019-01-01', 'M'),
        numpy.datetime64('2019-01-31', 'M'),
        within=True)
    assert period.as_resolution('Y') == Period(
        numpy.datetime64('2019-01-01', 'Y'),
        numpy.datetime64('2019-12-31', 'Y'),
        within=True)
    assert period.as_resolution('D') == period


def test_period_from_different_resolutions():
    """Tests the conversion of the period to a period with a different
    resolution."""
    begin = numpy.datetime64('2019-01-01T00:00:00', 's')
    end = numpy.datetime64('2019-01-03T00:00:00', 'D')
    period = Period(begin, end, within=True)
    assert period.resolution == 's'
    assert str(period) == '[2019-01-01T00:00:00, 2019-01-03T00:00:01)'


def test_period_list():
    """Tests the PeriodList class."""
    start = numpy.datetime64('2018-01-01', 's')
    end = numpy.datetime64('2019-01-01', 's')

    duration = numpy.timedelta64(1, 'D')
    samples = (end - start) // duration

    mask = numpy.random.randint(0, 2, samples)
    periods = []

    for ix in range(samples):
        if mask[ix]:
            periods.append(
                (start + ix * duration, start + (ix + 1) * duration))
    periods = PeriodList(numpy.array(periods).T)
    arr = periods.periods
    assert numpy.any((arr['begin'][1:] - arr['last'][:-1])  # type: ignore
                     == numpy.timedelta64(0, 's'))
    periods = periods.join_adjacent_periods()
    arr = periods.periods
    assert numpy.all((arr['begin'][1:] - arr['last'][:-1])  # type: ignore
                     != numpy.timedelta64(0, 's'))
    assert numpy.all(
        periods.filter(numpy.timedelta64(2, 'D')).duration() >=
        numpy.timedelta64(2, 'D'))


def test_period_list_empty():
    """Tests an empty PeriodList."""
    instance = PeriodList.empty()
    assert len(instance.periods) == 0

    other = pickle.loads(pickle.dumps(instance))
    assert len(other.periods) == 0

    assert len(instance.join_adjacent_periods()) == 0
    assert len(instance.merge(instance)) == 0

    assert len(instance.filter(numpy.timedelta64(1, 'D'))) == 0

    assert numpy.sum(instance.duration()) == 0

    start = numpy.datetime64('2018-01-01', 'ns')
    end = numpy.datetime64('2018-01-02', 'ns')

    duration = numpy.timedelta64(1, 'h')
    samples = (end - start) // duration

    periods = []

    for ix in range(samples):
        periods.append(
            (start + ix * duration,
             start + (ix + 1) * duration - numpy.timedelta64(1, 's')))
    periods = PeriodList(numpy.array(periods).T)

    assert numpy.all(instance.merge(periods).periods == periods.periods)

    instance = PeriodList.empty()

    assert len(instance.sort()) == 0


def test_period_list_cross_a_period():
    """Tests the PeriodList.cross_a_period method."""
    periods = numpy.array(
        [['2019-12-01T02:33:57.989', '2019-12-01T03:24:59.504'],
         ['2019-12-01T04:16:09.242', '2019-12-01T05:07:10.757'],
         ['2019-12-01T05:58:20.495', '2019-12-01T06:49:22.010'],
         ['2019-12-01T07:40:31.748', '2019-12-01T08:31:33.263'],
         ['2019-12-02T00:42:24.278', '2019-12-02T01:33:25.793']],
        dtype='datetime64[ms]',
    )
    handler = PeriodList(periods.T)

    dates = numpy.arange(numpy.datetime64('2019-12-02T00:42', 'ms'),
                         numpy.datetime64('2019-12-02T01:40', 'ms'),
                         numpy.timedelta64(1, 's'))
    flags = handler.cross_a_period(dates)
    indices = numpy.where(~flags)
    assert numpy.all(
        dates[indices] > numpy.datetime64('2019-12-02T01:33:25.794', 'ms'))

    dates = numpy.arange(numpy.datetime64('2019-12-02T00:42', 'ms'),
                         numpy.datetime64('2019-12-02T01:33', 'ms'),
                         numpy.timedelta64(1, 's'))
    flags = handler.cross_a_period(dates)
    assert numpy.all(flags)

    dates = numpy.arange(numpy.datetime64('2019-12-01T08:32', 'ms'),
                         numpy.datetime64('2019-12-02T00:40', 'ms'),
                         numpy.timedelta64(1, 's'))
    flags = handler.cross_a_period(dates)
    assert numpy.all(~flags)

    dates = numpy.arange(numpy.datetime64('2019-12-01T08:30', 'ms'),
                         numpy.datetime64('2019-12-02T00:40', 'us'),
                         numpy.timedelta64(1, 's'))
    flags = handler.cross_a_period(dates)
    assert not numpy.all(~flags)
    index = numpy.max(numpy.where(flags)[0])
    assert dates[index] > numpy.datetime64('2019-12-01T08:30')

    other = pickle.loads(pickle.dumps(handler))
    assert numpy.all(other.belong_to_a_period(dates) == flags)

    dates = numpy.arange(numpy.datetime64('2019-12-01T03:30:00.000', 'ms'),
                         numpy.datetime64('2019-12-01T04:00:00.000', 'ms'),
                         numpy.timedelta64(1, 's'))
    flags = handler.cross_a_period(dates)
    assert numpy.all(~flags)

    dates = numpy.arange(numpy.datetime64('2019-12-01T03:20:00.000', 'ms'),
                         numpy.datetime64('2019-12-01T04:00:00.000', 'ms'),
                         numpy.timedelta64(1, 's'))
    flags = handler.cross_a_period(dates)
    assert dates[~flags][0] > numpy.datetime64('2019-12-01T03:24:59.504', 'ms')


def test_period_list_belong_to_a_period():
    """Tests the PeriodList.belong_to_a_period method."""
    periods = numpy.array(
        [['2019-12-01T02:33:57.989', '2019-12-01T03:24:59.504'],
         ['2019-12-01T04:16:09.242', '2019-12-01T05:07:10.757'],
         ['2019-12-01T05:58:20.495', '2019-12-01T06:49:22.010'],
         ['2019-12-01T07:40:31.748', '2019-12-01T08:31:33.263'],
         ['2019-12-02T00:42:24.278', '2019-12-02T01:33:25.793']],
        dtype='datetime64[ms]',
    )
    handler = PeriodList(periods.T)
    dates = numpy.arange(numpy.datetime64('2019-12-01T00:00', 'ms'),
                         numpy.datetime64('2019-12-02T01:40', 'ms'),
                         numpy.timedelta64(1, 's'))
    flags = handler.belong_to_a_period(dates)
    mask = ((dates >= numpy.datetime64('2019-12-01T02:33:57.989', 'ms')) &
            (dates <= numpy.datetime64('2019-12-01T03:24:59.504', 'ms')))
    mask |= ((dates >= numpy.datetime64('2019-12-01T04:16:09.242', 'ms')) &
             (dates <= numpy.datetime64('2019-12-01T05:07:10.757', 'ms')))
    mask |= ((dates >= numpy.datetime64('2019-12-01T05:58:20.495', 'ms')) &
             (dates <= numpy.datetime64('2019-12-01T06:49:22.010', 'ms')))
    mask |= ((dates >= numpy.datetime64('2019-12-01T07:40:31.748', 'ms')) &
             (dates <= numpy.datetime64('2019-12-01T08:31:33.263', 'ms')))
    mask |= ((dates >= numpy.datetime64('2019-12-02T00:42:24.278', 'ms')) &
             (dates <= numpy.datetime64('2019-12-02T01:33:25.793', 'ms')))
    assert numpy.all(flags == mask)


def test_eclipse():
    periods = numpy.array(
        [['2019-12-10T11:02:53.581', '2019-12-10T11:02:53.581'],
         ['2019-12-10T11:03:03.494', '2019-12-10T11:03:03.494'],
         ['2019-12-11T00:46:11.517', '2019-12-11T00:46:11.517'],
         ['2019-12-11T01:20:02.205', '2019-12-11T01:20:02.205'],
         ['2019-12-12T00:46:41.879', '2019-12-12T00:46:41.879'],
         ['2019-12-12T01:20:45.555', '2019-12-12T01:20:45.555'],
         ['2019-12-13T00:47:13.629', '2019-12-13T00:47:13.629'],
         ['2019-12-13T01:21:28.255', '2019-12-13T01:21:28.255'],
         ['2019-12-14T00:47:46.709', '2019-12-14T00:47:46.709'],
         ['2019-12-14T01:22:10.361', '2019-12-14T01:22:10.361'],
         ['2019-12-15T00:48:21.092', '2019-12-15T00:48:21.092'],
         ['2019-12-15T01:22:51.942', '2019-12-15T01:22:51.942'],
         ['2019-12-16T00:48:56.771', '2019-12-16T00:48:56.771'],
         ['2019-12-16T01:23:33.071', '2019-12-16T01:23:33.071'],
         ['2019-12-17T00:49:33.750', '2019-12-17T00:49:33.750'],
         ['2019-12-17T01:24:13.812', '2019-12-17T01:24:13.812'],
         ['2019-12-18T00:50:12.161', '2019-12-18T00:50:12.161'],
         ['2019-12-18T01:24:54.342', '2019-12-18T01:24:54.342'],
         ['2019-12-19T00:50:52.012', '2019-12-19T00:50:52.012'],
         ['2019-12-19T01:25:34.706', '2019-12-19T01:25:34.706'],
         ['2019-12-20T00:51:33.186', '2019-12-20T00:51:33.186'],
         ['2019-12-20T01:26:14.803', '2019-12-20T01:26:14.803']],
        dtype='datetime64[ms]')

    handler = PeriodList(periods.T)

    assert handler.is_it_close(numpy.datetime64('2019-12-10T11:02:53.581'))
    assert handler.is_it_close(numpy.datetime64('2019-12-10T11:02:53'),
                               numpy.timedelta64(1, 's'))
    assert handler.is_it_close(numpy.datetime64('2019-12-10T11:02:50'),
                               numpy.timedelta64(4, 's'))

    assert handler.are_periods_sorted_and_disjointed()
    periods[2, :], periods[3, :] = periods[3, :], periods[2, :].copy()
    handler = PeriodList(periods.T)
    assert not handler.are_periods_sorted_and_disjointed()
    assert handler.sort().are_periods_sorted_and_disjointed()

    period = Period(numpy.datetime64('2019-12-13T01:21:28.255', 'ms'),
                    numpy.datetime64('2019-12-15T00:48:21.092', 'ms'))

    merged = handler.intersection(period)
    assert len(merged) == 4
    within = merged.within(period)
    assert len(within) == 4
    assert numpy.all(merged.periods == within.periods)

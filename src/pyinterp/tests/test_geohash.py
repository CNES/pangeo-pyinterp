import pickle

import numpy
import pytest
import xarray

import pyinterp

from .. import GeoHash

testcases = [['77mkh2hcj7mz', -26.015434642, -26.173663656],
             ['wthnssq3w00x', 29.291182895, 118.331595326],
             ['z3jsmt1sde4r', 51.400326027, 154.228244707],
             ['18ecpnqdg4s1', -86.976900779, -106.90988479],
             ['u90suzhjqv2s', 51.49934315, 23.417648894],
             ['k940p3ewmmyq', -39.365655496, 25.636144008],
             ['6g4wv2sze6ms', -26.934429639, -52.496991862],
             ['jhfyx4dqnczq', -62.123898484, 49.178194037],
             ['j80g4mkqz3z9', -89.442648795, 68.659722351],
             ['hq9z7cjwrcw4', -52.156511416, 13.883626414]]


def test_geohash():
    for code, lat, lon in testcases:
        instance = GeoHash(lon, lat, precision=12)
        assert str(instance) == code
        point = instance.center()
        assert lat == pytest.approx(point.lat)
        assert lon == pytest.approx(point.lon)

        other = GeoHash.from_string(code)
        assert str(other) == code

        other = pickle.loads(pickle.dumps(instance))
        assert str(other) == code

        assert instance.number_of_bits() == 60
        assert instance.precision() == 12

    instance = GeoHash.from_string('e')
    neighbors = instance.neighbors()
    assert len(neighbors) == 8
    assert [str(item)
            for item in neighbors] == ['g', 'u', 's', 'k', '7', '6', 'd', 'f']
    assert repr(instance) == 'GeoHash(-22.5, 22.5, 1)'


def test_geohash_grid():
    grid = GeoHash.grid()
    assert isinstance(grid, xarray.Dataset)
    assert grid.dims['lon'] == 8
    assert grid.dims['lat'] == 4
    assert grid.geohash.shape == (4, 8)
    assert grid.geohash.dtype == 'S1'


def test_geohash_converter():
    codes = numpy.concatenate([
        pyinterp.geohash.bounding_boxes(precision=2),
        pyinterp.geohash.bounding_boxes(precision=2)
    ])
    data = numpy.ones(codes.shape, dtype=numpy.float32)

    ds = pyinterp.geohash.to_xarray(codes, data)
    assert isinstance(ds, xarray.DataArray)

    with pytest.raises(ValueError):
        pyinterp.geohash.to_xarray(codes, data[::5])

    with pytest.raises(TypeError):
        pyinterp.geohash.to_xarray(data, data)

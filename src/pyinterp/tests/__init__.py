# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
from typing import Optional
import json
import pathlib

import numpy
import pytest
import xarray

ROOT = pathlib.Path(__file__).parent.joinpath('dataset').resolve()


def make_or_compare_reference(filename: str, values: numpy.ndarray,
                              dump: bool) -> None:
    """Make a reference file or compare against it.

    Args:
        filename (str): The filename to create or compare against.
        values (np.ndarray): The values to compare against.
        dump (bool): Whether to dump the values to the file.
    """
    path = ROOT.joinpath(filename)
    if dump:
        numpy.save(path, values)
        return
    if path.exists():
        reference = numpy.load(path)
        assert numpy.allclose(reference, values, equal_nan=True)


def _mask_and_scale(array: xarray.DataArray) -> numpy.ndarray:
    """Mask and scale data."""
    add_offset = array.attrs.pop('add_offset', 0)
    scale_factor = array.attrs.pop('scale_factor', 1)
    fill_value = array.attrs.pop('_FillValue', None)

    result = array.data.astype('float64')
    result = result * scale_factor + add_offset
    if fill_value is not None:
        result[array.data == fill_value] = numpy.nan
    return result


def load_grid2d() -> xarray.Dataset:
    """Load the grid2d dataset."""
    path = ROOT.joinpath('mss.json')
    with path.open('r') as stream:
        data = json.load(stream)
    ds = xarray.Dataset.from_dict(data)
    ds.variables['mss'].values = _mask_and_scale(ds['mss'])
    return ds


def load_aoml():
    """Return path to the AOML dataset."""

    def _decode_datetime64(array: numpy.ndarray) -> numpy.ndarray:
        """Decode datetime64 data."""
        array = array.astype('timedelta64[h]') + numpy.datetime64(
            '2001-12-19T18:00:00')
        return array

    path = ROOT.joinpath('aoml_v2019.json')
    with path.open('r') as stream:
        data = json.load(stream)
    for item in ('ud', 'vd'):
        data['data_vars'][item]['data'] = list(
            map(lambda x: x if x is not None else float('nan'),
                data['data_vars'][item]['data']))
    ds = xarray.Dataset.from_dict(data)
    ds['time'] = xarray.DataArray(_decode_datetime64(ds['time'].values),
                                  dims=['time'],
                                  attrs={'long_name': 'time'})
    return ds


def positions_path():
    """Return path to the ARGO positions."""
    return ROOT.joinpath('positions.csv')


def load_grid3d() -> xarray.Dataset:
    """Return path to the Grid 3D."""

    def _decode_datetime64(array: numpy.ndarray) -> numpy.ndarray:
        """Decode datetime64 data."""
        array = array.astype('timedelta64[h]') + numpy.datetime64('1900-01-01')
        return array

    path = ROOT.joinpath('tcw.json')
    with path.open('r') as stream:
        data = json.load(stream)
    ds = xarray.Dataset.from_dict(data)
    ds.variables['tcw'].values = _mask_and_scale(ds['tcw'])
    ds['time'] = xarray.DataArray(_decode_datetime64(ds['time'].values),
                                  dims=['time'],
                                  attrs={'long_name': 'time'})
    return ds


def load_grid4d():
    """Return path to the Grid 4D."""

    def _decode_datetime64(array: numpy.ndarray) -> numpy.ndarray:
        """Decode datetime64 data."""
        array = array.astype('timedelta64[s]') + numpy.datetime64('1970-01-01')
        return array

    path = ROOT.joinpath('pres_temp_4d.json')

    with path.open('r') as stream:
        data = json.load(stream)
    ds = xarray.Dataset.from_dict(data)
    ds['time'] = xarray.DataArray(_decode_datetime64(ds['time'].values),
                                  dims=['time'],
                                  attrs={'long_name': 'time'})
    ds = ds.assign_coords({
        'longitude': ds['longitude'].astype('float32'),
        'latitude': ds['latitude'].astype('float32')
    })

    ds['pressure'] = ds['pressure'].astype('float32')
    ds['temperature'] = ds['temperature'].astype('float32')
    return ds


def geohash_bbox_path():
    """Return path to the GeoHash bounding box."""
    return ROOT.joinpath('geohash_bbox.json')


def geohash_neighbors_path():
    """Return path to the GeoHash neighbors."""
    return ROOT.joinpath('geohash_neighbors.json')


def geohash_path():
    """Return path to the GeoHash dataset."""
    return ROOT.joinpath('geohash.json')


def polygon_path():
    """Return path to the polygon dataset."""
    return ROOT.joinpath('polygon.json')


def multipolygon_path():
    """Return path to the polygon dataset."""
    return ROOT.joinpath('multipolygon.json')


def swot_calval_ephemeris_path():
    """Return path to the SWOT Calval ephemeris."""
    return ROOT.joinpath('ephemeris_calval_sept2015.txt')


def run(pattern: Optional[str] = None) -> None:
    """Run tests.

    Args:
        pattern (str, optional): A regex pattern to match against test names.
    """
    args = ['-x', str(pathlib.Path(__file__).parent.resolve())]
    if pattern is not None:
        args += ['-k', pattern]
    pytest.main(args)

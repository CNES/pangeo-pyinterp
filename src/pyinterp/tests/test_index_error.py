# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import collections
import datetime

import numpy as np
import pytest

from . import load_grid3d, load_grid4d
from ..backends import xarray as xr_backend


def test_index_error_3d():
    ds = load_grid3d()
    grid = xr_backend.RegularGridInterpolator(ds.tcw, increasing_axes=True)
    lon = np.arange(-180, 180, 10)
    lat = np.arange(-80, 80, 10)
    time = np.array([datetime.datetime(2020, 7, 2, 15, 0)],
                    dtype='datetime64[ns]')
    x, y, t = np.meshgrid(lon, lat, time, indexing='ij')

    with pytest.raises(ValueError):
        grid(collections.OrderedDict(longitude=x.ravel(),
                                     latitude=y.ravel(),
                                     time=t.ravel()),
             bounds_error=True)

    with pytest.raises(ValueError, match='2020-07-02T15:00:00.000000000'):
        grid(collections.OrderedDict(longitude=x.ravel(),
                                     latitude=y.ravel(),
                                     time=t.ravel()),
             method='bicubic',
             bounds_error=True)

    ds = ds.assign_coords(time=ds.time.astype('int64'))
    grid = xr_backend.RegularGridInterpolator(ds.tcw, increasing_axes=True)

    with pytest.raises(ValueError, match='1593702000000000000.000000'):
        grid(collections.OrderedDict(longitude=x.ravel(),
                                     latitude=y.ravel(),
                                     time=t.astype('int64').ravel()),
             bounds_error=True)


def test_index_error_4d():
    ds = load_grid4d()
    grid = xr_backend.Grid4D(ds.pressure, increasing_axes=True)

    lon = np.arange(-120, -75, 0.25)
    lat = np.arange(34, 40, 0.25)
    level = 0.5
    time = np.datetime64('2010-01-01T12:00')
    x, y, z, t = np.meshgrid(lon, lat, level, time, indexing='ij')

    with pytest.raises(ValueError, match='2010-01-01T12:00:00.000000000'):
        grid.quadrivariate(collections.OrderedDict(longitude=x.ravel(),
                                                   latitude=y.ravel(),
                                                   level=z.ravel(),
                                                   time=t.ravel()),
                           bounds_error=True)

    ds = ds.assign_coords(time=ds.time.astype('int64'))
    grid = xr_backend.RegularGridInterpolator(ds.pressure,
                                              increasing_axes=True)
    with pytest.raises(ValueError, match='21039120'):
        grid(collections.OrderedDict(longitude=x.ravel(),
                                     latitude=y.ravel(),
                                     level=z.ravel(),
                                     time=t.astype('int64').ravel()),
             bounds_error=True)

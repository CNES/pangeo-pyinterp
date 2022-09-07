# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import numpy as np
import pytest

from .. import Axis, Grid2D, Grid3D, Grid4D, TemporalAxis, core, grid, interface


def test_core_class_suffix():
    lon = Axis(np.arange(0, 360, 1), is_circle=True)
    lat = Axis(np.arange(-80, 80, 1), is_circle=False)
    for dtype in [
            'float64', 'float32', 'int64', 'uint64', 'int32', 'uint32',
            'int16', 'uint16', 'int8', 'uint8'
    ]:
        matrix, _ = np.meshgrid(lon[:], lat[:])
        assert isinstance(
            Grid2D(lon, lat, matrix.T.astype(dtype=getattr(np, dtype))),
            Grid2D)

    with pytest.raises(ValueError):
        Grid2D(lon, lat, matrix.astype(complex))


def test_core_grid2d():
    x = Axis(np.arange(0, 360, 10), is_circle=True)
    y = Axis(np.arange(0, 360, 10), is_circle=False)

    tensor, _, = np.meshgrid(x[:], y[:])
    assert isinstance(Grid2D(x, y, tensor), Grid2D)

    with pytest.raises(ValueError):
        Grid2D(y, x, tensor)


def test_core_grid3d():
    x = Axis(np.arange(0, 360, 10), is_circle=True)
    y = Axis(np.arange(0, 360, 10), is_circle=False)
    z = Axis(np.arange(0, 360, 10), is_circle=False)

    tensor, _, _ = np.meshgrid(x[:], y[:], z[:])
    assert isinstance(Grid3D(x, y, z, tensor), Grid3D)

    with pytest.raises(ValueError):
        Grid3D(y, x, z, tensor)

    with pytest.raises(ValueError):
        Grid3D(y, z, x, tensor)


def test_core_grid4d():
    x = Axis(np.arange(0, 360, 10), is_circle=True)
    y = Axis(np.arange(0, 360, 10), is_circle=False)
    z = Axis(np.arange(0, 360, 10), is_circle=False)
    u = Axis(np.arange(0, 360, 10), is_circle=False)

    tensor, _, _, _ = np.meshgrid(x[:], y[:], z[:], u[:])
    assert isinstance(Grid4D(x, y, z, u, tensor), Grid4D)

    with pytest.raises(ValueError):
        Grid4D(y, x, z, u, tensor),

    with pytest.raises(ValueError):
        Grid4D(y, z, x, u, tensor),

    with pytest.raises(ValueError):
        Grid4D(y, z, u, x, tensor),


def test__core_function_suffix():
    with pytest.raises(TypeError):
        interface._core_function(1)  # type: ignore

    with pytest.raises(TypeError):
        interface._core_function('foo', str(1))

    lon = Axis(np.arange(0, 360, 1), is_circle=True)
    lat = Axis(np.arange(-80, 80, 1), is_circle=False)
    matrix, _ = np.meshgrid(lon[:], lat[:])
    assert interface._core_function('foo',
                                    core.Grid2DFloat64(
                                        lon, lat, matrix.T)) == 'foo_float64'
    assert interface._core_function('foo',
                                    core.Grid2DFloat32(
                                        lon, lat, matrix.T)) == 'foo_float32'

    time = TemporalAxis(np.array(['2000-01-01'], dtype='datetime64'))
    matrix, _, _ = np.meshgrid(lon[:], lat[:], time[:], indexing='ij')
    assert interface._core_function(
        'foo', core.TemporalGrid3DFloat64(lon, lat, time,
                                          matrix)) == 'foo_float64'


def test_core_variate_interpolator():
    lon = Axis(np.arange(0, 360, 1), is_circle=True)
    lat = Axis(np.arange(-80, 80, 1), is_circle=False)
    matrix, _ = np.meshgrid(lon[:], lat[:])

    instance = grid.Grid2D(lon, lat, matrix.T)

    with pytest.raises(TypeError):
        grid._core_variate_interpolator(None, '_')  # type: ignore

    with pytest.raises(ValueError):
        grid._core_variate_interpolator(instance, '_')  # type: ignore

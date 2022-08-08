# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pickle

import numpy as np
import pytest

from ... import core


def f4d(x, y, z, u):
    return u * np.exp(-x**2 - y**2 - z**2)


def load_data():
    x = np.arange(-1, 1, 0.2)
    y = np.arange(-1, 1, 0.2)
    z = np.arange(-1, 1, 0.2)
    u = np.arange(-1, 10, 0.2)

    mx, my, mz, mu = np.meshgrid(x, y, z, u)
    return core.Grid4DFloat64(core.Axis(x), core.Axis(y), core.Axis(z),
                              core.Axis(u), f4d(mx, my, mz, mu))


def test_grid4d_accessors():
    """Test construction and accessors of the object."""
    grid = load_data()
    assert isinstance(grid.x, core.Axis)
    assert isinstance(grid.y, core.Axis)
    assert isinstance(grid.z, core.Axis)
    assert isinstance(grid.u, core.Axis)
    assert isinstance(grid.array, np.ndarray)


def test_grid4d_pickle():
    """Serialization test."""
    grid = load_data()
    other = pickle.loads(pickle.dumps(grid))
    assert grid.x == other.x
    assert grid.y == other.y
    assert grid.z == other.z
    assert grid.u == other.u
    assert np.all(grid.array == other.array)


def test_interpolator():
    grid = load_data()

    x = np.arange(-1, 1, 0.2)
    y = np.arange(-1, 1, 0.2)
    z = np.arange(-1, 1, 0.2)
    u = np.arange(-1, 10, 0.2)

    mx, my, mz, mu = np.meshgrid(x, y, z, u)
    expected = f4d(mx, my, mz, mu)

    interpolator = core.Bilinear3D()

    calculated = core.quadrivariate_float64(grid,
                                            mx.ravel(),
                                            my.ravel(),
                                            mz.ravel(),
                                            mu.ravel(),
                                            interpolator,
                                            num_threads=0,
                                            bounds_error=True)
    assert np.all(expected.ravel() == calculated)

    x = np.arange(-1, 1, 0.2)
    y = np.arange(-1, 1, 0.2)
    z = np.arange(-1, 1, 0.2)
    u = np.arange(-1, 10, 0.33333)

    mx, my, mz, mu = np.meshgrid(x, y, z, u)
    expected = f4d(mx, my, mz, mu)

    interpolator = core.Bilinear3D()

    calculated = core.quadrivariate_float64(grid,
                                            mx.ravel(),
                                            my.ravel(),
                                            mz.ravel(),
                                            mu.ravel(),
                                            interpolator,
                                            num_threads=0,
                                            bounds_error=False)
    assert np.nanstd(expected.ravel() - calculated) == pytest.approx(0)

    other = core.quadrivariate_float64(grid,
                                       mx.ravel(),
                                       my.ravel(),
                                       mz.ravel(),
                                       mu.ravel(),
                                       interpolator,
                                       num_threads=0,
                                       z_method='linear',
                                       u_method='linear',
                                       bounds_error=False)
    assert np.nanstd(other - calculated) == pytest.approx(0)

    other = core.quadrivariate_float64(grid,
                                       mx.ravel(),
                                       my.ravel(),
                                       mz.ravel(),
                                       mu.ravel(),
                                       interpolator,
                                       num_threads=0,
                                       z_method='linear',
                                       u_method='nearest',
                                       bounds_error=False)
    assert np.nanstd(other - calculated) == pytest.approx(0, abs=1e-1)

    with pytest.raises(ValueError):
        other = core.quadrivariate_float64(  # type: ignore
            grid,
            mx.ravel(),
            my.ravel(),
            mz.ravel(),
            None,
            interpolator,
            num_threads=0,
            z_method='linear',
            u_method='nearest',
            bounds_error=False)

    with pytest.raises(ValueError):
        other = core.quadrivariate_float64(grid,
                                           mx.ravel(),
                                           my.ravel(),
                                           mz.ravel(),
                                           mu.ravel(),
                                           interpolator,
                                           num_threads=0,
                                           z_method='LINEAR',
                                           u_method='nearest',
                                           bounds_error=False)

    with pytest.raises(ValueError):
        other = core.quadrivariate_float64(grid,
                                           mx.ravel(),
                                           my.ravel(),
                                           mz.ravel(),
                                           mu.ravel(),
                                           interpolator,
                                           num_threads=0,
                                           z_method='linear',
                                           u_method='NEAREST',
                                           bounds_error=False)

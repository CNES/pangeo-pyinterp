# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import pickle

import numpy as np
import pytest

import pyinterp
import pyinterp.backends.xarray

from . import load_grid2d, make_or_compare_reference


def build_rtree(dtype):
    lon = np.arange(-180, 180, 10, dtype=dtype)
    lat = np.arange(-90, 90, 10, dtype=dtype)
    lon, lat = np.meshgrid(lon, lat)
    data = lon * 0
    mesh = pyinterp.RTree(dtype=dtype)
    assert isinstance(mesh, pyinterp.RTree)
    assert len(mesh) == 0
    assert not bool(mesh)
    mesh.packing(np.vstack((lon.ravel(), lat.ravel())).T, data.ravel())
    assert len(mesh) == len(lon.ravel())
    assert bool(mesh)
    (x_min, y_min, z_min), (x_max, y_max, z_max) = mesh.bounds()
    assert x_min == -180
    assert y_min == -90.0
    assert x_max == 180.0
    assert y_max == 80
    assert 0 == pytest.approx(z_min, abs=1e-6 if dtype == np.float64 else 0.5)
    assert 0 == pytest.approx(z_max, abs=1e-6 if dtype == np.float64 else 0.5)
    mesh.clear()
    assert len(mesh) == 0
    assert not bool(mesh)
    mesh.insert(np.vstack((lon.ravel(), lat.ravel())).T, data.ravel())
    assert len(mesh) == len(lon.ravel())
    assert isinstance(pickle.loads(pickle.dumps(mesh)), pyinterp.RTree)


def test_init():
    build_rtree(dtype=np.float32)
    build_rtree(dtype=np.float64)

    with pytest.raises(ValueError):
        build_rtree(np.int8)

    with pytest.raises(ValueError):
        mesh = pyinterp.RTree()
        mesh.__setstate__((1, ))

    with pytest.raises(ValueError):
        pyinterp.RTree(ndims=1)


def load_data():
    ds = load_grid2d()
    z = ds.mss.T
    x, y = np.meshgrid(ds.lon.values, ds.lat.values, indexing='ij')
    mesh = pyinterp.RTree()
    mesh.packing(np.vstack((x.ravel(), y.ravel())).T, z.values.ravel())
    return mesh


def test_interpolate(pytestconfig):
    dump = pytestconfig.getoption('dump')
    mesh = load_data()
    lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
    lat = np.arange(-90, 90, 1 / 3.0) + 1 / 3.0
    x, y = np.meshgrid(lon, lat, indexing='ij')
    coordinates = np.vstack((x.ravel(), y.ravel())).T
    points, values = mesh.value(coordinates)
    assert points.shape == (coordinates.shape[0], 4, 2)
    assert values.shape == (coordinates.shape[0], 4)
    data, _ = mesh.query(coordinates)
    make_or_compare_reference('rtree_query.npy', data, dump)
    data, _ = mesh.inverse_distance_weighting(coordinates)
    make_or_compare_reference('rtree_idw.npy', data, dump)
    data, _ = mesh.radial_basis_function(coordinates)
    make_or_compare_reference('rtree_rbf.npy', data, dump)
    data, _ = mesh.window_function(coordinates, radius=2_000_000)
    make_or_compare_reference('rtree_wf.npy', data, dump)

    with pytest.raises(ValueError):
        mesh.radial_basis_function(coordinates, epsilon=1, rbf='cubic')
    with pytest.raises(ValueError):
        mesh.radial_basis_function(coordinates, rbf='X')
    with pytest.raises(ValueError):
        mesh.window_function(coordinates, radius=1, wf='cubic')
    with pytest.raises(ValueError):
        mesh.window_function(coordinates, radius=1, wf='parzen', arg=-1)
    with pytest.raises(ValueError):
        mesh.window_function(coordinates, radius=1, wf='lanczos', arg=0)
    with pytest.raises(ValueError):
        mesh.window_function(coordinates, radius=1, wf='blackman', arg=2)

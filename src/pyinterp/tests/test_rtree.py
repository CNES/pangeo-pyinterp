# Copyright (c) 2021 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import os
import pytest
import pickle
import numpy as np
import xarray as xr
import pyinterp.backends.xarray
import pyinterp
from . import grid2d_path


def build_rtree(dtype):
    lon = np.arange(-180, 180, 10, dtype=dtype)
    lat = np.arange(-90, 90, 10, dtype=dtype)
    lon, lat = np.meshgrid(lon, lat)
    data = lon * 0
    mesh = pyinterp.RTree(dtype=dtype)
    assert isinstance(mesh, pyinterp.RTree)
    assert len(mesh) == 0
    assert not bool(mesh)
    mesh.packing(np.vstack((lon.flatten(), lat.flatten())).T, data.flatten())
    assert len(mesh) == len(lon.flatten())
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
    mesh.insert(np.vstack((lon.flatten(), lat.flatten())).T, data.flatten())
    assert len(mesh) == len(lon.flatten())
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
    ds = xr.load_dataset(grid2d_path())
    z = ds.mss.T
    x, y = np.meshgrid(ds.lon.values, ds.lat.values, indexing='ij')
    mesh = pyinterp.RTree()
    mesh.packing(np.vstack((x.flatten(), y.flatten())).T, z.values.flatten())
    return mesh


def test_interpolate():
    mesh = load_data()
    lon = np.arange(-180, 180, 1 / 3.0) + 1 / 3.0
    lat = np.arange(-90, 90, 1 / 3.0) + 1 / 3.0
    x, y = np.meshgrid(lon, lat, indexing="ij")
    coordinates = np.vstack((x.flatten(), y.flatten())).T
    mesh.query(coordinates)
    mesh.inverse_distance_weighting(coordinates)
    mesh.radial_basis_function(coordinates)

    with pytest.raises(ValueError):
        mesh.radial_basis_function(coordinates, epsilon=1, rbf="cubic")
    with pytest.raises(ValueError):
        mesh.radial_basis_function(coordinates, rbf="X")

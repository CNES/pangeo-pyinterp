# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for the RTree class."""
import pickle

import numpy as np
import pytest
from pytest import Config

import pyinterp

from . import load_grid2d, make_or_compare_reference


def build_rtree(dtype: np.dtype) -> None:
    """Build an RTree instance with a given dtype."""
    lon = np.arange(-180, 180, 10).astype(dtype)
    lat = np.arange(-90, 90, 10).astype(dtype)
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
    tolerance = 1e-12 if dtype == np.float64 else 1e-4
    assert -180 == pytest.approx(x_min, abs=tolerance)
    assert -90.0 == pytest.approx(y_min, abs=tolerance)
    assert 180.0 == pytest.approx(x_max, abs=tolerance)
    assert 80 == pytest.approx(y_max, abs=tolerance)
    tolerance = 1e-6 if dtype == np.float64 else 0.5
    assert 0 == pytest.approx(z_min, abs=tolerance)
    assert 0 == pytest.approx(z_max, abs=tolerance)
    mesh.clear()
    assert len(mesh) == 0
    assert not bool(mesh)
    mesh.insert(np.vstack((lon.ravel(), lat.ravel())).T, data.ravel())
    assert len(mesh) == len(lon.ravel())
    assert isinstance(pickle.loads(pickle.dumps(mesh)), pyinterp.RTree)


def test_init() -> None:
    """Test RTree initialization."""
    build_rtree(dtype=np.dtype(np.float32))
    build_rtree(dtype=np.dtype(np.float64))

    with pytest.raises(ValueError):
        build_rtree(np.dtype(np.int8))

    with pytest.raises(ValueError):
        mesh = pyinterp.RTree()
        mesh.__setstate__((1, ))


def load_data() -> pyinterp.RTree:
    """Load test data into an RTree instance."""
    ds = load_grid2d()
    z = ds.mss.T
    x, y = np.meshgrid(ds.lon.values, ds.lat.values, indexing='ij')
    mesh = pyinterp.RTree()
    mesh.packing(np.vstack((x.ravel(), y.ravel())).T, z.values.ravel())
    return mesh


def test_interpolate(pytestconfig: Config) -> None:
    """Test RTree interpolation methods."""
    dump = pytestconfig.getoption('dump')
    measure_coverage = pytestconfig.getoption('measure_coverage')
    step = 10 if measure_coverage else 1 / 3
    mesh = load_data()
    lon = np.arange(-180, 180, step) + 1 / 3
    lat = np.arange(-90, 90, step) + 1 / 3
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
    data, _ = mesh.kriging(coordinates)
    make_or_compare_reference('rtree_uk.npy', data, dump)

    # Test universal kriging with linear drift and nugget & custom alpha
    data_linear, _ = mesh.kriging(coordinates,
                                  covariance='matern_32',
                                  drift_function='linear',
                                  sigma=1.5,
                                  alpha=500_000.0,
                                  nugget=1e-6)
    assert data_linear.shape == data.shape
    # Ensure different result from simple kriging for at least some points
    assert np.any(np.abs(data_linear - data) > 0)

    # Test quadratic drift
    data_quadratic, _ = mesh.kriging(coordinates,
                                     covariance='gaussian',
                                     drift_function='quadratic',
                                     sigma=0.75,
                                     alpha=250_000.0,
                                     nugget=0.0)
    assert data_quadratic.shape == data.shape
    assert np.any(np.abs(data_quadratic - data_linear) > 0)

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
    with pytest.raises(ValueError):
        mesh.kriging(coordinates, radius=1, covariance='blackman')

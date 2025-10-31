# Copyright (c) 2025 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for geodetic RTree interpolation."""
from __future__ import annotations

from typing import TYPE_CHECKING
import os
import pickle

try:
    import matplotlib.colors
    import matplotlib.pyplot
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False
from _pytest.config import Config
import numpy as np

from .. import load_grid2d
from ... import core
from ...core import geodetic

if TYPE_CHECKING:
    from pytest import Config

    from ...typing import NDArray1D, NDArray2D


def plot(x: NDArray1D, y: NDArray1D, z: NDArray2D, filename: str) -> None:
    """Plot the interpolated data."""
    figure = matplotlib.pyplot.figure(figsize=(15, 15), dpi=150)
    value = z.mean()
    std = z.std()
    normalize = matplotlib.colors.Normalize(vmin=value - 3 * std,
                                            vmax=value + 3 * std)
    axe = figure.add_subplot(2, 1, 1)
    axe.pcolormesh(x, y, z, cmap='jet', norm=normalize, shading='auto')
    figure.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                filename),
                   bbox_inches='tight',
                   pad_inches=0.4)


def load_data(packing: bool = True) -> geodetic.RTree:
    """Load test data."""
    ds = load_grid2d()
    z = ds['mss'].values.T
    x = ds['lon'].values
    y = ds['lat'].values
    x = x[::10]
    y = y[::10]
    z = z[::10, ::10]
    x, y = np.meshgrid(x, y, indexing='ij')
    mesh = geodetic.RTree()
    if packing:
        mesh.packing(x.ravel(), y.ravel(), z.ravel())
    else:
        mesh.insert(x.ravel(), y.ravel(), z.ravel())
    return mesh


def test_geodetic_rtree_idw(pytestconfig: Config) -> None:
    """Interpolation test."""
    measure_coverage = pytestconfig.getoption('measure_coverage')
    step = 10 if measure_coverage else 1
    mesh = load_data()
    lon = np.arange(-180, 180, step)
    lat = np.arange(-90, 90, step)
    x, y = np.meshgrid(lon, lat, indexing='ij')
    z, _ = mesh.inverse_distance_weighting(x.ravel(),
                                           y.ravel(),
                                           within=False,
                                           radius=None,
                                           k=8,
                                           num_threads=0)
    z = np.ma.fix_invalid(z)

    if HAVE_PLT and pytestconfig.getoption('visualize'):
        plot(x, y, z.reshape((len(lon), len(lat))),
             'mss_geodetic_rtree_idw.png')


def test_geodetic_rtree_rbf(pytestconfig: Config) -> None:
    """Interpolation test."""
    measure_coverage = pytestconfig.getoption('measure_coverage')
    step = 10 if measure_coverage else 1
    mesh = load_data()
    lon = np.arange(-180, 180, step)
    lat = np.arange(-90, 90, step)
    x, y = np.meshgrid(lon, lat, indexing='ij')
    z, _ = mesh.radial_basis_function(x.ravel(),
                                      y.ravel(),
                                      within=True,
                                      radius=2_000_000,
                                      rbf=core.RadialBasisFunction.ThinPlate,
                                      epsilon=None,
                                      smooth=0,
                                      k=11,
                                      num_threads=0)
    z = np.ma.fix_invalid(z)

    if HAVE_PLT and pytestconfig.getoption('visualize'):
        plot(x, y, z.reshape((len(lon), len(lat))),
             'mss_geodetic_rtree_rbf.png')


def test_geodetic_rtree_window_function(pytestconfig: Config) -> None:
    """Interpolation test."""
    measure_coverage = pytestconfig.getoption('measure_coverage')
    step = 10 if measure_coverage else 1
    mesh = load_data()
    lon = np.arange(-180, 180, step)
    lat = np.arange(-90, 90, step)
    x, y = np.meshgrid(lon, lat, indexing='ij')
    z0, _ = mesh.window_function(x.ravel(),
                                 y.ravel(),
                                 within=False,
                                 radius=2_000_000,
                                 wf=core.WindowFunction.Hamming,
                                 k=11,
                                 num_threads=0)
    z0 = np.ma.fix_invalid(z0)

    if HAVE_PLT and pytestconfig.getoption('visualize'):
        plot(x, y, z0.reshape((len(lon), len(lat))),
             'mss_geodetic_rtree_wf.png')


def test_geodetic_rtree_insert() -> None:
    """Data insertion test."""
    mesh = load_data(packing=False)
    assert isinstance(mesh, geodetic.RTree)
    assert len(mesh) != 0


def test_geodetic_rtree_query() -> None:
    """Data insertion test."""
    mesh = load_data(packing=True)
    assert len(mesh) != 0
    distances, values = mesh.query(np.array([0.0]), np.array([0.0]))
    assert distances.shape == (1, 4)
    assert values.shape == (1, 4)


def test_geodetic_rtree_pickle() -> None:
    """Serialization test."""
    interpolator = load_data()
    other = pickle.loads(pickle.dumps(interpolator))
    assert isinstance(other, geodetic.RTree)

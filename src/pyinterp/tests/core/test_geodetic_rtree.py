# Copyright (c) 2022 CNES
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
import os
import pickle

try:
    import matplotlib.colors
    import matplotlib.pyplot
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False
import numpy as np

from .. import load_grid2d
from ... import core
from ...core import geodetic


def plot(x, y, z, filename):
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


def load_data(packing=True):
    """Creating the search tree."""
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


def test_geodetic_rtree_idw(pytestconfig):
    """Interpolation test."""
    mesh = load_data()
    lon = np.arange(-180, 180, 1)
    lat = np.arange(-90, 90, 1)
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


def test_geodetic_rtree_rbf(pytestconfig):
    """Interpolation test."""
    mesh = load_data()
    lon = np.arange(-180, 180, 1)
    lat = np.arange(-90, 90, 1)
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


def test_geodetic_rtree_window_function(pytestconfig):
    """Interpolation test."""
    mesh = load_data()
    lon = np.arange(-180, 180, 1)
    lat = np.arange(-90, 90, 1)
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


def test_geodetic_rtree_insert():
    """Data insertion test."""
    mesh = load_data(packing=False)
    assert isinstance(mesh, geodetic.RTree)
    assert len(mesh) != 0


def test_geodetic_rtree_query():
    """Data insertion test."""
    mesh = load_data(packing=True)
    assert len(mesh) != 0
    distances, values = mesh.query(np.array([0.0]), np.array([0.0]))
    assert distances.shape == (1, 4)
    assert values.shape == (1, 4)


def test_geodetic_rtree_pickle():
    """Serialization test."""
    interpolator = load_data()
    other = pickle.loads(pickle.dumps(interpolator))
    assert isinstance(other, geodetic.RTree)

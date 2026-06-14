# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for the OptimalInterpolation module.

These tests validate the OI estimator against a pure-numpy reference
implementation, exercise the seven covariance kernels, check the
scalar/Grid2D polymorphism, and verify limit behaviours
(zero / infinite measurement noise, infinite time scale).
"""

from __future__ import annotations

import numpy as np
import pytest

import pyinterp
from pyinterp.optimal_interpolation import _kernel_from_r2


_KERNELS = (
    "gaussian",
    "cauchy",
    "matern_12",
    "matern_32",
    "matern_52",
    "spherical",
    "wendland",
)


def _reference_oi(
    obs_coords: np.ndarray,
    obs_values: np.ndarray,
    obs_sigma2: np.ndarray,
    query: np.ndarray,
    lx: float,
    ly: float,
    lt: float,
    sigma: float,
    kernel: str,
) -> tuple[float, float]:
    """Direct numpy OI on a single query point (no R-tree)."""
    dx_oo = (obs_coords[:, 0, None] - obs_coords[None, :, 0]) / lx
    dy_oo = (obs_coords[:, 1, None] - obs_coords[None, :, 1]) / ly
    dt_oo = (obs_coords[:, 2, None] - obs_coords[None, :, 2]) / lt
    r2_oo = dx_oo**2 + dy_oo**2 + dt_oo**2

    qx, qy, qt = query
    dx_og = (qx - obs_coords[:, 0]) / lx
    dy_og = (qy - obs_coords[:, 1]) / ly
    dt_og = (qt - obs_coords[:, 2]) / lt
    r2_og = dx_og**2 + dy_og**2 + dt_og**2

    sig2 = sigma * sigma
    c_oo = sig2 * _kernel_from_r2(r2_oo, kernel)
    c_og = sig2 * _kernel_from_r2(r2_og, kernel)
    n = obs_coords.shape[0]
    c_oo[np.arange(n), np.arange(n)] += obs_sigma2

    w = np.linalg.solve(c_oo, c_og)
    value = float(w @ obs_values)
    err2 = float(sig2 - c_og @ w)
    return value, np.sqrt(max(err2, 0.0))


def _make_dataset(
    n: int = 80, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coords = rng.uniform([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], size=(n, 3))
    values = (
        np.sin(coords[:, 0])
        * np.cos(coords[:, 1])
        + 0.2 * coords[:, 2]
    )
    # Slight noise that varies across observations (mimics multi-mission σ²).
    sigma2 = 0.01 + 0.005 * rng.random(n)
    return coords, values, sigma2


@pytest.mark.parametrize("kernel", _KERNELS)
def test_matches_reference(kernel: str) -> None:
    """For each kernel, OI matches the numpy reference within tolerance."""
    coords, values, sigma2 = _make_dataset(n=40)
    oi = pyinterp.OptimalInterpolation(
        coords, values, sigma2, covariance=kernel
    )
    # Query points inside the domain
    rng = np.random.default_rng(123)
    q = rng.uniform([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], size=(8, 3))

    # Using k=N forces the OI to use *all* observations, eliminating any
    # k-NN approximation and isolating the OI-math comparison.
    result = oi(q, lx=2.0, ly=2.0, lt=2.0, sigma=1.0, k=coords.shape[0])

    for i in range(q.shape[0]):
        ref_v, ref_e = _reference_oi(
            coords, values, sigma2, q[i], 2.0, 2.0, 2.0, 1.0, kernel
        )
        np.testing.assert_allclose(
            result.value[i], ref_v, rtol=1e-9, atol=1e-12
        )
        np.testing.assert_allclose(
            result.error[i], ref_e, rtol=1e-7, atol=1e-10
        )


def test_zero_noise_recovers_observations() -> None:
    """With σ²_obs → 0, OI at an observation location returns its value."""
    coords, values, _ = _make_dataset(n=30)
    tiny = np.full(coords.shape[0], 1e-12)
    oi = pyinterp.OptimalInterpolation(coords, values, tiny)

    # Query at the obs locations themselves.
    result = oi(
        coords, lx=1.0, ly=1.0, lt=1.0, sigma=1.0, k=coords.shape[0]
    )
    np.testing.assert_allclose(result.value, values, atol=1e-6)
    # Formal error should be close to zero at the observation locations.
    np.testing.assert_allclose(result.error, 0.0, atol=1e-5)


def test_infinite_noise_returns_zero() -> None:
    """With σ²_obs → ∞, the analysis collapses to the prior mean (zero)."""
    coords, values, _ = _make_dataset(n=20)
    huge = np.full(coords.shape[0], 1e12)
    oi = pyinterp.OptimalInterpolation(coords, values, huge)

    q = np.array([[5.0, 5.0, 5.0]])
    result = oi(q, lx=1.0, ly=1.0, lt=1.0, sigma=1.0, k=20)
    np.testing.assert_allclose(result.value, 0.0, atol=1e-6)
    # Formal error tends to σ since no information is gained.
    np.testing.assert_allclose(result.error, 1.0, atol=1e-3)


def test_large_lt_reduces_to_spatial_oi() -> None:
    """Lt → ∞ ⇒ time dimension is ignored, recovering a 2D spatial OI."""
    coords_4d, values, sigma2 = _make_dataset(n=30)
    # 3D-spatial reference: collapse all obs to t=0.
    coords_spatial = coords_4d.copy()
    coords_spatial[:, 2] = 0.0

    q4d = np.array([[5.0, 5.0, 3.7]])
    q_spatial = np.array([[5.0, 5.0, 0.0]])

    oi_4d = pyinterp.OptimalInterpolation(coords_4d, values, sigma2)
    res_4d = oi_4d(q4d, lx=2.0, ly=2.0, lt=1e8, sigma=1.0, k=30)

    oi_sp = pyinterp.OptimalInterpolation(coords_spatial, values, sigma2)
    res_sp = oi_sp(q_spatial, lx=2.0, ly=2.0, lt=1.0, sigma=1.0, k=30)

    np.testing.assert_allclose(res_4d.value, res_sp.value, rtol=1e-6)


def test_scalar_equals_constant_grid() -> None:
    """A constant Grid2D for lx must give the same result as a scalar."""
    coords, values, sigma2 = _make_dataset(n=30)
    oi = pyinterp.OptimalInterpolation(coords, values, sigma2)
    q = np.array([[3.0, 4.0, 5.0], [7.0, 2.0, 6.0]])

    res_scalar = oi(q, lx=2.5, ly=2.5, lt=2.5, sigma=0.5, k=30)

    ax_x = pyinterp.Axis(np.linspace(-1.0, 11.0, 13))
    ax_y = pyinterp.Axis(np.linspace(-1.0, 11.0, 13))
    grid_lx = pyinterp.Grid2D(ax_x, ax_y, np.full((13, 13), 2.5))
    res_grid = oi(q, lx=grid_lx, ly=2.5, lt=2.5, sigma=0.5, k=30)

    np.testing.assert_allclose(res_scalar.value, res_grid.value, rtol=1e-12)
    np.testing.assert_allclose(res_scalar.error, res_grid.error, rtol=1e-12)


def test_invalid_inputs() -> None:
    """The constructor rejects malformed inputs."""
    coords = np.zeros((5, 3))
    values = np.zeros(5)
    sigma2 = np.ones(5)

    with pytest.raises(ValueError, match="obs_coords"):
        pyinterp.OptimalInterpolation(np.zeros((5, 2)), values, sigma2)
    with pytest.raises(ValueError, match="obs_values"):
        pyinterp.OptimalInterpolation(coords, np.zeros(4), sigma2)
    with pytest.raises(ValueError, match="obs_sigma2"):
        pyinterp.OptimalInterpolation(coords, values, np.zeros(4))
    with pytest.raises(ValueError, match="strictly positive"):
        pyinterp.OptimalInterpolation(coords, values, np.zeros(5))
    with pytest.raises(ValueError, match="time_scale"):
        pyinterp.OptimalInterpolation(
            coords, values, sigma2, time_scale=-1.0
        )

    oi = pyinterp.OptimalInterpolation(coords, values, sigma2)
    with pytest.raises(ValueError, match="query_coords"):
        oi(np.zeros((3, 2)), lx=1.0, ly=1.0, lt=1.0, sigma=1.0)
    with pytest.raises(ValueError, match="positive"):
        oi(np.zeros((1, 3)), lx=-1.0, ly=1.0, lt=1.0, sigma=1.0)
    with pytest.raises(ValueError, match="k"):
        oi(np.zeros((1, 3)), lx=1.0, ly=1.0, lt=1.0, sigma=1.0, k=0)


def test_insufficient_neighbors_returns_nan() -> None:
    """If the radius excludes all observations, return NaN gracefully."""
    coords, values, sigma2 = _make_dataset(n=10)
    oi = pyinterp.OptimalInterpolation(coords, values, sigma2)
    # Query far away with a tight radius.
    q = np.array([[1e6, 1e6, 1e6]])
    res = oi(q, lx=1.0, ly=1.0, lt=1.0, sigma=1.0, k=5, radius=1.0)
    assert np.isnan(res.value[0])
    assert np.isnan(res.error[0])
    assert res.neighbors[0] == 0


def test_multithreaded_matches_singlethreaded() -> None:
    """Multi-threaded execution must produce identical numerical results."""
    coords, values, sigma2 = _make_dataset(n=60)
    oi = pyinterp.OptimalInterpolation(coords, values, sigma2)
    rng = np.random.default_rng(7)
    q = rng.uniform([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], size=(50, 3))

    r1 = oi(q, lx=2.0, ly=2.0, lt=2.0, sigma=1.0, k=30, num_threads=1)
    r4 = oi(q, lx=2.0, ly=2.0, lt=2.0, sigma=1.0, k=30, num_threads=4)
    np.testing.assert_allclose(r1.value, r4.value, rtol=1e-12)
    np.testing.assert_allclose(r1.error, r4.error, rtol=1e-12)
    np.testing.assert_array_equal(r1.neighbors, r4.neighbors)


def test_knn_subset_matches_reference_when_k_less_than_n() -> None:
    """With ``k < N`` the OI must use exactly the ``k`` nearest observations.

    The other reference-matching tests use ``k = N`` to remove any k-NN
    approximation. This exercises the production path: the C++ backend
    selects the ``k`` nearest neighbours in the packed metric and builds the
    OI system from that subset only. We reproduce the selection in numpy
    (Euclidean over the packed ``(x, y, t)`` coordinates, since the padded
    ``z`` axis is neutralised and ``time_scale`` defaults to 1) and build
    the reference from exactly those ``k`` observations.
    """
    coords, values, sigma2 = _make_dataset(n=200, seed=11)
    kernel = "gaussian"
    lx = ly = lt = 2.0
    sigma = 1.0
    k = 20

    oi = pyinterp.OptimalInterpolation(
        coords, values, sigma2, covariance=kernel
    )
    rng = np.random.default_rng(321)
    q = rng.uniform([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], size=(10, 3))
    result = oi(q, lx=lx, ly=ly, lt=lt, sigma=sigma, k=k)

    for i in range(q.shape[0]):
        # Packed metric is Euclidean over (x, y, t), so the k nearest are:
        d2 = np.sum((coords - q[i]) ** 2, axis=1)
        idx = np.argsort(d2, kind="stable")[:k]
        ref_v, ref_e = _reference_oi(
            coords[idx], values[idx], sigma2[idx], q[i],
            lx, ly, lt, sigma, kernel,
        )
        assert result.neighbors[i] == k
        np.testing.assert_allclose(
            result.value[i], ref_v, rtol=1e-9, atol=1e-12
        )
        np.testing.assert_allclose(
            result.error[i], ref_e, rtol=1e-7, atol=1e-10
        )


def test_kernel_from_r2_known_values() -> None:
    """Spot-check kernel formulas at r²=0 and r²=1."""
    np.testing.assert_allclose(_kernel_from_r2(np.array([0.0]), "gaussian"), 1.0)
    np.testing.assert_allclose(_kernel_from_r2(np.array([0.0]), "cauchy"), 1.0)
    np.testing.assert_allclose(_kernel_from_r2(np.array([0.0]), "wendland"), 1.0)
    np.testing.assert_allclose(_kernel_from_r2(np.array([1.0]), "gaussian"), np.exp(-1))
    np.testing.assert_allclose(_kernel_from_r2(np.array([1.0]), "cauchy"), 0.5)
    np.testing.assert_allclose(_kernel_from_r2(np.array([1.0]), "wendland"), 0.0)
    np.testing.assert_allclose(_kernel_from_r2(np.array([1.0]), "spherical"), 0.0)

    with pytest.raises(ValueError, match="Unknown covariance kernel"):
        _kernel_from_r2(np.array([0.0]), "not_a_kernel")  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# Geographic-mode tests (Phase 5)
#
# In geographic mode the wrapper converts (lon, lat) → ECEF internally, uses
# `pyinterp.RTree4D` as the spatial index, and the C++ Optimal Interpolation
# backend works in 4D (x_m, y_m, z_m, t_s) Euclidean. The spatial length
# scale `l_spatial` is provided in meters and applied isotropically to the
# three ECEF axes.
# ----------------------------------------------------------------------------


def _ecef_reference_oi(
    obs_lonlat: np.ndarray,  # (N, 3) = (lon, lat, t)
    obs_values: np.ndarray,
    obs_sigma2: np.ndarray,
    query_lonlat: np.ndarray,  # (3,) = (lon, lat, t)
    l_spatial: float,
    lt: float,
    sigma: float,
    kernel: str,
) -> tuple[float, float]:
    """Numpy reference: convert to ECEF, run anisotropic OI in (x, y, z, t)."""
    from pyinterp.core.geometry.geographic import (
        Coordinates,
        Spheroid,
    )

    sph = Spheroid()
    coords = Coordinates(sph)

    obs_x, obs_y, obs_z = coords.lla_to_ecef(
        obs_lonlat[:, 0], obs_lonlat[:, 1], np.zeros(obs_lonlat.shape[0])
    )
    obs4d = np.column_stack([obs_x, obs_y, obs_z, obs_lonlat[:, 2]])

    q_x, q_y, q_z = coords.lla_to_ecef(
        np.array([query_lonlat[0]]),
        np.array([query_lonlat[1]]),
        np.array([0.0]),
    )
    q4d = np.array([q_x[0], q_y[0], q_z[0], query_lonlat[2]])

    dx = (obs4d[:, None, :] - obs4d[None, :, :])
    scale = np.array([l_spatial, l_spatial, l_spatial, lt])
    r2_oo = np.sum((dx / scale) ** 2, axis=-1)
    diff_og = q4d - obs4d
    r2_og = np.sum((diff_og / scale) ** 2, axis=-1)

    sig2 = sigma * sigma
    c_oo = sig2 * _kernel_from_r2(r2_oo, kernel)
    c_og = sig2 * _kernel_from_r2(r2_og, kernel)
    n = obs4d.shape[0]
    c_oo[np.arange(n), np.arange(n)] += obs_sigma2

    w = np.linalg.solve(c_oo, c_og)
    value = float(w @ obs_values)
    err2 = float(sig2 - c_og @ w)
    return value, np.sqrt(max(err2, 0.0))


def _geographic_dataset(n: int = 30, seed: int = 0):
    """Random (lon, lat, t) observations in a small mid-latitude region."""
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-5.0, 5.0, n)
    lat = rng.uniform(40.0, 50.0, n)
    t = rng.uniform(0.0, 86400.0, n)  # seconds in a day
    obs = np.column_stack([lon, lat, t])
    values = np.sin(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))
    sigma2 = 0.01 + 0.005 * rng.random(n)
    return obs, values, sigma2


@pytest.mark.parametrize("kernel", _KERNELS)
def test_geographic_matches_ecef_reference(kernel: str) -> None:
    """Geographic-mode OI matches a hand-rolled ECEF-space reference."""
    obs, values, sigma2 = _geographic_dataset(n=20, seed=1)
    oi = pyinterp.OptimalInterpolation(
        obs, values, sigma2,
        covariance=kernel,
        coordinate_system="geographic",
    )
    # Single query in the middle of the region.
    q = np.array([[0.0, 45.0, 43200.0]])  # noon UTC
    # 150 km horizontal, 6 hours temporal.
    result = oi(q, l_spatial=150e3, lt=6 * 3600.0, sigma=1.0, k=20)

    ref_v, ref_e = _ecef_reference_oi(
        obs, values, sigma2, q[0],
        l_spatial=150e3, lt=6 * 3600.0, sigma=1.0, kernel=kernel,
    )
    np.testing.assert_allclose(result.value[0], ref_v, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.error[0], ref_e, rtol=1e-7, atol=1e-10)


def test_geographic_grid2d_sampled_at_lon_lat() -> None:
    """A Grid2D for ``l_spatial`` must be evaluated at the user's (lon, lat).

    This is the regression test for the design discussion: even though the
    R-tree internally stores ECEF, the Grid2D lookup happens at the
    untouched ``(lon, lat)`` that the user passed.
    """
    obs, values, sigma2 = _geographic_dataset(n=20, seed=42)
    oi = pyinterp.OptimalInterpolation(
        obs, values, sigma2, coordinate_system="geographic"
    )

    # Constant Grid2D over the lon/lat domain that overlaps our data.
    ax_lon = pyinterp.Axis(np.linspace(-10.0, 10.0, 21))
    ax_lat = pyinterp.Axis(np.linspace(35.0, 55.0, 21))
    grid_ls = pyinterp.Grid2D(
        ax_lon, ax_lat, np.full((21, 21), 200e3)
    )

    q = np.array([[1.0, 44.0, 43200.0]])
    res_scalar = oi(q, l_spatial=200e3, lt=6 * 3600.0, sigma=1.0, k=20)
    res_grid = oi(q, l_spatial=grid_ls, lt=6 * 3600.0, sigma=1.0, k=20)
    np.testing.assert_allclose(res_scalar.value, res_grid.value, rtol=1e-12)
    np.testing.assert_allclose(res_scalar.error, res_grid.error, rtol=1e-12)


def test_geographic_grid2d_varying_values() -> None:
    """A spatially varying Grid2D for ``l_spatial`` is honoured per query."""
    obs, values, sigma2 = _geographic_dataset(n=40, seed=7)
    oi = pyinterp.OptimalInterpolation(
        obs, values, sigma2, coordinate_system="geographic"
    )

    # Build a Grid2D that maps a different L_spatial depending on the lat band.
    # Bilinear interpolation will return the cell-centred value far from the
    # nearest edge, so we put very contrasted values in well-separated cells.
    ax_lon = pyinterp.Axis(np.array([-5.0, 5.0]))
    ax_lat = pyinterp.Axis(np.array([40.0, 50.0]))
    # Array shape is (n_lon, n_lat): row index is lon, column index is lat.
    # To make L_spatial increase with latitude (small in the south, large in
    # the north), the second axis (lat) must carry the contrast.
    grid_ls = pyinterp.Grid2D(
        ax_lon, ax_lat,
        np.array([[50e3, 500e3], [50e3, 500e3]]),  # increases with latitude
    )

    # Two queries: south (small L) and north (large L).
    q = np.array([
        [0.0, 41.0, 43200.0],  # south → ~50 km
        [0.0, 49.0, 43200.0],  # north → ~500 km
    ])
    res = oi(q, l_spatial=grid_ls, lt=6 * 3600.0, sigma=1.0, k=20)

    # Compute the manually sampled values for cross-checking.
    expected_ls = pyinterp.bivariate(
        grid_ls, q[:, 0], q[:, 1], "bilinear"
    )
    assert expected_ls[0] < expected_ls[1]  # increasing northward

    # And confirm the OI uses the right ``l_spatial`` per query: a one-shot
    # ``l_spatial = expected_ls[i]`` scalar must reproduce ``res.value[i]``.
    for i in range(2):
        ref = oi(
            q[i : i + 1],
            l_spatial=float(expected_ls[i]),
            lt=6 * 3600.0, sigma=1.0, k=20,
        )
        np.testing.assert_allclose(res.value[i], ref.value[0], rtol=1e-12)
        np.testing.assert_allclose(res.error[i], ref.error[0], rtol=1e-12)


def test_geographic_zero_noise_recovers_observations() -> None:
    """σ²_obs → 0 in geographic mode reconstructs obs at their own positions."""
    obs, values, _ = _geographic_dataset(n=15, seed=2)
    tiny = np.full(obs.shape[0], 1e-12)
    oi = pyinterp.OptimalInterpolation(
        obs, values, tiny, coordinate_system="geographic"
    )
    res = oi(obs, l_spatial=100e3, lt=3600.0, sigma=1.0, k=obs.shape[0])
    np.testing.assert_allclose(res.value, values, atol=1e-6)
    np.testing.assert_allclose(res.error, 0.0, atol=1e-5)


def test_geographic_rejects_cartesian_kwargs() -> None:
    """In geographic mode, passing ``lx``/``ly`` is an error."""
    obs, values, sigma2 = _geographic_dataset(n=10)
    oi = pyinterp.OptimalInterpolation(
        obs, values, sigma2, coordinate_system="geographic"
    )
    q = np.array([[0.0, 45.0, 0.0]])
    with pytest.raises(ValueError, match="geographic mode"):
        oi(q, lx=1.0, ly=1.0, lt=3600.0, sigma=1.0)
    with pytest.raises(ValueError, match="l_spatial is required"):
        oi(q, lt=3600.0, sigma=1.0)


def test_cartesian_rejects_l_spatial() -> None:
    """In cartesian mode, ``l_spatial`` is rejected to avoid silent confusion."""
    obs, values, sigma2 = _make_dataset(n=10)
    oi = pyinterp.OptimalInterpolation(obs, values, sigma2)
    q = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="only valid in geographic mode"):
        oi(q, lx=1.0, ly=1.0, lt=1.0, sigma=1.0, l_spatial=100.0)


def test_geographic_time_scale_invariant_with_all_neighbors() -> None:
    """In geographic mode ``time_scale`` must cancel out of the covariance.

    It only rebalances the k-NN neighbour selection, so when every
    observation is used (``k = N``) the analysis is independent of
    ``time_scale``. A regression that scaled the packed time but not ``lt``
    (or vice versa) would change the result and fail here.
    """
    obs, values, sigma2 = _geographic_dataset(n=20, seed=5)
    q = np.array([[0.0, 45.0, 43200.0]])
    kw = {"l_spatial": 150e3, "lt": 6 * 3600.0, "sigma": 1.0, "k": obs.shape[0]}

    oi_ref = pyinterp.OptimalInterpolation(
        obs, values, sigma2, coordinate_system="geographic"
    )  # default time_scale = 1.0
    oi_scaled = pyinterp.OptimalInterpolation(
        obs, values, sigma2, coordinate_system="geographic", time_scale=500.0
    )
    r_ref = oi_ref(q, **kw)
    r_scaled = oi_scaled(q, **kw)
    np.testing.assert_allclose(
        r_ref.value, r_scaled.value, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        r_ref.error, r_scaled.error, rtol=1e-7, atol=1e-10
    )


def test_invalid_coordinate_system() -> None:
    """Unknown coordinate_system string is rejected up-front."""
    obs, values, sigma2 = _make_dataset(n=5)
    with pytest.raises(ValueError, match="coordinate_system"):
        pyinterp.OptimalInterpolation(
            obs, values, sigma2, coordinate_system="polar"  # type: ignore[arg-type]
        )

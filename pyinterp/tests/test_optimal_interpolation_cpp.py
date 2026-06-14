# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for the C++ OptimalInterpolation backend exposed on RTree4D.

These tests validate:

1. **Numerical parity** between the C++ implementation and a pure-numpy
   reference written from scratch (7 covariance kernels).
2. **Backward agreement** between the C++ 4D path (with a flat 4th axis)
   and the Phase-0 Python 3D estimator.
3. **Limit behaviours**: σ²_obs → 0 recovers observations exactly,
   σ²_obs → ∞ collapses to zero, missing neighbours yield NaN.
4. **Determinism under threading**: single-thread and multi-thread runs
   produce bit-identical results.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyinterp
from pyinterp.core.config.rtree import CovarianceFunction
from pyinterp.core.config.rtree import OptimalInterpolation as OIConfig
from pyinterp.optimal_interpolation import _kernel_from_r2


_KERNEL_NAME_TO_ENUM = {
    "gaussian": CovarianceFunction.GAUSSIAN,
    "cauchy": CovarianceFunction.CAUCHY,
    "matern_12": CovarianceFunction.MATERN_12,
    "matern_32": CovarianceFunction.MATERN_32,
    "matern_52": CovarianceFunction.MATERN_52,
    "spherical": CovarianceFunction.SPHERICAL,
    "wendland": CovarianceFunction.WENDLAND,
}


def _numpy_oi_4d(
    obs: np.ndarray,
    values: np.ndarray,
    sigma2: np.ndarray,
    query: np.ndarray,
    lx: float,
    ly: float,
    lz: float,
    lt: float,
    sigma: float,
    kernel: str,
) -> tuple[float, float]:
    """Direct 4D OI on a single query point — no R-tree, no scratch reuse."""
    inv_l = np.array([1.0 / lx, 1.0 / ly, 1.0 / lz, 1.0 / lt])
    diff_oo = (obs[:, None, :] - obs[None, :, :]) * inv_l
    r2_oo = np.sum(diff_oo**2, axis=-1)
    diff_og = (query - obs) * inv_l
    r2_og = np.sum(diff_og**2, axis=-1)

    sig2 = sigma * sigma
    c_oo = sig2 * _kernel_from_r2(r2_oo, kernel)
    c_og = sig2 * _kernel_from_r2(r2_og, kernel)
    n = obs.shape[0]
    c_oo[np.arange(n), np.arange(n)] += sigma2

    w = np.linalg.solve(c_oo, c_og)
    value = float(w @ values)
    err2 = float(sig2 - c_og @ w)
    return value, np.sqrt(max(err2, 0.0))


def _make_dataset(n: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    obs = rng.uniform([0, 0, 0, 0], [10, 10, 10, 10], size=(n, 4))
    values = (
        np.sin(obs[:, 0])
        * np.cos(obs[:, 1])
        + 0.2 * obs[:, 2]
        - 0.1 * obs[:, 3]
    )
    sigma2 = 0.01 + 0.005 * rng.random(n)
    return obs, values, sigma2


@pytest.mark.parametrize("kernel", list(_KERNEL_NAME_TO_ENUM))
def test_cpp_matches_numpy_reference(kernel: str) -> None:
    """For each kernel, the C++ OI must match the numpy reference."""
    obs, values, sigma2 = _make_dataset(n=30, seed=1)
    tree = pyinterp.RTree4D()
    tree.packing(obs, values, sigma2)

    rng = np.random.default_rng(99)
    q = rng.uniform([0, 0, 0, 0], [10, 10, 10, 10], size=(6, 4))
    m = q.shape[0]
    lx = np.full(m, 2.0)
    ly = np.full(m, 2.5)
    lz = np.full(m, 1.5)
    lt = np.full(m, 3.0)
    sig = np.full(m, 1.0)

    # k = N guarantees that every neighbour is used → no k-NN artefact.
    cfg = OIConfig().with_covariance_model(
        _KERNEL_NAME_TO_ENUM[kernel]
    ).with_k(30)
    cpp_v, cpp_e, cpp_n = tree.optimal_interpolation(
        q, lx, ly, lz, lt, sig, cfg
    )

    for i in range(m):
        ref_v, ref_e = _numpy_oi_4d(
            obs, values, sigma2, q[i], 2.0, 2.5, 1.5, 3.0, 1.0, kernel
        )
        np.testing.assert_allclose(cpp_v[i], ref_v, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(cpp_e[i], ref_e, rtol=1e-7, atol=1e-10)
        assert cpp_n[i] == 30


def test_cpp_4d_matches_python_3d_when_lz_infinite() -> None:
    """Setting Lz → ∞ collapses the C++ 4D OI onto the Python 3D OI."""
    rng = np.random.default_rng(7)
    n = 25
    obs3d = rng.uniform([0, 0, 0], [10, 10, 10], size=(n, 3))
    values = np.sin(obs3d[:, 0]) + obs3d[:, 2] * 0.3
    sigma2 = np.full(n, 0.02)

    # Pad to 4D with z = 0 everywhere; we will neutralise z via Lz → ∞.
    obs4d = np.column_stack([obs3d[:, 0], obs3d[:, 1], np.zeros(n), obs3d[:, 2]])

    # Python 3D OI: (x, y, t) → use Phase-0 API.
    py_oi = pyinterp.OptimalInterpolation(obs3d, values, sigma2)
    q3d = np.array([[5.0, 5.0, 5.0]])
    py_res = py_oi(q3d, lx=2.0, ly=2.0, lt=2.0, sigma=1.0, k=n)

    # C++ 4D OI: same point, padded to (x, y, z=0, t), Lz huge.
    tree = pyinterp.RTree4D()
    tree.packing(obs4d, values, sigma2)
    cfg = OIConfig().with_covariance_model(CovarianceFunction.GAUSSIAN).with_k(n)
    q4d = np.array([[5.0, 5.0, 0.0, 5.0]])
    cpp_v, cpp_e, _ = tree.optimal_interpolation(
        q4d,
        np.full(1, 2.0),
        np.full(1, 2.0),
        np.full(1, 1e12),   # Lz → ∞ : z contribution vanishes
        np.full(1, 2.0),
        np.full(1, 1.0),
        cfg,
    )
    np.testing.assert_allclose(cpp_v, py_res.value, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(cpp_e, py_res.error, rtol=1e-7, atol=1e-10)


def test_zero_noise_recovers_observation() -> None:
    """σ²_obs → 0: query at an obs location returns the obs value."""
    obs, values, _ = _make_dataset(n=20, seed=2)
    tiny = np.full(obs.shape[0], 1e-12)
    tree = pyinterp.RTree4D()
    tree.packing(obs, values, tiny)

    cfg = OIConfig().with_covariance_model(CovarianceFunction.GAUSSIAN).with_k(
        obs.shape[0]
    )
    n = obs.shape[0]
    lx = np.full(n, 1.0)
    ly = np.full(n, 1.0)
    lz = np.full(n, 1.0)
    lt = np.full(n, 1.0)
    sig = np.full(n, 1.0)
    v, e, _ = tree.optimal_interpolation(obs, lx, ly, lz, lt, sig, cfg)
    np.testing.assert_allclose(v, values, atol=1e-6)
    np.testing.assert_allclose(e, 0.0, atol=1e-5)


def test_infinite_noise_collapses_to_prior() -> None:
    """σ²_obs → ∞: the analysis is the zero-mean prior."""
    obs, values, _ = _make_dataset(n=15, seed=3)
    huge = np.full(obs.shape[0], 1e12)
    tree = pyinterp.RTree4D()
    tree.packing(obs, values, huge)

    cfg = OIConfig().with_covariance_model(CovarianceFunction.GAUSSIAN).with_k(15)
    q = np.array([[5.0, 5.0, 5.0, 5.0]])
    v, e, _ = tree.optimal_interpolation(
        q, np.full(1, 1.0), np.full(1, 1.0), np.full(1, 1.0), np.full(1, 1.0),
        np.full(1, 1.0), cfg,
    )
    np.testing.assert_allclose(v, 0.0, atol=1e-6)
    np.testing.assert_allclose(e, 1.0, atol=1e-3)  # error → σ


def test_no_neighbor_returns_nan() -> None:
    """Far query with tight radius → NaN value, NaN error, 0 neighbours."""
    obs, values, sigma2 = _make_dataset(n=10, seed=4)
    tree = pyinterp.RTree4D()
    tree.packing(obs, values, sigma2)

    cfg = (
        OIConfig()
        .with_covariance_model(CovarianceFunction.GAUSSIAN)
        .with_k(5)
        .with_radius(1.0)
    )
    q = np.array([[1e6, 1e6, 1e6, 1e6]])
    v, e, n = tree.optimal_interpolation(
        q, np.full(1, 1.0), np.full(1, 1.0), np.full(1, 1.0), np.full(1, 1.0),
        np.full(1, 1.0), cfg,
    )
    assert np.isnan(v[0])
    assert np.isnan(e[0])
    assert n[0] == 0


def test_multithreaded_equals_single_threaded() -> None:
    """Identical numerical output regardless of the thread count."""
    obs, values, sigma2 = _make_dataset(n=60, seed=5)
    tree = pyinterp.RTree4D()
    tree.packing(obs, values, sigma2)

    rng = np.random.default_rng(101)
    q = rng.uniform([0, 0, 0, 0], [10, 10, 10, 10], size=(40, 4))
    m = q.shape[0]
    args = (np.full(m, 2.0),) * 4 + (np.full(m, 1.0),)

    cfg1 = (
        OIConfig()
        .with_covariance_model(CovarianceFunction.GAUSSIAN)
        .with_k(30)
        .with_num_threads(1)
    )
    cfg4 = (
        OIConfig()
        .with_covariance_model(CovarianceFunction.GAUSSIAN)
        .with_k(30)
        .with_num_threads(4)
    )
    v1, e1, n1 = tree.optimal_interpolation(q, *args, cfg1)
    v4, e4, n4 = tree.optimal_interpolation(q, *args, cfg4)

    np.testing.assert_array_equal(v1, v4)
    np.testing.assert_array_equal(e1, e4)
    np.testing.assert_array_equal(n1, n4)


def test_input_shape_validation() -> None:
    """Misaligned per-query arrays produce a clear error."""
    tree = pyinterp.RTree4D()
    obs, values, sigma2 = _make_dataset(n=5)
    tree.packing(obs, values, sigma2)

    q = np.array([[1.0, 2.0, 3.0, 4.0]])
    bad = np.array([1.0, 2.0])  # wrong length
    cfg = OIConfig().with_k(3)
    with pytest.raises(ValueError, match="same length"):
        tree.optimal_interpolation(
            q, bad, np.full(1, 1.0), np.full(1, 1.0), np.full(1, 1.0),
            np.full(1, 1.0), cfg,
        )

    with pytest.raises(ValueError, match="\\(m, 4\\)"):
        tree.optimal_interpolation(
            np.zeros((1, 3)),
            np.full(1, 1.0), np.full(1, 1.0), np.full(1, 1.0),
            np.full(1, 1.0), np.full(1, 1.0), cfg,
        )

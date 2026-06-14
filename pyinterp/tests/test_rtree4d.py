# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for the 4D Cartesian RTree.

The 4D tree is the indexing primitive feeding the Optimal Interpolation
estimator: it stores per-observation error variance alongside each value,
and exposes k-NN queries together with the ``optimal_interpolation`` backend
(no kriging / IDW / RBF / window-function methods — those remain on RTree3D).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import pyinterp
from pyinterp.core.config.rtree import Query


def _make_dataset(n: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 10.0, size=(n, 4))
    values = rng.uniform(-1.0, 1.0, size=n)
    sigma2 = 0.01 + 0.005 * rng.random(n)
    return coords, values, sigma2


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_construct_and_pack(dtype: str) -> None:
    """Default constructor + packing succeeds for both precisions."""
    tree = pyinterp.RTree4D(dtype=dtype)
    coords, values, sigma2 = _make_dataset()
    tree.packing(
        coords.astype(dtype), values.astype(dtype), sigma2.astype(dtype)
    )
    assert tree.size() == coords.shape[0]
    assert not tree.empty()
    bounds = tree.bounds()
    assert bounds is not None
    lower, upper = bounds
    assert lower.shape == (4,)
    assert upper.shape == (4,)
    assert np.all(lower <= upper)


def test_query_returns_value_and_sigma2() -> None:
    """k-NN query returns three aligned matrices: distance, value, sigma2."""
    tree = pyinterp.RTree4D()
    coords = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 5.0],
        ]
    )
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    sigma2 = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    tree.packing(coords, values, sigma2)

    q = np.array([[0.1, 0.0, 0.0, 0.0]])
    distances, vals, sig2 = tree.query(q, Query().with_k(2))
    assert distances.shape == (1, 2)
    assert vals.shape == (1, 2)
    assert sig2.shape == (1, 2)

    # Closest is (0,0,0,0)
    assert vals[0, 0] == 10.0
    assert sig2[0, 0] == 0.01


def test_query_fills_nan_when_fewer_neighbors() -> None:
    """If fewer than k neighbors are found (e.g. via radius), trailing cells are NaN."""
    tree = pyinterp.RTree4D()
    coords, values, sigma2 = _make_dataset(n=5)
    tree.packing(coords, values, sigma2)

    # Far query with tight radius → 0 neighbors
    q = np.array([[1e6, 1e6, 1e6, 1e6]])
    cfg = Query().with_k(3).with_radius(1.0)
    distances, vals, sig2 = tree.query(q, cfg)
    assert np.all(np.isnan(distances))
    assert np.all(np.isnan(vals))
    assert np.all(np.isnan(sig2))


def test_sigma2_must_be_positive() -> None:
    """Zero or negative measurement variance is rejected at packing time."""
    tree = pyinterp.RTree4D()
    coords = np.zeros((3, 4))
    values = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="strictly positive"):
        tree.packing(coords, values, np.array([0.01, 0.0, 0.02]))
    with pytest.raises(ValueError, match="strictly positive"):
        tree.packing(coords, values, np.array([0.01, -1.0, 0.02]))


def test_input_shape_validation() -> None:
    """Wrong coordinate width or mismatched lengths produce clear errors."""
    tree = pyinterp.RTree4D()
    with pytest.raises(ValueError, match="\\(n, 4\\)"):
        tree.packing(
            np.zeros((3, 3)), np.array([1.0, 2.0, 3.0]),
            np.array([0.1, 0.2, 0.3])
        )
    with pytest.raises(ValueError, match="same length"):
        tree.packing(
            np.zeros((3, 4)), np.array([1.0, 2.0]),
            np.array([0.1, 0.2, 0.3])
        )


def test_insert_appends_to_existing_tree() -> None:
    """`insert` adds points without erasing the existing index."""
    tree = pyinterp.RTree4D()
    coords1, values1, sigma2_1 = _make_dataset(n=10, seed=0)
    tree.packing(coords1, values1, sigma2_1)
    assert tree.size() == 10

    coords2, values2, sigma2_2 = _make_dataset(n=5, seed=1)
    tree.insert(coords2, values2, sigma2_2)
    assert tree.size() == 15


def test_pickle_round_trip() -> None:
    """Serialisation preserves coordinates, values and sigma2."""
    tree = pyinterp.RTree4D()
    coords, values, sigma2 = _make_dataset(n=30, seed=42)
    tree.packing(coords, values, sigma2)

    restored = pickle.loads(pickle.dumps(tree))
    assert restored.size() == tree.size()

    # Same query on both must yield identical (value, sigma2) pairs.
    q = np.array([[5.0, 5.0, 5.0, 5.0]])
    d_orig, v_orig, s_orig = tree.query(q, Query().with_k(10))
    d_rest, v_rest, s_rest = restored.query(q, Query().with_k(10))
    np.testing.assert_array_equal(d_orig, d_rest)
    np.testing.assert_array_equal(v_orig, v_rest)
    np.testing.assert_array_equal(s_orig, s_rest)


def test_clear_empties_the_tree() -> None:
    """`clear` resets the tree to empty."""
    tree = pyinterp.RTree4D()
    coords, values, sigma2 = _make_dataset(n=10)
    tree.packing(coords, values, sigma2)
    assert tree.size() == 10
    tree.clear()
    assert tree.size() == 0
    assert tree.empty()
    assert tree.bounds() is None


def test_distance_is_euclidean_in_4d() -> None:
    """The packed metric is the plain Euclidean distance in ℝ⁴."""
    tree = pyinterp.RTree4D()
    # Single point at the origin
    tree.packing(
        np.array([[0.0, 0.0, 0.0, 0.0]]),
        np.array([1.0]),
        np.array([0.1]),
    )
    # Query at (1, 1, 1, 1): distance = sqrt(4) = 2.
    distances, _, _ = tree.query(
        np.array([[1.0, 1.0, 1.0, 1.0]]), Query().with_k(1)
    )
    np.testing.assert_allclose(distances[0, 0], 2.0, rtol=1e-12)


def test_dtype_specific_classes() -> None:
    """Both `RTree4DFloat32` and `RTree4DFloat64` are concrete classes."""
    assert pyinterp.RTree4DFloat32 is not pyinterp.RTree4DFloat64
    tree32 = pyinterp.RTree4D(dtype="float32")
    tree64 = pyinterp.RTree4D(dtype="float64")
    assert isinstance(tree32, pyinterp.RTree4DFloat32)
    assert isinstance(tree64, pyinterp.RTree4DFloat64)

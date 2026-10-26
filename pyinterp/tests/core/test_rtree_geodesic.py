# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Geodesic distances in RTree3D, and the state saved by pickling.

The geographic RTree searches its nodes with geodesic distances and serves as
the reference: RTree3D in geodesic mode must find the same neighbors, at the
same distances.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ...core import RTree3D
from ...core.config import rtree as config
from ...core.geometry.geographic import RTree, Spheroid


if TYPE_CHECKING:
    from ...core import RTree3DDistance


def _points(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform points on the sphere, plus points at the poles and seam."""
    lon = rng.uniform(-180, 180, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    special = np.array(
        [[0.0, 90.0], [45.0, -90.0], [180.0, 0.0], [-180.0, 10.0], [179.9, 0]]
    )
    return np.vstack((np.column_stack((lon, lat)), special))


@pytest.fixture(scope="module")
def trees() -> tuple[np.ndarray, RTree, object]:
    """Index the same points in the geographic tree and a geodesic RTree3D."""
    rng = np.random.default_rng(42)
    coordinates = _points(rng, 20_000)
    values = rng.standard_normal(len(coordinates))
    geographic = RTree()
    geographic.packing(coordinates, values)
    geodesic = RTree3D(Spheroid(), distance="geodesic")
    geodesic.packing(coordinates, values)
    queries = np.vstack(
        (
            _points(rng, 2000),
            # Near the poles and on both sides of the antimeridian
            [[10.0, 89.99], [-170.0, -89.95], [179.99, 5.0], [-179.99, 5.0]],
        )
    )
    return queries, geographic, geodesic


def test_query_matches_geographic_tree(
    trees: tuple[np.ndarray, RTree, object],
) -> None:
    """Same neighbors, in the same order, at the same distances."""
    queries, geographic, geodesic = trees
    settings = config.Query().with_k(16)
    expected_distances, expected_values = geographic.query(queries, settings)
    distances, values = geodesic.query(queries, settings)  # type: ignore[attr-defined]
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_allclose(distances, expected_distances, rtol=1e-9)


@pytest.mark.parametrize(
    "method",
    ["inverse_distance_weighting", "window_function"],
)
def test_interpolators_match_geographic_tree(
    trees: tuple[np.ndarray, RTree, object], method: str
) -> None:
    """The weights derived from the distances are geodesic too."""
    queries, geographic, geodesic = trees
    settings = (
        config.InverseDistanceWeighting()
        if method == "inverse_distance_weighting"
        else config.InterpolationWindow()
    ).with_k(16)
    expected, expected_counts = getattr(geographic, method)(queries, settings)
    result, counts = getattr(geodesic, method)(queries, settings)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-12)


def test_radius_and_small_trees() -> None:
    """Radius filtering, and more neighbors requested than points."""
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    values = np.array([1.0, 2.0, 3.0])
    geographic = RTree()
    geographic.packing(coordinates, values)
    geodesic = RTree3D(Spheroid(), distance="geodesic")
    geodesic.packing(coordinates, values)
    query = np.array([[0.2, 0.1]])
    for settings in (
        config.Query().with_k(10),
        config.Query().with_k(10).with_radius(150_000.0),
    ):
        expected_distances, expected_values = geographic.query(query, settings)
        distances, found = geodesic.query(query, settings)
        np.testing.assert_array_equal(found, expected_values)
        np.testing.assert_allclose(distances, expected_distances, rtol=1e-9)

    empty = RTree3D(Spheroid(), distance="geodesic")
    distances, found = empty.query(query, config.Query().with_k(3))
    assert np.isnan(distances).all()
    assert np.isnan(found).all()


def test_float32() -> None:
    """Single precision trees use the same search."""
    rng = np.random.default_rng(1)
    coordinates = _points(rng, 2000)
    values = rng.standard_normal(len(coordinates))
    reference = RTree3D(Spheroid(), distance="geodesic")
    reference.packing(coordinates, values)
    single = RTree3D(Spheroid(), dtype="float32", distance="geodesic")
    single.packing(coordinates.astype("float32"), values.astype("float32"))
    query = _points(rng, 100)
    settings = config.Query().with_k(4)
    distances, _ = single.query(query.astype("float32"), settings)
    expected, _ = reference.query(query, settings)
    # ECEF coordinates in single precision are resolved to about 0.5 m.
    np.testing.assert_allclose(distances, expected, rtol=1e-4, atol=5.0)


def test_invalid_settings() -> None:
    """Geodesic distances need longitudes, latitudes and no altitude."""
    assert RTree3D().distance == "euclidean"
    assert RTree3D(Spheroid(), distance="geodesic").distance == "geodesic"
    with pytest.raises(ValueError, match="require a spheroid"):
        RTree3D(distance="geodesic")
    with pytest.raises(ValueError, match="euclidean"):
        RTree3D(Spheroid(), distance="haversine")  # type: ignore[call-overload]

    tree = RTree3D(Spheroid(), distance="geodesic")
    with pytest.raises(ValueError, match="altitudes must be zero"):
        tree.packing(np.array([[0.0, 0.0, 10.0]]), np.array([1.0]))
    # Zero altitudes are accepted
    tree.packing(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]), np.ones(2))
    with pytest.raises(ValueError, match="altitudes must be zero"):
        tree.query(np.array([[0.0, 0.0, 1.0]]))


@pytest.mark.parametrize(
    ("spheroid", "distance"),
    [
        (None, "euclidean"),
        (Spheroid(), "euclidean"),
        (Spheroid(6378137.0, 1 / 300.0), "euclidean"),
        (Spheroid(), "geodesic"),
    ],
)
def test_pickle_keeps_settings(
    spheroid: Spheroid | None, distance: RTree3DDistance
) -> None:
    """The coordinate system and the distance survive pickling."""
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tree = RTree3D(spheroid, distance=distance)
    tree.packing(coordinates, np.arange(3.0))
    restored = pickle.loads(pickle.dumps(tree))

    assert restored.distance == distance
    if spheroid is None:
        assert restored.spheroid is None
    else:
        assert restored.spheroid is not None
        assert restored.spheroid.flattening == spheroid.flattening
        assert restored.spheroid.semi_major_axis == spheroid.semi_major_axis
    query = np.array([[0.1, 0.0]])
    settings = config.Query().with_k(2)
    for expected, actual in zip(
        tree.query(query, settings),
        restored.query(query, settings),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_old_pickle_state() -> None:
    """States saved without the coordinate system load as WGS84 trees."""
    tree = RTree3D(Spheroid())
    tree.packing(np.array([[0.0, 0.0], [1.0, 0.0]]), np.arange(2.0))
    state = tree.__getstate__()
    # As pickle does: an uninitialized instance, then its state
    cls = type(tree)
    restored = cls.__new__(cls)
    restored.__setstate__(state[:1])
    assert restored.spheroid is not None
    assert restored.distance == "euclidean"
    assert restored.size() == 2

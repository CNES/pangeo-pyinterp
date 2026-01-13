# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for RTree3D with parametrized Float32/Float64 tests."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ...core import RTree3D, RTree3DFloat32, RTree3DFloat64
from ...core.geometry.geographic import Spheroid


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest

    from ...core import RTree3DHolder


@pytest.fixture(params=["float64", "float32"])
def dtype_param(request: FixtureRequest) -> str:
    """Parametrize tests with dtype strings."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def rtree_float64() -> RTree3DHolder[np.float64]:
    """Create an RTree3D with float64 dtype."""
    return RTree3D(dtype="float64")


@pytest.fixture
def rtree_float32() -> RTree3DHolder[np.float32]:
    """Create an RTree3D with float32 dtype."""
    return RTree3D(dtype="float32")


@pytest.fixture(params=[np.float64, np.float32])
def float_dtype(request: FixtureRequest) -> type[np.float64 | np.float32]:
    """Parametrize tests with numpy float dtypes."""
    return request.param


class TestRTree3DInitialization:
    """Tests for RTree3D initialization."""

    def test_init_no_spheroid_float64(
        self, rtree_float64: RTree3DFloat64
    ) -> None:
        """Test RTree3D initialization without spheroid."""
        assert rtree_float64.spheroid is None
        assert rtree_float64.empty()
        assert rtree_float64.size() == 0

    def test_init_no_spheroid_float32(
        self, rtree_float32: RTree3DFloat32
    ) -> None:
        """Test RTree3D initialization without spheroid."""
        assert rtree_float32.spheroid is None
        assert rtree_float32.empty()
        assert rtree_float32.size() == 0

    def test_init_with_spheroid_float64(self) -> None:
        """Test RTree3D initialization with spheroid."""
        spheroid = Spheroid()
        tree = RTree3D(spheroid=spheroid, dtype="float64")
        assert tree.spheroid is not None
        assert tree.empty()
        assert tree.size() == 0

    def test_init_with_spheroid_float32(self) -> None:
        """Test RTree3D initialization with spheroid."""
        spheroid = Spheroid()
        tree = RTree3D(spheroid=spheroid, dtype="float32")
        assert tree.spheroid is not None
        assert tree.empty()
        assert tree.size() == 0

    def test_init_parametrized(self, dtype_param: str) -> None:
        """Test initialization for both Float64 and Float32."""
        tree = RTree3D(dtype=dtype_param)
        assert tree.empty()
        assert tree.size() == 0


class TestRTree3DPacking:
    """Tests for RTree3D packing operations."""

    def test_packing_3d_cartesian_float64(
        self, rtree_float64: RTree3DFloat64
    ) -> None:
        """Test packing 3D Cartesian points with float64."""
        coordinates = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )

        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)

        rtree_float64.packing(coordinates, values)

        assert not rtree_float64.empty()
        assert rtree_float64.size() == 5

    def test_packing_3d_cartesian_float32(
        self, rtree_float32: RTree3DFloat32
    ) -> None:
        """Test packing 3D Cartesian points with float32."""
        coordinates = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)

        rtree_float32.packing(coordinates, values)

        assert not rtree_float32.empty()
        assert rtree_float32.size() == 5

    def test_packing_2d_float64(self, rtree_float64: RTree3DFloat64) -> None:
        """Test packing 2D points (Z=0) with float64."""
        coordinates = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
        )

        values = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)

        rtree_float64.packing(coordinates, values)

        assert not rtree_float64.empty()
        assert rtree_float64.size() == 4

    def test_packing_2d_float32(self, rtree_float32: RTree3DFloat32) -> None:
        """Test packing 2D points (Z=0) with float32."""
        coordinates = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
        )

        values = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

        rtree_float32.packing(coordinates, values)

        assert not rtree_float32.empty()
        assert rtree_float32.size() == 4

    def test_packing_parametrized_3d(
        self,
        dtype_param: str,
        float_dtype: type[np.float64 | np.float32],
    ) -> None:
        """Test packing 3D points for both Float64 and Float32."""
        tree = RTree3D(dtype=dtype_param)  # type: ignore[call-arg]

        coordinates = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float_dtype,
        )

        values = np.array([10.0, 20.0, 30.0], dtype=float_dtype)

        tree.packing(coordinates, values)

        assert tree.size() == 3
        assert not tree.empty()

    def test_packing_geodetic_float64(
        self, rtree_float64: RTree3DFloat64
    ) -> None:
        """Test packing geodetic (lon, lat, alt) coordinates with float64."""
        spheroid = Spheroid()
        tree = RTree3D(spheroid=spheroid, dtype="float64")
        # Sample points around Paris
        coordinates = np.array(
            [
                [2.35, 48.85, 0.0],  # Paris
                [2.45, 48.90, 100.0],  # Northeast
                [2.25, 48.80, 50.0],  # Southwest
            ],
            dtype=np.float64,
        )

        values = np.array([100.0, 150.0, 120.0], dtype=np.float64)

        tree.packing(coordinates, values)

        assert tree.size() == 3
        assert tree.spheroid is not None

    def test_packing_geodetic_float32(
        self, rtree_float32: RTree3DFloat32
    ) -> None:
        """Test packing geodetic (lon, lat, alt) coordinates with float32."""
        spheroid = Spheroid()
        tree = RTree3D(spheroid=spheroid, dtype="float32")

        coordinates = np.array(
            [
                [2.35, 48.85, 0.0],
                [2.45, 48.90, 100.0],
                [2.25, 48.80, 50.0],
            ],
            dtype=np.float32,
        )

        values = np.array([100.0, 150.0, 120.0], dtype=np.float32)

        tree.packing(coordinates, values)

        assert tree.size() == 3
        assert tree.spheroid is not None

    def test_packing_geodetic_parametrized(
        self,
        dtype_param: str,
        float_dtype: type[np.float64 | np.float32],
    ) -> None:
        """Test packing geodetic coordinates for both Float64 and Float32."""
        spheroid = Spheroid()
        tree = RTree3D(spheroid=spheroid, dtype=dtype_param)

        coordinates = np.array(
            [
                [2.35, 48.85, 0.0],
                [2.45, 48.90, 100.0],
                [2.25, 48.80, 50.0],
            ],
            dtype=float_dtype,
        )

        values = np.array([100.0, 150.0, 120.0], dtype=float_dtype)

        tree.packing(coordinates, values)

        assert tree.size() == 3
        assert tree.spheroid is not None


class TestRTree3DInsertion:
    """Tests for RTree3D insertion operations."""

    def test_insert_float64(self, rtree_float64: RTree3DFloat64) -> None:
        """Test incremental insertion with float64."""
        coords1 = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
        )
        values1 = np.array([10.0, 20.0], dtype=np.float64)
        rtree_float64.insert(coords1, values1)

        assert rtree_float64.size() == 2

        coords2 = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        values2 = np.array([30.0], dtype=np.float64)
        rtree_float64.insert(coords2, values2)

        assert rtree_float64.size() == 3

    def test_insert_float32(self, rtree_float32: RTree3DFloat32) -> None:
        """Test incremental insertion with float32."""
        coords1 = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        )
        values1 = np.array([10.0, 20.0], dtype=np.float32)
        rtree_float32.insert(coords1, values1)

        assert rtree_float32.size() == 2

        coords2 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        values2 = np.array([30.0], dtype=np.float32)
        rtree_float32.insert(coords2, values2)

        assert rtree_float32.size() == 3

    def test_insert_parametrized(
        self,
        dtype_param: str,
        float_dtype: type[np.float64 | np.float32],
    ) -> None:
        """Test insertion for both Float64 and Float32."""
        tree = RTree3D(dtype=dtype_param)

        coordinates = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float_dtype,
        )
        values = np.array([10.0, 20.0, 30.0], dtype=float_dtype)

        tree.insert(coordinates, values)

        assert tree.size() == 3


class TestRTree3DMaintenanceOperations:
    """Tests for RTree3D maintenance operations."""

    def test_clear_float64(self, rtree_float64: RTree3DFloat64) -> None:
        """Test clearing tree with float64."""
        coordinates = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64
        )
        values = np.array([10.0, 20.0], dtype=np.float64)

        rtree_float64.packing(coordinates, values)
        assert not rtree_float64.empty()
        assert rtree_float64.size() == 2

        rtree_float64.clear()
        assert rtree_float64.empty()
        assert rtree_float64.size() == 0

    def test_clear_float32(self, rtree_float32: RTree3DFloat32) -> None:
        """Test clearing tree with float32."""
        coordinates = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32
        )
        values = np.array([10.0, 20.0], dtype=np.float32)

        rtree_float32.packing(coordinates, values)
        assert not rtree_float32.empty()
        assert rtree_float32.size() == 2

        rtree_float32.clear()
        assert rtree_float32.empty()
        assert rtree_float32.size() == 0

    def test_clear_parametrized(
        self,
        dtype_param: str,
        float_dtype: type[np.float64 | np.float32],
    ) -> None:
        """Test clearing for both Float64 and Float32."""
        tree = RTree3D(dtype=dtype_param)

        coordinates = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float_dtype
        )
        values = np.array([10.0, 20.0], dtype=float_dtype)

        tree.packing(coordinates, values)
        assert tree.size() == 2

        tree.clear()
        assert tree.empty()

    def test_bounds_empty_float64(self, rtree_float64: RTree3DFloat64) -> None:
        """Test bounds on empty tree with float64."""
        assert rtree_float64.bounds() is None

    def test_bounds_empty_float32(self, rtree_float32: RTree3DFloat32) -> None:
        """Test bounds on empty tree with float32."""
        assert rtree_float32.bounds() is None

    def test_bounds_with_points_float64(
        self, rtree_float64: RTree3DFloat64
    ) -> None:
        """Test bounds calculation with float64."""
        coordinates = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [2.5, 3.5, 4.5]],
            dtype=np.float64,
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        rtree_float64.packing(coordinates, values)
        bounds = rtree_float64.bounds()

        assert bounds is not None
        min_point, max_point = bounds
        assert len(min_point) == 3
        assert len(max_point) == 3

    def test_bounds_with_points_float32(
        self, rtree_float32: RTree3DFloat32
    ) -> None:
        """Test bounds calculation with float32."""
        coordinates = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [2.5, 3.5, 4.5]],
            dtype=np.float32,
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float32)

        rtree_float32.packing(coordinates, values)
        bounds = rtree_float32.bounds()

        assert bounds is not None
        min_point, max_point = bounds
        assert len(min_point) == 3
        assert len(max_point) == 3

    def test_bounds_parametrized(
        self,
        dtype_param: str,
        float_dtype: type[np.float64 | np.float32],
    ) -> None:
        """Test bounds for both Float64 and Float32."""
        tree = RTree3D(dtype=dtype_param)

        coordinates = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float_dtype
        )
        values = np.array([10.0, 20.0], dtype=float_dtype)

        tree.packing(coordinates, values)
        bounds = tree.bounds()

        assert bounds is not None


class TestRTree3DPickling:
    """Tests for RTree3D serialization."""

    def test_pickle_float64(self, rtree_float64: RTree3DFloat64) -> None:
        """Test pickling and unpickling with float64."""
        coordinates = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            dtype=np.float64,
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        rtree_float64.packing(coordinates, values)

        tree_bytes = pickle.dumps(rtree_float64)
        tree2 = pickle.loads(tree_bytes)

        assert tree2.size() == 3
        assert not tree2.empty()

    def test_pickle_float32(self, rtree_float32: RTree3DFloat32) -> None:
        """Test pickling and unpickling with float32."""
        coordinates = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            dtype=np.float32,
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float32)

        rtree_float32.packing(coordinates, values)

        tree_bytes = pickle.dumps(rtree_float32)
        tree2 = pickle.loads(tree_bytes)

        assert tree2.size() == 3
        assert not tree2.empty()

    def test_pickle_parametrized(
        self,
        dtype_param: str,
        float_dtype: type[np.float64 | np.float32],
    ) -> None:
        """Test pickling for both Float64 and Float32."""
        tree = RTree3D(dtype=dtype_param)

        coordinates = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float_dtype
        )
        values = np.array([10.0, 20.0], dtype=float_dtype)

        tree.packing(coordinates, values)

        tree_copy = pickle.loads(pickle.dumps(tree))

        assert tree_copy.size() == 2

    def test_pickle_geodetic_float64(
        self, rtree_float64: RTree3DFloat64
    ) -> None:
        """Test pickling with geodetic spheroid float64."""
        spheroid = Spheroid()
        tree = RTree3D(spheroid=spheroid, dtype="float64")

        coordinates = np.array(
            [[2.35, 48.85, 0.0], [2.45, 48.90, 100.0]], dtype=np.float64
        )
        values = np.array([100.0, 150.0], dtype=np.float64)

        tree.packing(coordinates, values)

        tree_copy = pickle.loads(pickle.dumps(tree))

        assert tree_copy.size() == 2
        assert tree_copy.spheroid is not None


class TestRTree3DEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_point_float64(self, rtree_float64: RTree3DFloat64) -> None:
        """Test tree with a single point with float64."""
        coordinates = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        values = np.array([42.0], dtype=np.float64)

        rtree_float64.packing(coordinates, values)

        assert rtree_float64.size() == 1

    def test_single_point_float32(self, rtree_float32: RTree3DFloat32) -> None:
        """Test tree with a single point with float32."""
        coordinates = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        values = np.array([42.0], dtype=np.float32)

        rtree_float32.packing(coordinates, values)

        assert rtree_float32.size() == 1

    def test_duplicate_coordinates_float64(
        self, rtree_float64: RTree3DFloat64
    ) -> None:
        """Test inserting duplicate coordinates with float64."""
        coordinates = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=np.float64,
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        rtree_float64.packing(coordinates, values)

        assert rtree_float64.size() == 3

    def test_duplicate_coordinates_float32(
        self, rtree_float32: RTree3DFloat32
    ) -> None:
        """Test inserting duplicate coordinates with float32."""
        coordinates = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float32)

        rtree_float32.packing(coordinates, values)

        assert rtree_float32.size() == 3

    def test_large_values_float64(self, rtree_float64: RTree3DFloat64) -> None:
        """Test with large coordinate values (ECEF) with float64."""
        coordinates = np.array(
            [
                [6378137.0, 0.0, 0.0],
                [0.0, 6378137.0, 0.0],
                [0.0, 0.0, 6356752.0],
            ],
            dtype=np.float64,
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        rtree_float64.packing(coordinates, values)

        assert rtree_float64.size() == 3

    def test_large_values_float32(self, rtree_float32: RTree3DFloat32) -> None:
        """Test with large coordinate values (ECEF) with float32."""
        coordinates = np.array(
            [
                [6378137.0, 0.0, 0.0],
                [0.0, 6378137.0, 0.0],
                [0.0, 0.0, 6356752.0],
            ],
            dtype=np.float32,
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float32)

        rtree_float32.packing(coordinates, values)

        assert rtree_float32.size() == 3

    def test_small_values_float64(self, rtree_float64: RTree3DFloat64) -> None:
        """Test with very small coordinate values with float64."""
        coordinates = np.array(
            [
                [1e-10, 1e-10, 1e-10],
                [2e-10, 2e-10, 2e-10],
                [3e-10, 3e-10, 3e-10],
            ],
            dtype=np.float64,
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        rtree_float64.packing(coordinates, values)

        assert rtree_float64.size() == 3

    def test_small_values_float32(self, rtree_float32: RTree3DFloat32) -> None:
        """Test with very small coordinate values with float32."""
        coordinates = np.array(
            [
                [1e-10, 1e-10, 1e-10],
                [2e-10, 2e-10, 2e-10],
                [3e-10, 3e-10, 3e-10],
            ],
            dtype=np.float32,
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float32)

        rtree_float32.packing(coordinates, values)

        assert rtree_float32.size() == 3

    def test_negative_coordinates_float64(
        self, rtree_float64: RTree3DFloat64
    ) -> None:
        """Test with negative coordinates with float64."""
        coordinates = np.array(
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]],
            dtype=np.float64,
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        rtree_float64.packing(coordinates, values)

        assert rtree_float64.size() == 3

    def test_negative_coordinates_float32(
        self, rtree_float32: RTree3DFloat32
    ) -> None:
        """Test with negative coordinates with float32."""
        coordinates = np.array(
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]],
            dtype=np.float32,
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float32)

        rtree_float32.packing(coordinates, values)

        assert rtree_float32.size() == 3

    def test_mismatched_dimensions_float64(
        self, rtree_float64: RTree3DFloat64
    ) -> None:
        """Test error handling for mismatched dimensions with float64."""
        coordinates = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        with pytest.raises(
            ValueError,
            match="Number of coordinates must match number of values",
        ):
            rtree_float64.packing(coordinates, values)

    def test_mismatched_dimensions_float32(
        self, rtree_float32: RTree3DFloat32
    ) -> None:
        """Test error handling for mismatched dimensions with float32."""
        coordinates = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float32)

        with pytest.raises(
            ValueError,
            match="Number of coordinates must match number of values",
        ):
            rtree_float32.packing(coordinates, values)

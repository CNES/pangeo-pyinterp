# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for geographic RTree."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from .....core.config.rtree import (
    InterpolationWindow,
    InverseDistanceWeighting,
    Kriging,
    Query,
    RadialBasisFunction,
    WindowKernel,
)
from .....core.geometry.geographic import RTree


class TestRTreeInitialization:
    """Tests for RTree initialization."""

    def test_init_default(self) -> None:
        """Test RTree default initialization."""
        tree = RTree()
        assert tree.empty()
        assert tree.size() == 0

    def test_init_empty_state(self) -> None:
        """Test initial empty state."""
        tree = RTree()
        assert tree.empty()
        assert tree.size() == 0
        assert tree.bounds() is None


class TestRTreePacking:
    """Tests for RTree packing operations."""

    def test_packing_basic(self) -> None:
        """Test basic packing with geographic coordinates."""
        tree = RTree()
        coordinates = np.array(
            [
                [2.0, 48.0],  # Near Paris
                [3.0, 49.0],  # Northeast
                [1.0, 47.0],  # Southwest
            ],
            dtype=np.float64,
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        tree.packing(coordinates, values)

        assert not tree.empty()
        assert tree.size() == 3

    def test_packing_single_point(self) -> None:
        """Test packing with a single point."""
        tree = RTree()
        coordinates = np.array([[2.35, 48.85]], dtype=np.float64)
        values = np.array([42.0], dtype=np.float64)

        tree.packing(coordinates, values)

        assert tree.size() == 1
        assert not tree.empty()

    def test_packing_many_points(self) -> None:
        """Test packing with many points."""
        tree = RTree()
        n_points = 100
        # Create grid of points around Paris
        lons = np.linspace(2.0, 3.0, 10)
        lats = np.linspace(48.0, 49.0, 10)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        coordinates = np.column_stack(
            [lon_grid.ravel(), lat_grid.ravel()]
        ).astype(np.float64)
        values = np.arange(n_points, dtype=np.float64)

        tree.packing(coordinates, values)

        assert tree.size() == n_points

    def test_packing_replaces_old_data(self) -> None:
        """Test that packing replaces existing data."""
        tree = RTree()

        # First packing
        coords1 = np.array([[1.0, 45.0], [2.0, 46.0]], dtype=np.float64)
        values1 = np.array([10.0, 20.0], dtype=np.float64)
        tree.packing(coords1, values1)
        assert tree.size() == 2

        # Second packing should replace
        coords2 = np.array([[3.0, 47.0]], dtype=np.float64)
        values2 = np.array([30.0], dtype=np.float64)
        tree.packing(coords2, values2)
        assert tree.size() == 1

    def test_packing_mismatched_dimensions(self) -> None:
        """Test error handling for mismatched coordinate/value dimensions."""
        tree = RTree()
        coordinates = np.array([[1.0, 45.0], [2.0, 46.0]], dtype=np.float64)
        values = np.array([10.0], dtype=np.float64)  # Wrong size

        with pytest.raises(
            ValueError,
            match="Number of coordinates must match number of values",
        ):
            tree.packing(coordinates, values)


class TestRTreeInsertion:
    """Tests for RTree insertion operations."""

    def test_insert_basic(self) -> None:
        """Test basic insertion."""
        tree = RTree()
        coords = np.array([[2.0, 48.0], [3.0, 49.0]], dtype=np.float64)
        values = np.array([10.0, 20.0], dtype=np.float64)

        tree.insert(coords, values)

        assert tree.size() == 2
        assert not tree.empty()

    def test_insert_incremental(self) -> None:
        """Test incremental insertion."""
        tree = RTree()

        # First insertion
        coords1 = np.array([[1.0, 45.0]], dtype=np.float64)
        values1 = np.array([10.0], dtype=np.float64)
        tree.insert(coords1, values1)
        assert tree.size() == 1

        # Second insertion
        coords2 = np.array([[2.0, 46.0], [3.0, 47.0]], dtype=np.float64)
        values2 = np.array([20.0, 30.0], dtype=np.float64)
        tree.insert(coords2, values2)
        assert tree.size() == 3

    def test_insert_duplicate_coordinates(self) -> None:
        """Test inserting duplicate coordinates."""
        tree = RTree()
        coordinates = np.array(
            [[2.0, 48.0], [2.0, 48.0], [3.0, 49.0]], dtype=np.float64
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        tree.insert(coordinates, values)

        # Both duplicates should be inserted
        assert tree.size() == 3


class TestRTreeMaintenanceOperations:
    """Tests for RTree maintenance operations."""

    def test_clear(self) -> None:
        """Test clearing the tree."""
        tree = RTree()
        coordinates = np.array([[1.0, 45.0], [2.0, 46.0]], dtype=np.float64)
        values = np.array([10.0, 20.0], dtype=np.float64)

        tree.packing(coordinates, values)
        assert tree.size() == 2

        tree.clear()
        assert tree.empty()
        assert tree.size() == 0

    def test_bounds_empty(self) -> None:
        """Test bounds on empty tree."""
        tree = RTree()
        assert tree.bounds() is None

    def test_bounds_with_points(self) -> None:
        """Test bounds calculation."""
        tree = RTree()
        coordinates = np.array(
            [[1.0, 45.0], [3.0, 49.0], [2.0, 47.0]], dtype=np.float64
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        tree.packing(coordinates, values)
        bounds = tree.bounds()

        assert bounds is not None
        min_point, max_point = bounds

        # Check bounds dimensions
        assert len(min_point) == 2
        assert len(max_point) == 2

        # Check bounds values
        assert min_point[0] == pytest.approx(1.0)
        assert min_point[1] == pytest.approx(45.0)
        assert max_point[0] == pytest.approx(3.0)
        assert max_point[1] == pytest.approx(49.0)


class TestRTreeQuery:
    """Tests for RTree query operations."""

    def test_query_basic(self) -> None:
        """Test basic k-nearest neighbor query."""
        tree = RTree()
        coordinates = np.array(
            [[2.0, 48.0], [2.1, 48.1], [2.2, 48.2]], dtype=np.float64
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        tree.packing(coordinates, values)

        # Query at the first point
        query_coords = np.array([[2.0, 48.0]], dtype=np.float64)
        config = Query().with_k(2)

        distances, neighbor_values = tree.query(query_coords, config)

        assert distances.shape == (1, 2)
        assert neighbor_values.shape == (1, 2)

        # First neighbor should be the point itself (distance ~0)
        assert distances[0, 0] == pytest.approx(0.0, abs=1.0)
        assert neighbor_values[0, 0] == pytest.approx(10.0)

    def test_query_multiple_points(self) -> None:
        """Test querying multiple points at once."""
        tree = RTree()
        coordinates = np.array(
            [[2.0, 48.0], [2.5, 48.5], [3.0, 49.0]], dtype=np.float64
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        tree.packing(coordinates, values)

        # Query two points
        query_coords = np.array([[2.0, 48.0], [3.0, 49.0]], dtype=np.float64)
        config = Query().with_k(2)

        distances, neighbor_values = tree.query(query_coords, config)

        assert distances.shape == (2, 2)
        assert neighbor_values.shape == (2, 2)

    def test_query_with_radius(self) -> None:
        """Test query with radius limit."""
        tree = RTree()
        coordinates = np.array(
            [[2.0, 48.0], [2.1, 48.1], [10.0, 50.0]], dtype=np.float64
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        tree.packing(coordinates, values)

        # Query with small radius (meters)
        query_coords = np.array([[2.0, 48.0]], dtype=np.float64)
        config = Query().with_k(3).with_radius(20000.0)  # 20 km

        distances, _ = tree.query(query_coords, config)

        # Should only find nearby points
        assert distances.shape == (1, 3)
        # At least first two should be valid, third might be NaN
        assert not np.isnan(distances[0, 0])
        assert not np.isnan(distances[0, 1])

    def test_query_default_config(self) -> None:
        """Test query with default configuration."""
        tree = RTree()
        coordinates = np.array([[2.0, 48.0], [2.1, 48.1]], dtype=np.float64)
        values = np.array([10.0, 20.0], dtype=np.float64)
        tree.packing(coordinates, values)

        query_coords = np.array([[2.0, 48.0]], dtype=np.float64)

        # Should use default config (k=8)
        distances, neighbor_values = tree.query(query_coords, None)

        assert distances.shape[0] == 1
        assert neighbor_values.shape[0] == 1


class TestRTreeInterpolation:
    """Tests for RTree interpolation methods."""

    def test_inverse_distance_weighting(self) -> None:
        """Test IDW interpolation."""
        tree = RTree()
        coordinates = np.array(
            [[2.0, 48.0], [2.1, 48.0], [2.0, 48.1]], dtype=np.float64
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        tree.packing(coordinates, values)

        query_coords = np.array([[2.05, 48.05]], dtype=np.float64)
        config = InverseDistanceWeighting().with_k(3).with_p(2)

        interpolated, counts = tree.inverse_distance_weighting(
            query_coords, config
        )

        assert interpolated.shape == (1,)
        assert counts.shape == (1,)
        assert counts[0] == 3
        # Value should be between min and max
        assert 10.0 <= interpolated[0] <= 30.0

    def test_idw_default_config(self) -> None:
        """Test IDW with default configuration."""
        tree = RTree()
        coordinates = np.array([[2.0, 48.0], [2.1, 48.0]], dtype=np.float64)
        values = np.array([10.0, 20.0], dtype=np.float64)
        tree.packing(coordinates, values)

        query_coords = np.array([[2.05, 48.0]], dtype=np.float64)

        interpolated, counts = tree.inverse_distance_weighting(
            query_coords, None
        )

        assert interpolated.shape == (1,)
        assert counts[0] == 2

    def test_kriging(self) -> None:
        """Test Kriging interpolation."""
        tree = RTree()
        coordinates = np.array(
            [[2.0, 48.0], [2.1, 48.0], [2.0, 48.1]], dtype=np.float64
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        tree.packing(coordinates, values)

        query_coords = np.array([[2.05, 48.05]], dtype=np.float64)
        config = Kriging().with_k(3)

        interpolated, counts = tree.kriging(query_coords, config)

        assert interpolated.shape == (1,)
        assert counts.shape == (1,)
        assert counts[0] == 3

    def test_radial_basis_function(self) -> None:
        """Test RBF interpolation."""
        tree = RTree()
        coordinates = np.array(
            [[2.0, 48.0], [2.1, 48.0], [2.0, 48.1]], dtype=np.float64
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        tree.packing(coordinates, values)

        query_coords = np.array([[2.05, 48.05]], dtype=np.float64)
        config = RadialBasisFunction().with_k(3)

        interpolated, counts = tree.radial_basis_function(query_coords, config)

        assert interpolated.shape == (1,)
        assert counts.shape == (1,)
        assert counts[0] == 3

    def test_window_function(self) -> None:
        """Test window function interpolation."""
        tree = RTree()
        coordinates = np.array(
            [[2.0, 48.0], [2.1, 48.0], [2.0, 48.1]], dtype=np.float64
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        tree.packing(coordinates, values)

        query_coords = np.array([[2.05, 48.05]], dtype=np.float64)
        config = (
            InterpolationWindow()
            .with_k(3)
            .with_wf(WindowKernel.BOXCAR)
            .with_radius(100_000.0)
        )

        interpolated, counts = tree.window_function(query_coords, config)
        assert interpolated.shape == (1,)
        assert counts.shape == (1,)
        assert counts[0] == 3


class TestRTreePickling:
    """Tests for RTree serialization."""

    def test_pickle_empty(self) -> None:
        """Test pickling empty tree."""
        tree = RTree()
        tree_bytes = pickle.dumps(tree)
        tree2 = pickle.loads(tree_bytes)

        assert tree2.empty()
        assert tree2.size() == 0

    def test_pickle_with_data(self) -> None:
        """Test pickling tree with data."""
        tree = RTree()
        coordinates = np.array(
            [[2.0, 48.0], [2.5, 48.5], [3.0, 49.0]], dtype=np.float64
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        tree.packing(coordinates, values)

        tree_bytes = pickle.dumps(tree)
        tree2 = pickle.loads(tree_bytes)

        assert tree2.size() == 3
        assert not tree2.empty()

        # Test that queries work on unpickled tree
        query_coords = np.array([[2.5, 48.5]], dtype=np.float64)
        config = Query().with_k(2)
        distances, neighbor_values = tree2.query(query_coords, config)

        assert distances.shape == (1, 2)
        assert neighbor_values.shape == (1, 2)

    def test_pickle_preserves_bounds(self) -> None:
        """Test that pickling preserves bounds."""
        tree = RTree()
        coordinates = np.array([[1.0, 45.0], [3.0, 49.0]], dtype=np.float64)
        values = np.array([10.0, 20.0], dtype=np.float64)
        tree.packing(coordinates, values)

        bounds_original = tree.bounds()

        tree2 = pickle.loads(pickle.dumps(tree))
        bounds_restored = tree2.bounds()

        assert bounds_original is not None
        assert bounds_restored is not None

        min1, max1 = bounds_original
        min2, max2 = bounds_restored

        np.testing.assert_array_almost_equal(min1, min2)
        np.testing.assert_array_almost_equal(max1, max2)


class TestRTreeEdgeCases:
    """Test edge cases and special conditions."""

    def test_antipodal_points(self) -> None:
        """Test with points on opposite sides of the globe."""
        tree = RTree()
        coordinates = np.array(
            [[0.0, 0.0], [180.0, 0.0], [0.0, 90.0]], dtype=np.float64
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        tree.packing(coordinates, values)
        assert tree.size() == 3

    def test_prime_meridian_crossing(self) -> None:
        """Test points near the prime meridian."""
        tree = RTree()
        coordinates = np.array(
            [[-1.0, 45.0], [0.0, 45.0], [1.0, 45.0]], dtype=np.float64
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        tree.packing(coordinates, values)
        assert tree.size() == 3

    def test_high_latitude_points(self) -> None:
        """Test points near the poles."""
        tree = RTree()
        coordinates = np.array(
            [[0.0, 89.0], [90.0, 89.0], [180.0, 89.0]], dtype=np.float64
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        tree.packing(coordinates, values)
        assert tree.size() == 3

    def test_equator_points(self) -> None:
        """Test points along the equator."""
        tree = RTree()
        coordinates = np.array(
            [[0.0, 0.0], [90.0, 0.0], [180.0, 0.0], [-90.0, 0.0]],
            dtype=np.float64,
        )
        values = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)

        tree.packing(coordinates, values)
        assert tree.size() == 4

    def test_negative_longitude(self) -> None:
        """Test with negative longitude values."""
        tree = RTree()
        coordinates = np.array(
            [[-122.4, 37.8], [-118.2, 34.1]], dtype=np.float64
        )
        values = np.array([10.0, 20.0], dtype=np.float64)

        tree.packing(coordinates, values)
        assert tree.size() == 2

    def test_negative_latitude(self) -> None:
        """Test with negative latitude values (southern hemisphere)."""
        tree = RTree()
        coordinates = np.array(
            [[151.2, -33.9], [144.9, -37.8]], dtype=np.float64
        )
        values = np.array([10.0, 20.0], dtype=np.float64)

        tree.packing(coordinates, values)
        assert tree.size() == 2

    def test_very_close_points(self) -> None:
        """Test with points very close together."""
        tree = RTree()
        coordinates = np.array(
            [
                [2.0, 48.0],
                [2.0 + 1e-6, 48.0],
                [2.0, 48.0 + 1e-6],
            ],
            dtype=np.float64,
        )
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        tree.packing(coordinates, values)
        assert tree.size() == 3

        # Query should find all three as nearest neighbors
        query_coords = np.array([[2.0, 48.0]], dtype=np.float64)
        config = Query().with_k(3)
        distances, _ = tree.query(query_coords, config)

        assert distances.shape == (1, 3)
        # All distances should be very small
        assert distances[0, 0] < 1.0  # Less than 1 meter

    def test_large_dataset(self) -> None:
        """Test with a larger dataset."""
        tree = RTree()
        n_points = 1000

        # Create random points around the globe
        rng = np.random.default_rng(42)
        lons = rng.uniform(-180, 180, n_points)
        lats = rng.uniform(-90, 90, n_points)
        coordinates = np.column_stack([lons, lats]).astype(np.float64)
        values = rng.random(n_points).astype(np.float64)

        tree.packing(coordinates, values)
        assert tree.size() == n_points

        # Perform query
        query_coords = np.array([[0.0, 0.0]], dtype=np.float64)
        config = Query().with_k(10)
        distances, neighbor_values = tree.query(query_coords, config)

        assert distances.shape == (1, 10)
        assert neighbor_values.shape == (1, 10)
        # Distances should be sorted
        assert np.all(distances[0, :-1] <= distances[0, 1:])

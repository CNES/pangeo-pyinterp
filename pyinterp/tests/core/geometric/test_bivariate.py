# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for geometric bivariate interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from .... import core
from ....core.config import geometric
from ... import load_grid2d


class TestBivariateGeometric:
    """Test geometric bivariate interpolation."""

    @staticmethod
    def create_analytical_grid2d(
        dtype: type[np.float32 | np.float64],
    ) -> core.Grid2D:
        """Create a 2D grid with an analytical field.

        f(x, y) = sin(x) * cos(y)

        This provides a smooth, continuous field with known values.
        """
        x_vals = np.linspace(0, 2 * np.pi, 25)
        y_vals = np.linspace(0, np.pi, 20)

        x_axis = core.Axis(x_vals, period=None)
        y_axis = core.Axis(y_vals)

        x_grid, y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")

        # Create analytical field: f(x, y) = sin(x) * cos(y)
        data = (np.sin(x_grid) * np.cos(y_grid)).astype(dtype)
        # Ensure C-contiguous for grid creation
        data = np.ascontiguousarray(data)

        return core.Grid(x_axis, y_axis, data)

    def test_single_point_bilinear(self) -> None:
        """Perform bilinear interpolation at a single point.

        Test geometric bivariate interpolation at a single point with
        bilinear method.
        """
        grid = self.create_analytical_grid2d(np.float64)

        # Test point: (π/2, π/4)
        # Expected: sin(π/2) * cos(π/4) = 1 * √2/2
        x = np.array([np.pi / 2])
        y = np.array([np.pi / 4])

        expected = np.sin(np.pi / 2) * np.cos(np.pi / 4)

        config = geometric.Bivariate.bilinear()
        result = core.bivariate(grid, x, y, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])
        # Bilinear should have good accuracy for smooth functions
        np.testing.assert_allclose(result[0], expected, rtol=0.02)

    def test_multiple_points_bilinear(self) -> None:
        """Validate geometric bivariate interpolation at multiple points."""
        grid = self.create_analytical_grid2d(np.float64)

        # Multiple test points with known analytical values
        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])

        # Calculate expected values using the analytical function
        expected = np.array(
            [np.sin(x[i]) * np.cos(y[i]) for i in range(len(x))]
        )

        config = geometric.Bivariate.bilinear()
        result = core.bivariate(grid, x, y, config)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        # Validate against analytical values
        np.testing.assert_allclose(result, expected, rtol=0.05)

    def test_nearest_method(self) -> None:
        """Use nearest neighbor method for geometric bivariate interpolation.

        Test geometric bivariate interpolation with nearest neighbor method.
        """
        grid = self.create_analytical_grid2d(np.float64)

        x = np.array([1.5])
        y = np.array([1.2])

        config = geometric.Bivariate.nearest()
        result = core.bivariate(grid, x, y, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_idw_method(self) -> None:
        """Test geometric bivariate interpolation with IDW method."""
        grid = self.create_analytical_grid2d(np.float64)

        x = np.array([1.5])
        y = np.array([1.2])

        config = geometric.Bivariate.idw()
        result = core.bivariate(grid, x, y, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_bounds_error(self) -> None:
        """Test bounds_error parameter with geometric interpolation."""
        grid = self.create_analytical_grid2d(np.float64)

        # Point outside grid bounds
        x = np.array([3 * np.pi])
        y = np.array([0.5])

        # With bounds_error=False, should return NaN
        config = geometric.Bivariate.bilinear()
        result = core.bivariate(grid, x, y, config)

        assert result.shape == (1,)
        assert np.isnan(result[0])

        # With bounds_error=True, should raise an error
        config = geometric.Bivariate.bilinear().with_bounds_error(True)
        with pytest.raises(ValueError, match="out of bounds"):
            core.bivariate(grid, x, y, config)

    def test_with_real_data(self) -> None:
        """Test geometric bivariate interpolation with real grid data."""
        grid_data = load_grid2d()
        x_axis = core.Axis(grid_data.lon.values, period=360.0)
        y_axis = core.Axis(grid_data.lat.values)

        matrix = np.ascontiguousarray(grid_data.mss.values.T)
        grid = core.Grid(x_axis, y_axis, matrix)

        # Test points within bounds
        lon_vals = grid_data.lon.values
        lat_vals = grid_data.lat.values
        x = np.array(
            [
                lon_vals[0] + 5,
                (lon_vals[0] + lon_vals[-1]) / 2,
                lon_vals[-1] - 5,
            ]
        )
        y = np.array(
            [
                lat_vals[0] + 2,
                (lat_vals[0] + lat_vals[-1]) / 2,
                lat_vals[-1] - 2,
            ]
        )

        config = geometric.Bivariate.bilinear()
        result = core.bivariate(grid, x, y, config)

        assert result.shape == (3,)
        # At least some values should be finite (not all NaNs)
        assert np.any(np.isfinite(result))

    def test_dtype_float32(self) -> None:
        """Test geometric bivariate interpolation with float32 data."""
        grid = self.create_analytical_grid2d(np.float32)

        x = np.array([np.pi / 4])
        y = np.array([np.pi / 4])

        config = geometric.Bivariate.bilinear()
        result = core.bivariate(grid, x, y, config)

        assert result.dtype == np.float32
        assert np.isfinite(result[0])

    def test_num_threads(self) -> None:
        """Test geometric bivariate interpolation with varying thread counts.

        Validate performance with different thread configurations.
        """
        grid = self.create_analytical_grid2d(np.float64)

        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])

        # Test with 1 thread
        config_single = geometric.Bivariate.bilinear().with_num_threads(1)
        result_single = core.bivariate(grid, x, y, config_single)

        # Test with multiple threads
        config_multi = geometric.Bivariate.bilinear().with_num_threads(4)
        result_multi = core.bivariate(grid, x, y, config_multi)

        # Results should be identical or very close
        np.testing.assert_array_almost_equal(result_single, result_multi)

    def test_continuity(self) -> None:
        """Test that geometric interpolation is continuous (smooth)."""
        grid = self.create_analytical_grid2d(np.float64)

        # Create two close points
        x1 = np.array([1.0])
        y1 = np.array([1.0])

        x2 = np.array([1.001])
        y2 = np.array([1.001])

        config = geometric.Bivariate.bilinear()
        result1 = core.bivariate(grid, x1, y1, config)
        result2 = core.bivariate(grid, x2, y2, config)

        # Results should be close (continuity)
        assert np.abs(result1[0] - result2[0]) < 0.1

    def test_corner_point(self) -> None:
        """Test geometric interpolation at grid corner."""
        grid = self.create_analytical_grid2d(np.float64)

        # Get actual corner values from grid
        x_val = grid.x[0]
        y_val = grid.y[0]

        x = np.array([x_val])
        y = np.array([y_val])

        config = geometric.Bivariate.bilinear()
        result = core.bivariate(grid, x, y, config)

        expected = grid.array[0, 0]

        assert np.isfinite(result[0])
        assert np.abs(result[0] - expected) < 1e-10

    def test_analytical_accuracy(self) -> None:
        """Test geometric interpolation accuracy against analytical function."""
        grid = self.create_analytical_grid2d(np.float64)

        # Define the analytical function
        def analytical_func(x: float, y: float) -> float:
            return float(np.sin(x) * np.cos(y))

        # Test at interior points
        x = np.array([0.5, 1.0, 1.5, 2.0])
        y = np.array([0.5, 1.0, 1.5, 2.0])

        config = geometric.Bivariate.bilinear()
        result = core.bivariate(grid, x, y, config)

        # Compare with analytical values
        expected = np.array(
            [analytical_func(x[i], y[i]) for i in range(len(x))]
        )

        # Allow some tolerance for interpolation error
        np.testing.assert_allclose(result, expected, rtol=0.05)

    def test_large_array(self) -> None:
        """Test geometric bivariate interpolation with large arrays."""
        grid = self.create_analytical_grid2d(np.float64)

        # Create large arrays of points
        n_points = 1000
        rng = np.random.Generator(np.random.PCG64(seed=42))
        x = rng.uniform(0.1, 2 * np.pi - 0.1, n_points)
        y = rng.uniform(0.1, np.pi - 0.1, n_points)

        config = geometric.Bivariate.bilinear()
        result = core.bivariate(grid, x, y, config)

        assert result.shape == (n_points,)
        # Most values should be finite
        assert np.sum(np.isfinite(result)) > n_points * 0.99

    def test_method_chaining(self) -> None:
        """Test that geometric methods can be chained."""
        config = (
            geometric.Bivariate.bilinear()
            .with_num_threads(4)
            .with_bounds_error(True)
        )

        assert isinstance(config, geometric.Bivariate)

    def test_all_interpolation_methods(self) -> None:
        """Test all available geometric bivariate methods."""
        grid = self.create_analytical_grid2d(np.float64)

        methods = ["nearest", "idw", "bilinear"]

        x = np.array([np.pi / 2])
        y = np.array([np.pi / 2])

        for method in methods:
            config = getattr(geometric.Bivariate, method)()
            result = core.bivariate(grid, x, y, config)

            assert result.shape == (1,)
            assert np.isfinite(result[0]), f"Method {method} produced NaN"

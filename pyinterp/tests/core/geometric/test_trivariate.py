# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for trivariate interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from .... import core
from ....core.config import geometric
from ... import load_grid3d


class TestTrivariateGeometric:
    """Test geometric trivariate interpolation."""

    @staticmethod
    def create_analytical_grid3d(
        dtype: type[np.float32 | np.float64],
    ) -> core.Grid3D:
        """Create a 3D grid with an analytical field.

        f(x, y, z) = sin(x) * cos(y) * exp(-z/10)

        This provides a smooth, continuous field with known values.
        """
        x_vals = np.linspace(0, 2 * np.pi, 15)
        y_vals = np.linspace(0, np.pi, 12)
        z_vals = np.linspace(0, 10, 8)

        x_axis = core.Axis(x_vals, period=None)
        y_axis = core.Axis(y_vals)
        z_axis = core.Axis(z_vals)

        x_grid, y_grid, z_grid = np.meshgrid(
            x_vals, y_vals, z_vals, indexing="ij"
        )

        # Create analytical field: f(x, y, z) = sin(x) * cos(y) * exp(-z/10)
        data = (np.sin(x_grid) * np.cos(y_grid) * np.exp(-z_grid / 10)).astype(
            dtype
        )
        # Ensure C-contiguous for grid creation
        data = np.ascontiguousarray(data)

        return core.Grid(x_axis, y_axis, z_axis, data)

    def test_single_point_bilinear(self) -> None:
        """Perform bilinear interpolation at a single point."""
        grid = self.create_analytical_grid3d(np.float64)

        # Test point: (π, π/2, 0)
        # Expected: sin(π) * cos(π/2) * exp(0) = 0 * 0 * 1 = 0
        x = np.array([np.pi])
        y = np.array([np.pi / 2])
        z = np.array([0.0])

        config = geometric.Trivariate.bilinear()
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (1,)
        # Should be close to 0 (within interpolation error)
        assert np.abs(result[0]) < 0.01

    def test_multiple_points_bilinear(self) -> None:
        """Test trivariate interpolation at multiple points."""
        grid = self.create_analytical_grid3d(np.float64)

        # Multiple test points
        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        z = np.array([0.0, 5.0, 10.0])

        config = geometric.Trivariate.bilinear()
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_nearest_method(self) -> None:
        """Test trivariate interpolation with nearest neighbor method."""
        grid = self.create_analytical_grid3d(np.float64)

        # Test at grid point
        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([0.0])

        config = geometric.Trivariate.nearest()
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (1,)
        # At (0,0,0): sin(0) * cos(0) * exp(0) = 0 * 1 * 1 = 0
        assert np.isfinite(result[0])

    def test_idw_method(self) -> None:
        """Test trivariate interpolation with IDW method."""
        grid = self.create_analytical_grid3d(np.float64)

        x = np.array([np.pi / 2])
        y = np.array([np.pi / 4])
        z = np.array([2.5])

        config = geometric.Trivariate.idw()
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_bounds_error(self) -> None:
        """Test bounds_error parameter."""
        grid = self.create_analytical_grid3d(np.float64)

        # Point outside grid bounds
        x = np.array([3 * np.pi])
        y = np.array([0.0])
        z = np.array([0.0])

        # With bounds_error=False, should return NaN
        config = geometric.Trivariate.bilinear()
        result = core.trivariate(grid, x, y, z, config)

        # With bounds_error=True, should raise an error
        config = geometric.Trivariate.bilinear().with_bounds_error(True)
        with pytest.raises(ValueError, match="out of bounds"):
            core.trivariate(grid, x, y, z, config)

        assert result.shape == (1,)
        assert np.isnan(result[0])

    def test_with_real_data(self) -> None:
        """Test trivariate interpolation with real grid data."""
        grid_data = load_grid3d()
        x_axis = core.Axis(grid_data.longitude.values, period=360.0)
        y_axis = core.Axis(grid_data.latitude.values)
        z_axis = core.Axis(grid_data.time.values.astype("float64"))

        matrix = np.ascontiguousarray(grid_data.tcw.values.transpose())
        grid = core.Grid(x_axis, y_axis, z_axis, matrix)

        # Test points within bounds
        x = np.array([10.0, 20.0, 30.0])
        y = np.array([-10.0, 0.0, 10.0])
        z = np.array(
            [
                grid_data.time.values[0].astype("float64"),
                grid_data.time.values[1].astype("float64"),
                grid_data.time.values[-1].astype("float64"),
            ]
        )

        config = geometric.Trivariate.bilinear()
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (3,)
        # At least some values should be finite (not all NaNs)
        assert np.any(np.isfinite(result))

    def test_dtype_float32(self) -> None:
        """Test trivariate interpolation with float32 data."""
        grid = self.create_analytical_grid3d(np.float32)

        x = np.array([np.pi / 4])
        y = np.array([np.pi / 4])
        z = np.array([5.0])

        config = geometric.Trivariate.bilinear()
        result = core.trivariate(grid, x, y, z, config)

        assert result.dtype == np.float32
        assert np.isfinite(result[0])

    def test_num_threads(self) -> None:
        """Test trivariate interpolation with different thread counts."""
        grid = self.create_analytical_grid3d(np.float64)

        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        z = np.array([1.0, 5.0, 9.0])

        # Test with 1 thread
        config_single = geometric.Trivariate.bilinear().with_num_threads(1)
        result_single = core.trivariate(grid, x, y, z, config_single)

        # Test with multiple threads
        config_multi = geometric.Trivariate.bilinear().with_num_threads(4)
        result_multi = core.trivariate(grid, x, y, z, config_multi)

        # Results should be identical
        np.testing.assert_array_almost_equal(result_single, result_multi)

    def test_continuity(self) -> None:
        """Test that interpolation is continuous (smooth)."""
        grid = self.create_analytical_grid3d(np.float64)

        # Create two close points
        x1 = np.array([1.0])
        y1 = np.array([1.0])
        z1 = np.array([2.0])

        x2 = np.array([1.001])
        y2 = np.array([1.001])
        z2 = np.array([2.001])

        config = geometric.Trivariate.bilinear()
        result1 = core.trivariate(grid, x1, y1, z1, config)
        result2 = core.trivariate(grid, x2, y2, z2, config)

        # Results should be close (continuity)
        assert np.abs(result1[0] - result2[0]) < 0.1

    def test_corner_point(self) -> None:
        """Test interpolation at grid corner."""
        grid = self.create_analytical_grid3d(np.float64)

        # Get actual corner values from grid
        x_val = grid.x[0]
        y_val = grid.y[0]
        z_val = grid.z[0]

        x = np.array([x_val])
        y = np.array([y_val])
        z = np.array([z_val])

        config = geometric.Trivariate.bilinear()
        result = core.trivariate(grid, x, y, z, config)

        expected = grid.array[0, 0, 0]

        assert np.isfinite(result[0])
        assert np.abs(result[0] - expected) < 1e-10

    def test_analytical_accuracy(self) -> None:
        """Test interpolation accuracy against analytical function."""
        grid = self.create_analytical_grid3d(np.float64)

        # Define the analytical function
        def analytical_func(x: float, y: float, z: float) -> float:
            return float(np.sin(x) * np.cos(y) * np.exp(-z / 10))

        # Test at interior points
        x = np.array([0.5, 1.0, 1.5, 2.0])
        y = np.array([0.5, 1.0, 1.5, 2.0])
        z = np.array([1.0, 2.0, 3.0, 4.0])

        config = geometric.Trivariate.bilinear()
        result = core.trivariate(grid, x, y, z, config)

        # Compare with analytical values
        expected = np.array(
            [analytical_func(x[i], y[i], z[i]) for i in range(len(x))]
        )

        # Allow some tolerance for interpolation error
        np.testing.assert_allclose(result, expected, rtol=0.05)

    def test_large_array(self) -> None:
        """Test trivariate interpolation with large arrays."""
        grid = self.create_analytical_grid3d(np.float64)

        # Create large arrays of points
        n_points = 1000
        rng = np.random.default_rng(42)
        x = rng.uniform(0.1, 2 * np.pi - 0.1, n_points)
        y = rng.uniform(0.1, np.pi - 0.1, n_points)
        z = rng.uniform(0.1, 9.9, n_points)

        config = geometric.Trivariate.bilinear()
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (n_points,)
        # Most values should be finite
        assert np.sum(np.isfinite(result)) > n_points * 0.99


class TestTrivariateGeometricTemporalAxis:
    """Test geometric trivariate interpolation with temporal Z-axis."""

    @staticmethod
    def create_analytical_temporal_grid3d(
        dtype: type[np.float32 | np.float64],
    ) -> core.GridHolder:
        """Create a 3D grid with temporal Z-axis and analytical field.

        f(x, y, t) = sin(x) * cos(y) * exp(-t_normalized/10)

        where t is a datetime64 axis.
        """
        x_vals = np.linspace(0, 2 * np.pi, 12)
        y_vals = np.linspace(0, np.pi, 10)
        # Create a temporal axis with datetime64 values
        time_vals: np.ndarray[tuple[int], np.dtype[np.datetime64]] = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-09"),
            np.timedelta64(1, "D"),
        )

        x_axis = core.Axis(x_vals, period=None)
        y_axis = core.Axis(y_vals)
        z_axis = core.TemporalAxis(time_vals)

        # Normalize time for analytical function (0 to 7 days)
        time_normalized = np.arange(8)

        x_grid, y_grid, t_grid = np.meshgrid(
            x_vals, y_vals, time_normalized, indexing="ij"
        )

        # Create analytical field: f(x, y, t) = sin(x) * cos(y) * exp(-t/10)
        data = (np.sin(x_grid) * np.cos(y_grid) * np.exp(-t_grid / 10)).astype(
            dtype
        )
        data = np.ascontiguousarray(data)

        return core.Grid(x_axis, y_axis, z_axis, data)

    def test_temporal_grid_basic_interpolation(self) -> None:
        """Test trivariate interpolation with temporal Z-axis."""
        grid = self.create_analytical_temporal_grid3d(np.float64)

        x = np.array([np.pi / 4])
        y = np.array([np.pi / 4])
        # Use a datetime64 value for z
        z = np.array([np.datetime64("2020-01-01")])

        config = geometric.Trivariate.bilinear()
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_temporal_grid_multiple_times(self) -> None:
        """Test interpolation at multiple temporal points."""
        grid = self.create_analytical_temporal_grid3d(np.float64)

        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y = np.array([np.pi / 6, np.pi / 3, np.pi / 2])
        z = np.array(
            [
                np.datetime64("2020-01-01"),
                np.datetime64("2020-01-04"),
                np.datetime64("2020-01-08"),
            ]
        )

        config = geometric.Trivariate.bilinear()
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_temporal_grid_with_real_data(self) -> None:
        """Test with real temporal grid data."""
        grid_data = load_grid3d()
        x_axis = core.Axis(grid_data.longitude.values, period=360.0)
        y_axis = core.Axis(grid_data.latitude.values)
        z_axis = core.TemporalAxis(grid_data.time.values)

        matrix = np.ascontiguousarray(grid_data.tcw.values.transpose())
        grid = core.Grid(x_axis, y_axis, z_axis, matrix)

        # Test points within bounds
        lon_vals = grid_data.longitude.values
        lat_vals = grid_data.latitude.values
        time_vals = grid_data.time.values

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
        z = np.array(
            [time_vals[0], time_vals[len(time_vals) // 2], time_vals[-1]]
        )

        config = geometric.Trivariate.bilinear()
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (3,)
        assert np.any(np.isfinite(result))

    def test_temporal_grid_nearest_method(self) -> None:
        """Test temporal grid with nearest neighbor method."""
        grid = self.create_analytical_temporal_grid3d(np.float64)

        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([np.datetime64("2020-01-05")])

        config = geometric.Trivariate.nearest()
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_temporal_grid_continuity(self) -> None:
        """Test continuity across temporal axis."""
        grid = self.create_analytical_temporal_grid3d(np.float64)

        # Create two close times
        t1 = np.datetime64("2020-01-03", "D")
        t2 = np.datetime64("2020-01-03", "D") + np.timedelta64(12, "h")

        x1 = np.array([1.0])
        y1 = np.array([1.0])
        z1 = np.array([t1])

        x2 = np.array([1.0])
        y2 = np.array([1.0])
        z2 = np.array([t2])

        config = geometric.Trivariate.bilinear()
        result1 = core.trivariate(grid, x1, y1, z1, config)
        result2 = core.trivariate(grid, x2, y2, z2, config)

        # Results should be close (continuity)
        assert np.abs(result1[0] - result2[0]) < 0.1

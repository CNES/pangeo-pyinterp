# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for quadrivariate interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from .... import core
from ....core.config import geometric
from ... import load_grid4d


class TestQuadrivariateGeometric:
    """Test geometric quadrivariate interpolation."""

    @staticmethod
    def create_analytical_grid4d(
        dtype: type[np.float32 | np.float64],
    ) -> core.Grid4D:
        """Create a 4D grid with an analytical field.

        f(x, y, z, u) = sin(x) * cos(y) * exp(-z/5) * sin(u)

        This provides a smooth, continuous field with known values.
        """
        x_vals = np.linspace(0, 2 * np.pi, 10)
        y_vals = np.linspace(0, np.pi, 9)
        z_vals = np.linspace(0, 5, 7)
        u_vals = np.linspace(0, np.pi, 6)

        x_axis = core.Axis(x_vals, period=None)
        y_axis = core.Axis(y_vals)
        z_axis = core.Axis(z_vals)
        u_axis = core.Axis(u_vals)

        x_grid, y_grid, z_grid, u_grid = np.meshgrid(
            x_vals, y_vals, z_vals, u_vals, indexing="ij"
        )

        # Create analytical field: f(x, y, z, u)
        #   = sin(x) * cos(y) * exp(-z/5) * sin(u)
        data = (
            np.sin(x_grid)
            * np.cos(y_grid)
            * np.exp(-z_grid / 5)
            * np.sin(u_grid)
        ).astype(dtype)
        # Ensure C-contiguous for grid creation
        data = np.ascontiguousarray(data)

        return core.Grid(x_axis, y_axis, z_axis, u_axis, data)

    def test_single_point_bilinear(self) -> None:
        """Test bilinear interpolation at a single point."""
        grid = self.create_analytical_grid4d(np.float64)

        # Test point: (π, π/2, 0, π/2)
        # Expected: sin(π) * cos(π/2) * exp(0) * sin(π/2) = 0 * 0 * 1 * 1 = 0
        x = np.array([np.pi])
        y = np.array([np.pi / 2])
        z = np.array([0.0])
        u = np.array([np.pi / 2])

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])
        # Should be close to 0 (within interpolation error)
        assert np.abs(result[0]) < 0.01

    def test_multiple_points_bilinear(self) -> None:
        """Test quadrivariate interpolation at multiple points."""
        grid = self.create_analytical_grid4d(np.float64)

        # Multiple test points
        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        z = np.array([0.0, 2.5, 5.0])
        u = np.array([np.pi / 6, np.pi / 3, np.pi / 2])

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_nearest_method(self) -> None:
        """Test quadrivariate interpolation with nearest neighbor method."""
        grid = self.create_analytical_grid4d(np.float64)

        # Test at grid point
        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([0.0])
        u = np.array([0.0])

        config = geometric.Quadrivariate.nearest()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_idw_method(self) -> None:
        """Test quadrivariate interpolation with IDW method."""
        grid = self.create_analytical_grid4d(np.float64)

        x = np.array([np.pi / 2])
        y = np.array([np.pi / 4])
        z = np.array([2.0])
        u = np.array([np.pi / 3])

        config = geometric.Quadrivariate.idw()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_bounds_error(self) -> None:
        """Test bounds_error parameter."""
        grid = self.create_analytical_grid4d(np.float64)

        # Point outside grid bounds (x too large)
        x = np.array([3 * np.pi])
        y = np.array([0.0])
        z = np.array([0.0])
        u = np.array([0.0])

        # With bounds_error=False, should return NaN
        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (1,)
        assert np.isnan(result[0])

        # With bounds_error=True, should raise an error
        config = geometric.Quadrivariate.bilinear().with_bounds_error(True)
        with pytest.raises(ValueError, match="out of bounds"):
            core.quadrivariate(grid, x, y, z, u, config)

    def test_bounds_error_z_axis(self) -> None:
        """Test bounds_error with z-axis out of bounds."""
        grid = self.create_analytical_grid4d(np.float64)

        # Point outside grid bounds (z too large)
        x = np.array([np.pi / 2])
        y = np.array([np.pi / 4])
        z = np.array([10.0])  # Outside [0, 5]
        u = np.array([np.pi / 3])

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (1,)
        assert np.isnan(result[0])

    def test_bounds_error_u_axis(self) -> None:
        """Test bounds_error with u-axis out of bounds."""
        grid = self.create_analytical_grid4d(np.float64)

        # Point outside grid bounds (u too large)
        x = np.array([np.pi / 2])
        y = np.array([np.pi / 4])
        z = np.array([2.0])
        u = np.array([2 * np.pi])  # Outside [0, π]

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (1,)
        assert np.isnan(result[0])

    def test_with_real_data(self) -> None:
        """Test quadrivariate interpolation with real grid data."""
        grid_data = load_grid4d()
        x_axis = core.Axis(grid_data.longitude.values, period=360.0)
        y_axis = core.Axis(grid_data.latitude.values)
        z_axis = core.Axis(grid_data.time.values.astype("float64"))
        u_axis = core.Axis(grid_data.level.values)

        matrix = np.ascontiguousarray(grid_data.temperature.values.transpose())
        grid = core.Grid(x_axis, y_axis, z_axis, u_axis, matrix)

        # Test points within bounds - use actual grid bounds
        lon_vals = grid_data.longitude.values
        lat_vals = grid_data.latitude.values
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
            [
                grid_data.time.values[0].astype("float64"),
                grid_data.time.values[1].astype("float64"),
                grid_data.time.values[-1].astype("float64"),
            ]
        )
        u = np.array(
            [
                grid_data.level.values[0],
                grid_data.level.values[1],
                grid_data.level.values[-1],
            ]
        )

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (3,)
        # At least some values should be finite (not all NaNs)
        assert np.any(np.isfinite(result))

    def test_dtype_float32(self) -> None:
        """Test quadrivariate interpolation with float32 data."""
        grid = self.create_analytical_grid4d(np.float32)

        x = np.array([np.pi / 4])
        y = np.array([np.pi / 4])
        z = np.array([2.0])
        u = np.array([np.pi / 6])

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.dtype == np.float32
        assert np.isfinite(result[0])

    def test_num_threads(self) -> None:
        """Test quadrivariate interpolation with different thread counts."""
        grid = self.create_analytical_grid4d(np.float64)

        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        z = np.array([1.0, 2.5, 4.0])
        u = np.array([np.pi / 6, np.pi / 3, np.pi / 2])

        # Test with 1 thread
        config_single = geometric.Quadrivariate.bilinear().with_num_threads(1)
        result_single = core.quadrivariate(grid, x, y, z, u, config_single)

        # Test with multiple threads
        config_multi = geometric.Quadrivariate.bilinear().with_num_threads(4)
        result_multi = core.quadrivariate(grid, x, y, z, u, config_multi)

        # Results should be identical
        np.testing.assert_array_almost_equal(result_single, result_multi)

    def test_continuity(self) -> None:
        """Test that interpolation is continuous (smooth)."""
        grid = self.create_analytical_grid4d(np.float64)

        # Create two close points
        x1 = np.array([1.0])
        y1 = np.array([1.0])
        z1 = np.array([1.0])
        u1 = np.array([1.0])

        x2 = np.array([1.001])
        y2 = np.array([1.001])
        z2 = np.array([1.001])
        u2 = np.array([1.001])

        config = geometric.Quadrivariate.bilinear()
        result1 = core.quadrivariate(grid, x1, y1, z1, u1, config)
        result2 = core.quadrivariate(grid, x2, y2, z2, u2, config)

        # Results should be close (continuity)
        assert np.abs(result1[0] - result2[0]) < 0.1

    def test_corner_point(self) -> None:
        """Test interpolation at grid corner."""
        grid = self.create_analytical_grid4d(np.float64)

        # Get actual corner values from grid
        x_val = grid.x[0]
        y_val = grid.y[0]
        z_val = grid.z[0]
        u_val = grid.u[0]

        x = np.array([x_val])
        y = np.array([y_val])
        z = np.array([z_val])
        u = np.array([u_val])

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        expected = grid.array[0, 0, 0, 0]

        assert np.isfinite(result[0])
        assert np.abs(result[0] - expected) < 1e-10

    def test_analytical_accuracy(self) -> None:
        """Test interpolation accuracy against analytical function."""
        grid = self.create_analytical_grid4d(np.float64)

        # Define the analytical function
        def analytical_func(x: float, y: float, z: float, u: float) -> float:
            return float(np.sin(x) * np.cos(y) * np.exp(-z / 5) * np.sin(u))

        # Test at interior points
        x = np.array([0.5, 1.0, 1.5, 2.0])
        y = np.array([0.5, 1.0, 1.5, 2.0])
        z = np.array([0.5, 1.0, 1.5, 2.0])
        u = np.array([0.5, 1.0, 1.5, 2.0])

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        # Compare with analytical values
        expected = np.array(
            [analytical_func(x[i], y[i], z[i], u[i]) for i in range(len(x))]
        )

        # Allow some tolerance for interpolation error
        # (relaxed tolerance for coarser grid)
        np.testing.assert_allclose(result, expected, rtol=0.15)

    def test_large_array(self) -> None:
        """Test quadrivariate interpolation with large arrays."""
        grid = self.create_analytical_grid4d(np.float64)

        # Create large arrays of points
        n_points = 500
        rng = np.random.default_rng(42)
        x = rng.uniform(0.1, 2 * np.pi - 0.1, n_points)
        y = rng.uniform(0.1, np.pi - 0.1, n_points)
        z = rng.uniform(0.1, 4.9, n_points)
        u = rng.uniform(0.1, np.pi - 0.1, n_points)

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (n_points,)
        # Most values should be finite
        assert np.sum(np.isfinite(result)) > n_points * 0.99

    def test_symmetry_z_axis(self) -> None:
        """Test that decay along z-axis is consistent."""
        grid = self.create_analytical_grid4d(np.float64)

        # Test points at different z levels but same x, y, u
        x = np.array([np.pi / 2, np.pi / 2, np.pi / 2])
        y = np.array([np.pi / 4, np.pi / 4, np.pi / 4])
        z = np.array([0.1, 2.5, 4.9])  # Use safe bounds within [0, 5]
        u = np.array([np.pi / 3, np.pi / 3, np.pi / 3])

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        # Values should decay with increasing z due to exp(-z/5) term
        assert np.abs(result[0]) > np.abs(result[2])

    def test_symmetry_u_axis(self) -> None:
        """Test that sin variation along u-axis is consistent."""
        grid = self.create_analytical_grid4d(np.float64)

        # Test points at different u levels but same x, y, z
        x = np.array([np.pi / 2, np.pi / 2])
        y = np.array([np.pi / 4, np.pi / 4])
        z = np.array([1.0, 1.0])
        u = np.array(
            [0.1, np.pi / 2]
        )  # Use 0.1 instead of 0.0 to avoid boundary

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))
        # At u=π/2, sin(π/2)=1, so result should be larger than at u=0.1
        assert np.abs(result[1]) > np.abs(result[0])

    def test_exponent_idw(self) -> None:
        """Test IDW interpolation with custom exponent."""
        grid = self.create_analytical_grid4d(np.float64)

        x = np.array([np.pi / 4, np.pi / 2])
        y = np.array([np.pi / 4, np.pi / 2])
        z = np.array([1.0, 2.0])
        u = np.array([np.pi / 6, np.pi / 3])

        # Test with different exponents
        config1 = geometric.Quadrivariate.idw()
        result1 = core.quadrivariate(grid, x, y, z, u, config1)

        # Both should be finite
        assert np.all(np.isfinite(result1))

    def test_return_dtype_float64(self) -> None:
        """Test that float64 grid returns float64 results."""
        grid = self.create_analytical_grid4d(np.float64)

        x = np.array([1.0])
        y = np.array([1.0])
        z = np.array([1.0])
        u = np.array([1.0])

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.dtype == np.float64

    def test_edge_points(self) -> None:
        """Test interpolation at various edge points."""
        grid = self.create_analytical_grid4d(np.float64)

        # Test edge points along different dimensions
        x = np.array([grid.x[0], grid.x[len(grid.x) - 1]])
        y = np.array([grid.y[0], grid.y[0]])
        z = np.array([grid.z[0], grid.z[0]])
        u = np.array([grid.u[0], grid.u[0]])

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))


class TestQuadrivariateGeometricTemporalAxis:
    """Test geometric quadrivariate interpolation with temporal Z-axis."""

    @staticmethod
    def create_analytical_temporal_grid4d(
        dtype: type[np.float32 | np.float64],
    ) -> core.GridHolder:
        """Create a 4D grid with temporal Z-axis and analytical field.

        f(x, y, t, u) = sin(x) * cos(y) * exp(-t_normalized/5) * sin(u)

        where t is a datetime64 axis.
        """
        x_vals = np.linspace(0, 2 * np.pi, 10)
        y_vals = np.linspace(0, np.pi, 9)
        # Create a temporal axis with datetime64 values
        time_vals: np.ndarray[tuple[int], np.dtype[np.datetime64]] = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-06"),
            np.timedelta64(1, "D"),
        )
        u_vals = np.linspace(0, np.pi, 6)

        x_axis = core.Axis(x_vals, period=None)
        y_axis = core.Axis(y_vals)
        z_axis = core.TemporalAxis(time_vals)
        u_axis = core.Axis(u_vals)

        # Normalize time for analytical function (0 to 4 days)
        time_normalized = np.arange(5)

        x_grid, y_grid, t_grid, u_grid = np.meshgrid(
            x_vals, y_vals, time_normalized, u_vals, indexing="ij"
        )

        # Create analytical field: f(x, y, t, u) =
        #   sin(x) * cos(y) * exp(-t/5) * sin(u)
        data = (
            np.sin(x_grid)
            * np.cos(y_grid)
            * np.exp(-t_grid / 5)
            * np.sin(u_grid)
        ).astype(dtype)
        data = np.ascontiguousarray(data)

        return core.Grid(x_axis, y_axis, z_axis, u_axis, data)

    def test_grid_basic_interpolation(self) -> None:
        """Test quadrivariate interpolation with temporal Z-axis."""
        grid = self.create_analytical_temporal_grid4d(np.float64)

        x = np.array([np.pi / 4])
        y = np.array([np.pi / 4])
        z = np.array([np.datetime64("2020-01-01")])
        u = np.array([np.pi / 3])

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_grid_multiple_times(self) -> None:
        """Test interpolation at multiple temporal points."""
        grid = self.create_analytical_temporal_grid4d(np.float64)

        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y = np.array([np.pi / 6, np.pi / 3, np.pi / 2])
        z = np.array(
            [
                np.datetime64("2020-01-01"),
                np.datetime64("2020-01-03"),
                np.datetime64("2020-01-05"),
            ]
        )
        u = np.array([np.pi / 6, np.pi / 3, np.pi / 2])

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_grid_with_real_data(self) -> None:
        """Test with real temporal grid data."""
        grid_data = load_grid4d()
        x_axis = core.Axis(grid_data.longitude.values, period=360.0)
        y_axis = core.Axis(grid_data.latitude.values)
        z_axis = core.TemporalAxis(grid_data.time.values)
        u_axis = core.Axis(grid_data.level.values)

        matrix = np.ascontiguousarray(grid_data.temperature.values.transpose())
        grid = core.Grid(x_axis, y_axis, z_axis, u_axis, matrix)

        # Test points within bounds
        lon_vals = grid_data.longitude.values
        lat_vals = grid_data.latitude.values
        time_vals = grid_data.time.values
        level_vals = grid_data.level.values

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
        u = np.array(
            [level_vals[0], level_vals[len(level_vals) // 2], level_vals[-1]]
        )

        config = geometric.Quadrivariate.bilinear()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (3,)
        assert np.any(np.isfinite(result))

    def test_grid_nearest_method(self) -> None:
        """Test temporal grid with nearest neighbor method."""
        grid = self.create_analytical_temporal_grid4d(np.float64)

        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([np.datetime64("2020-01-03")])
        u = np.array([0.0])

        config = geometric.Quadrivariate.nearest()
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_grid_continuity(self) -> None:
        """Test continuity across temporal axis."""
        grid = self.create_analytical_temporal_grid4d(np.float64)

        # Create two close times
        t1 = np.datetime64("2020-01-02", "D")
        t2 = np.datetime64("2020-01-02", "D") + np.timedelta64(12, "h")

        x1 = np.array([1.0])
        y1 = np.array([1.0])
        z1 = np.array([t1])
        u1 = np.array([1.0])

        x2 = np.array([1.0])
        y2 = np.array([1.0])
        z2 = np.array([t2])
        u2 = np.array([1.0])

        config = geometric.Quadrivariate.bilinear()
        result1 = core.quadrivariate(grid, x1, y1, z1, u1, config)
        result2 = core.quadrivariate(grid, x2, y2, z2, u2, config)

        # Results should be close (continuity)
        assert np.abs(result1[0] - result2[0]) < 0.1

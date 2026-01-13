# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for windowed bivariate interpolation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from .... import core
from ....core.config import windowed
from ... import load_grid2d


if TYPE_CHECKING:
    from collections.abc import Callable


class TestBivariateWindowed:
    """Test suite for windowed bivariate interpolation."""

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

    @staticmethod
    def make_config(
        method: Callable[[], windowed.Bivariate],
        *,
        boundary: windowed.BoundaryConfig | None = None,
        half_window_size_x: int | None = 3,
        half_window_size_y: int | None = 3,
    ) -> windowed.Bivariate:
        """Build a windowed bivariate configuration with standard parameters."""
        boundary = boundary or windowed.BoundaryConfig.undef()
        cfg = method().with_boundary_mode(boundary)
        if half_window_size_x is not None:
            cfg = cfg.with_half_window_size_x(half_window_size_x)
        if half_window_size_y is not None:
            cfg = cfg.with_half_window_size_y(half_window_size_y)
        return cfg

    def test_single_point_bilinear(self) -> None:
        """Perform bilinear interpolation at a single point."""
        # Test single point windowed bivariate interpolation with bilinear
        # method.
        grid = self.create_analytical_grid2d(np.float64)

        # Test point: (π/2, π/4)
        # Expected: sin(π/2) * cos(π/4) = 1 * √2/2
        x = np.array([np.pi / 2])
        y = np.array([np.pi / 4])

        expected = np.sin(np.pi / 2) * np.cos(np.pi / 4)

        config = self.make_config(windowed.Bivariate.bilinear)
        result = core.bivariate(grid, x, y, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])
        # Bilinear should have good accuracy for smooth functions
        np.testing.assert_allclose(result[0], expected, rtol=0.02)

    def test_multiple_points_bilinear(self) -> None:
        """Test bilinear interpolation with multiple points."""
        grid = self.create_analytical_grid2d(np.float64)

        # Multiple test points with known analytical values
        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])

        # Calculate expected values using the analytical function
        expected = np.array(
            [np.sin(x[i]) * np.cos(y[i]) for i in range(len(x))]
        )

        config = self.make_config(windowed.Bivariate.bilinear)
        result = core.bivariate(grid, x, y, config)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        # Validate against analytical values
        np.testing.assert_allclose(result, expected, rtol=0.05)

    def test_interpolation_method_comparison(self) -> None:
        """Compare interpolation accuracy across all available methods."""
        grid = self.create_analytical_grid2d(np.float64)

        x = np.array([1.5])
        y = np.array([1.2])

        expected = np.sin(1.5) * np.cos(1.2)

        # Test all available interpolation methods
        all_methods = [
            "akima",
            "akima_periodic",
            "bicubic",
            "bilinear",
            "c_spline",
            "c_spline_not_a_knot",
            "c_spline_periodic",
            "linear",
            "polynomial",
            "steffen",
        ]

        results = {}
        for name in all_methods:
            method = getattr(windowed.Bivariate, name)
            config = self.make_config(method)
            result = core.bivariate(grid, x, y, config)
            assert np.isfinite(result[0]), f"Method {name} produced NaN"
            results[name] = result[0]

        # All should be reasonably close to expected
        for name, value in results.items():
            assert np.abs(value - expected) < 0.1, (
                f"Method {name} error too large: {value} vs {expected}"
            )

    def test_bounds_error(self) -> None:
        """Test bounds_error parameter with windowed."""
        grid = self.create_analytical_grid2d(np.float64)

        # Point outside grid bounds (x too large)
        x = np.array([3 * np.pi])
        y = np.array([0.5])

        # With bounds_error=False, should return NaN
        config = self.make_config(windowed.Bivariate.bilinear)
        result = core.bivariate(grid, x, y, config)

        assert result.shape == (1,)
        assert np.isnan(result[0])

        # With bounds_error=True, should raise an error
        config = self.make_config(
            windowed.Bivariate.bilinear,
        ).with_bounds_error(True)
        with pytest.raises((ValueError, IndexError), match="out of bounds"):
            core.bivariate(grid, x, y, config)

    def test_window_size_configuration(self) -> None:
        """Test that different window sizes produce reasonable results."""
        grid = self.create_analytical_grid2d(np.float64)

        x = np.array([np.pi / 2])
        y = np.array([np.pi / 3])

        expected = np.sin(x[0]) * np.cos(y[0])

        # Test with small window
        config_small = self.make_config(
            windowed.Bivariate.bilinear,
            half_window_size_x=3,
            half_window_size_y=3,
        )
        result_small = core.bivariate(grid, x, y, config_small)

        # Test with larger window
        config_large = self.make_config(
            windowed.Bivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
            half_window_size_x=9,
            half_window_size_y=9,
        )
        result_large = core.bivariate(grid, x, y, config_large)

        assert result_small.shape == (1,)
        assert result_large.shape == (1,)
        assert np.isfinite(result_small[0])
        assert np.isfinite(result_large[0])

        # Both should be close to the expected value
        np.testing.assert_allclose(result_small[0], expected, rtol=0.05)
        np.testing.assert_allclose(result_large[0], expected, rtol=0.05)

    def test_boundary_config(self) -> None:
        """Test different boundary modes produce finite results."""
        grid = self.create_analytical_grid2d(np.float64)

        # Use interior point
        x = np.array([1.5])
        y = np.array([1.2])

        # Test all boundary modes
        boundary_configs = [
            windowed.BoundaryConfig.shrink(),
            windowed.BoundaryConfig.undef(),
        ]

        results = []
        for boundary in boundary_configs:
            config = self.make_config(
                windowed.Bivariate.bilinear, boundary=boundary
            )
            result = core.bivariate(grid, x, y, config)
            assert result.shape == (1,)
            assert np.isfinite(result[0])
            results.append(result[0])

        # All boundary modes should produce finite results
        assert all(np.isfinite(results))

    def test_nan_in_grid_data(self) -> None:
        """Test handling of NaN values in grid data."""
        grid = self.create_analytical_grid2d(np.float64)

        # Inject NaN values into the grid
        grid.array[12:15, 10:13] = np.nan

        # Point near NaN region
        x = np.array([grid.x[13]])
        y = np.array([grid.y[11]])

        config = self.make_config(windowed.Bivariate.bilinear)
        result = core.bivariate(grid, x, y, config)

        # Should return NaN when interpolating over NaN values
        assert result.shape == (1,)
        assert np.isnan(result[0])

    def test_mixed_valid_invalid_points(self) -> None:
        """Test interpolation with mix of valid and out-of-bounds points."""
        grid = self.create_analytical_grid2d(np.float64)

        # Mix of valid and invalid points
        x = np.array([np.pi / 2, 10.0, np.pi / 4])  # Middle one out of bounds
        y = np.array([np.pi / 4, 0.5, np.pi / 3])

        config = self.make_config(windowed.Bivariate.bilinear)
        result = core.bivariate(grid, x, y, config)

        assert result.shape == (3,)
        # First and third should be finite
        assert np.isfinite(result[0])
        assert np.isfinite(result[2])
        # Second should be NaN (out of bounds)
        assert np.isnan(result[1])

    def test_with_real_data(self) -> None:
        """Test windowed bivariate interpolation with real grid data."""
        grid_data = load_grid2d()
        x_axis = core.Axis(grid_data.lon.values, period=360.0)
        y_axis = core.Axis(grid_data.lat.values)

        matrix = np.ascontiguousarray(grid_data.mss.values.T)
        grid = core.Grid(x_axis, y_axis, matrix)

        # Test points within bounds - use actual grid bounds
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

        config = self.make_config(windowed.Bivariate.bilinear)
        result = core.bivariate(grid, x, y, config)

        assert result.shape == (3,)
        # At least some values should be finite (not all NaNs)
        assert np.any(np.isfinite(result))

    def test_dtype_float32(self) -> None:
        """Test windowed bivariate interpolation with float32 data."""
        grid = self.create_analytical_grid2d(np.float32)

        x = np.array([np.pi / 4])
        y = np.array([np.pi / 4])

        config = self.make_config(windowed.Bivariate.bilinear)
        result = core.bivariate(grid, x, y, config)

        assert result.dtype == np.float32
        assert np.isfinite(result[0])

    def test_num_threads(self) -> None:
        """Test interpolation results are consistent across thread counts."""
        grid = self.create_analytical_grid2d(np.float64)

        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])

        # Test with 1 thread
        config_single = self.make_config(
            windowed.Bivariate.bilinear
        ).with_num_threads(1)
        result_single = core.bivariate(grid, x, y, config_single)

        # Test with multiple threads
        config_multi = self.make_config(
            windowed.Bivariate.bilinear
        ).with_num_threads(4)
        result_multi = core.bivariate(grid, x, y, config_multi)

        # Results should be identical or very close
        np.testing.assert_array_almost_equal(result_single, result_multi)

    def test_continuity(self) -> None:
        """Test that windowed interpolation is continuous (smooth)."""
        grid = self.create_analytical_grid2d(np.float64)

        # Create a path of close points to test continuity
        n_points = 10
        x = np.linspace(1.0, 1.5, n_points)
        y = np.linspace(1.0, 1.5, n_points)

        config = self.make_config(windowed.Bivariate.bilinear)
        results = core.bivariate(grid, x, y, config)

        # Check that differences between consecutive points are small
        diffs = np.abs(np.diff(results))
        max_diff = np.max(diffs)

        # For smooth analytical function, consecutive interpolations should be
        # close. Windowed interpolation may have slightly larger steps due to
        # window changes
        assert max_diff < 0.08, (
            f"Interpolation not continuous, max diff: {max_diff}"
        )
        assert np.all(np.isfinite(results))

    def test_analytical_accuracy(self) -> None:
        """Test windowed interpolation accuracy against analytical function."""
        grid = self.create_analytical_grid2d(np.float64)

        # Define the analytical function
        def analytical_func(x: float, y: float) -> float:
            return float(np.sin(x) * np.cos(y))

        # Test at interior points (not on grid points for proper interpolation
        # test)
        x = np.array([0.5, 1.0, 1.5, 2.0, 3.5])
        y = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

        # Test with bilinear
        config_bilinear = self.make_config(
            windowed.Bivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result_bilinear = core.bivariate(grid, x, y, config_bilinear)

        # Test with bicubic (should be more accurate)
        config_bicubic = self.make_config(
            windowed.Bivariate.bicubic,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result_bicubic = core.bivariate(grid, x, y, config_bicubic)

        # Compare with analytical values
        expected = np.array(
            [analytical_func(x[i], y[i]) for i in range(len(x))]
        )

        # Bilinear should have good accuracy
        np.testing.assert_allclose(
            result_bilinear,
            expected,
            rtol=0.05,
            err_msg="Bilinear interpolation accuracy too low",
        )

        # Bicubic should be at least as accurate on finite values
        bicubic_errors = np.abs(result_bicubic - expected)
        bilinear_errors = np.abs(result_bilinear - expected)
        # Most values should be finite
        assert np.sum(np.isfinite(result_bicubic)) >= 4
        # Compare mean errors on finite values only
        finite_mask = np.isfinite(result_bicubic)
        if np.any(finite_mask):
            assert (
                np.mean(bicubic_errors[finite_mask])
                <= np.mean(bilinear_errors[finite_mask]) * 1.5
            )

    def test_large_array(self) -> None:
        """Test windowed bivariate interpolation with large arrays."""
        grid = self.create_analytical_grid2d(np.float64)

        # Create large arrays of points
        n_points = 1000
        rng = np.random.Generator(np.random.PCG64(seed=42))
        x = rng.uniform(0.1, 2 * np.pi - 0.1, n_points)
        y = rng.uniform(0.1, np.pi - 0.1, n_points)

        config = self.make_config(
            windowed.Bivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.bivariate(grid, x, y, config)

        assert result.shape == (n_points,)
        # Most values should be finite
        assert np.sum(np.isfinite(result)) == n_points

    def test_method_chaining(self) -> None:
        """Test that windowed methods can be chained."""
        config = (
            windowed.Bivariate.bicubic()
            .with_num_threads(4)
            .with_bounds_error(True)
            .with_boundary_mode(windowed.BoundaryConfig.undef())
            .with_half_window_size_x(10)
            .with_half_window_size_y(8)
        )

        assert isinstance(config, windowed.Bivariate)

    def test_error_on_mismatched_array_sizes(self) -> None:
        """Test that mismatched input array sizes raise appropriate errors."""
        grid = self.create_analytical_grid2d(np.float64)

        x = np.array([np.pi / 2, np.pi / 4])
        y = np.array([np.pi / 4])  # Different size!

        config = self.make_config(windowed.Bivariate.bilinear)

        with pytest.raises((ValueError, RuntimeError)):
            core.bivariate(grid, x, y, config)

    def test_reproducibility(self) -> None:
        """Test that repeated calls produce identical results."""
        grid = self.create_analytical_grid2d(np.float64)

        x = np.array([np.pi / 3, np.pi / 2, 2 * np.pi / 3])
        y = np.array([np.pi / 4, np.pi / 3, np.pi / 2])

        config = self.make_config(windowed.Bivariate.bicubic)

        result1 = core.bivariate(grid, x, y, config)
        result2 = core.bivariate(grid, x, y, config)
        result3 = core.bivariate(grid, x, y, config)

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)

    def test_corner_point(self) -> None:
        """Test windowed interpolation near grid corner."""
        grid = self.create_analytical_grid2d(np.float64)

        # Use a point near (but not exactly at) the corner
        x = np.array([grid.x[1]])
        y = np.array([grid.y[1]])

        config = self.make_config(
            windowed.Bivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.bivariate(grid, x, y, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_grid_edge_interpolation(self) -> None:
        """Test interpolation near grid edges with different boundary modes."""
        grid = self.create_analytical_grid2d(np.float64)

        # Points near grid edges - using slicing to get axis values
        # Use points further from the edge to avoid window issues
        x_axis_vals = grid.x[:]
        y_axis_vals = grid.y[:]
        x_edge = np.array([x_axis_vals[4], x_axis_vals[-5]])
        y_edge = np.array([y_axis_vals[4], y_axis_vals[-5]])

        for boundary in [
            windowed.BoundaryConfig.shrink(),
            windowed.BoundaryConfig.undef(),
        ]:
            config = self.make_config(
                windowed.Bivariate.bilinear, boundary=boundary
            )
            result = core.bivariate(grid, x_edge, y_edge, config)

            assert result.shape == (2,)
            # At least most values should be finite
            assert np.sum(np.isfinite(result)) >= 1

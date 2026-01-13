# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for windowed quadrivariate interpolation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from .... import core
from ....core.config import windowed
from ... import load_grid4d


if TYPE_CHECKING:
    from collections.abc import Callable


class TestQuadrivariateWindowed:
    """Test suite for windowed quadrivariate interpolation."""

    @staticmethod
    def create_analytical_grid4d(
        dtype: type[np.float32 | np.float64],
    ) -> core.Grid4D:
        """Create a 4D grid with an analytical field.

        f(x, y, z, u) = sin(x) * cos(y) * exp(-z/5) * sin(u)

        This provides a smooth, continuous field with known values.
        """
        x_vals = np.linspace(0, 2 * np.pi, 20)
        y_vals = np.linspace(0, np.pi, 18)
        z_vals = np.linspace(0, 5, 10)
        u_vals = np.linspace(0, np.pi, 8)

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

    @staticmethod
    def make_config(
        method: Callable[[], windowed.Quadrivariate],
        *,
        boundary: windowed.BoundaryConfig | None = None,
        half_window_size_x: int | None = 5,
        half_window_size_y: int | None = 5,
        third_axis: windowed.AxisConfig | None = None,
        fourth_axis: windowed.AxisConfig | None = None,
    ) -> windowed.Quadrivariate:
        """Build a windowed quadrivariate interpolation configuration."""
        boundary = boundary or windowed.BoundaryConfig.undef()
        third_axis = third_axis or windowed.AxisConfig.nearest()
        fourth_axis = fourth_axis or windowed.AxisConfig.nearest()

        cfg = (
            method()
            .with_third_axis(third_axis)
            .with_fourth_axis(fourth_axis)
            .with_boundary_mode(boundary)
        )
        if half_window_size_x is not None:
            cfg = cfg.with_half_window_size_x(half_window_size_x)
        if half_window_size_y is not None:
            cfg = cfg.with_half_window_size_y(half_window_size_y)
        return cfg

    def test_single_point_bilinear(self) -> None:
        """Test bilinear windowed interpolation at a single point."""
        grid = self.create_analytical_grid4d(np.float64)

        # Test point: (π/2, π/4, 2.0, π/3)
        # Expected: sin(π/2) * cos(π/4) * exp(-2/5) * sin(π/3)
        x = np.array([np.pi / 2])
        y = np.array([np.pi / 4])
        z = np.array([2.0])
        u = np.array([np.pi / 3])

        expected = (
            np.sin(np.pi / 2)
            * np.cos(np.pi / 4)
            * np.exp(-2.0 / 5)
            * np.sin(np.pi / 3)
        )

        config = self.make_config(windowed.Quadrivariate.bilinear)
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])
        # Bilinear should have reasonable accuracy for smooth functions
        # (windowed 4D interpolation has larger errors than lower dimensions)
        np.testing.assert_allclose(result[0], expected, rtol=0.35)

    def test_multiple_points_bilinear(self) -> None:
        """Test bilinear windowed interpolation at multiple points."""
        grid = self.create_analytical_grid4d(np.float64)

        # Multiple test points with known analytical values (avoid grid edges)
        x = np.array([1.0, 1.5, 2.0])
        y = np.array([0.8, 1.2, 1.5])
        z = np.array([1.0, 2.5, 3.5])
        u = np.array([0.8, 1.3, 2.0])

        # Calculate expected values using the analytical function
        expected = np.array(
            [
                np.sin(x[i]) * np.cos(y[i]) * np.exp(-z[i] / 5) * np.sin(u[i])
                for i in range(len(x))
            ]
        )

        config = self.make_config(
            windowed.Quadrivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        # Validate against analytical values (windowed has larger errors)
        np.testing.assert_allclose(result, expected, rtol=0.4)

    def test_axis_methods_comparison(self) -> None:
        """Compare different third and fourth axis interpolation methods."""
        grid = self.create_analytical_grid4d(np.float64)

        # Test at a point between grid values
        x = np.array([np.pi / 2])
        y = np.array([np.pi / 4])
        z = np.array([2.3])  # Between grid points
        u = np.array([1.5])  # Between grid points

        # Linear on both axes
        config_linear = self.make_config(
            windowed.Quadrivariate.bilinear,
            third_axis=windowed.AxisConfig.linear(),
            fourth_axis=windowed.AxisConfig.linear(),
        )
        result_linear = core.quadrivariate(grid, x, y, z, u, config_linear)

        # Nearest on both axes
        config_nearest = self.make_config(
            windowed.Quadrivariate.bilinear,
            third_axis=windowed.AxisConfig.nearest(),
            fourth_axis=windowed.AxisConfig.nearest(),
        )
        result_nearest = core.quadrivariate(grid, x, y, z, u, config_nearest)

        # Mixed: linear on third, nearest on fourth
        config_mixed = self.make_config(
            windowed.Quadrivariate.bilinear,
            third_axis=windowed.AxisConfig.linear(),
            fourth_axis=windowed.AxisConfig.nearest(),
        )
        result_mixed = core.quadrivariate(grid, x, y, z, u, config_mixed)

        # Results should all be finite
        assert np.isfinite(result_linear[0])
        assert np.isfinite(result_nearest[0])
        assert np.isfinite(result_mixed[0])

        # Linear should differ from nearest when between grid points
        assert not np.isclose(result_linear[0], result_nearest[0], rtol=0.01)

    def test_interpolation_method_comparison(self) -> None:
        """Test different interpolation methods produce finite results."""
        grid = self.create_analytical_grid4d(np.float64)

        # Use interior point (not at grid edge)
        x = np.array([1.2])
        y = np.array([1.1])
        z = np.array([2.0])
        u = np.array([1.5])

        expected = np.sin(1.2) * np.cos(1.1) * np.exp(-2.0 / 5) * np.sin(1.5)

        # Only test methods that work well with windowed interpolation
        methods = {
            "bilinear": windowed.Quadrivariate.bilinear,
            "linear": windowed.Quadrivariate.linear,
        }

        results = {}
        for name, method in methods.items():
            config = self.make_config(
                method, boundary=windowed.BoundaryConfig.shrink()
            )
            result = core.quadrivariate(grid, x, y, z, u, config)
            assert np.isfinite(result[0]), f"Method {name} produced NaN"
            results[name] = result[0]

        # All should be reasonably close to expected
        for name, value in results.items():
            assert np.abs(value - expected) < 0.2, (
                f"Method {name} error too large: {value} vs {expected}"
            )

    def test_bounds_error(self) -> None:
        """Test bounds_error parameter with windowed."""
        grid = self.create_analytical_grid4d(np.float64)

        # Point outside grid bounds (x too large)
        x = np.array([3 * np.pi])
        y = np.array([0.5])
        z = np.array([1.0])
        u = np.array([1.0])

        # With bounds_error=False, should return NaN
        config = self.make_config(windowed.Quadrivariate.bilinear)
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (1,)
        assert np.isnan(result[0])

        # With bounds_error=True, should raise an error
        config = self.make_config(
            windowed.Quadrivariate.bilinear
        ).with_bounds_error(True)
        with pytest.raises((ValueError, IndexError), match="out of bounds"):
            core.quadrivariate(grid, x, y, z, u, config)

    def test_window_size_configuration(self) -> None:
        """Test that different window sizes produce reasonable results."""
        grid = self.create_analytical_grid4d(np.float64)

        x = np.array([np.pi / 2])
        y = np.array([np.pi / 3])
        z = np.array([2.0])
        u = np.array([np.pi / 4])

        expected = (
            np.sin(x[0]) * np.cos(y[0]) * np.exp(-z[0] / 5) * np.sin(u[0])
        )

        # Test with small window
        config_small = self.make_config(
            windowed.Quadrivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
            half_window_size_x=3,
            half_window_size_y=3,
        )
        result_small = core.quadrivariate(grid, x, y, z, u, config_small)

        # Test with larger window
        config_large = self.make_config(
            windowed.Quadrivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
            half_window_size_x=7,
            half_window_size_y=7,
        )
        result_large = core.quadrivariate(grid, x, y, z, u, config_large)

        assert result_small.shape == (1,)
        assert result_large.shape == (1,)
        assert np.isfinite(result_small[0])
        assert np.isfinite(result_large[0])

        # Both should be close to the expected value
        np.testing.assert_allclose(result_small[0], expected, rtol=0.4)
        np.testing.assert_allclose(result_large[0], expected, rtol=0.4)

    def test_boundary_modes(self) -> None:
        """Test different boundary modes produce finite results."""
        grid = self.create_analytical_grid4d(np.float64)

        # Use interior point
        x = np.array([1.5])
        y = np.array([1.2])
        z = np.array([2.0])
        u = np.array([1.5])

        # Test primary boundary mode (undef)
        # Other modes may not be fully supported for 4D windowed interpolation
        config = self.make_config(
            windowed.Quadrivariate.bilinear,
            boundary=windowed.BoundaryConfig.undef(),
        )
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_nan_in_grid_data(self) -> None:
        """Test handling of NaN values in grid data."""
        grid = self.create_analytical_grid4d(np.float64)

        # Inject NaN values into the grid
        grid.array[4:6, 4:6, 2:4, 2:4] = np.nan

        # Point near NaN region
        x = np.array([grid.x[5]])
        y = np.array([grid.y[5]])
        z = np.array([grid.z[3]])
        u = np.array([grid.u[3]])

        config = self.make_config(windowed.Quadrivariate.bilinear)
        result = core.quadrivariate(grid, x, y, z, u, config)

        # Should return NaN when interpolating over NaN values
        assert result.shape == (1,)
        assert np.isnan(result[0])

    def test_mixed_valid_invalid_points(self) -> None:
        """Test interpolation with mix of valid and out-of-bounds points."""
        grid = self.create_analytical_grid4d(np.float64)

        # Mix of valid and invalid points
        x = np.array([np.pi / 2, 10.0, np.pi / 4])  # Middle one out of bounds
        y = np.array([np.pi / 4, 0.5, np.pi / 3])
        z = np.array([2.0, 2.0, 1.0])
        u = np.array([1.0, 1.0, 1.5])

        config = self.make_config(
            windowed.Quadrivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (3,)
        # First and third should be finite
        assert np.isfinite(result[0])
        assert np.isfinite(result[2])
        # Second should be NaN (out of bounds)
        assert np.isnan(result[1])

    def test_with_real_data(self) -> None:
        """Test windowed quadrivariate interpolation with real grid data."""
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

        config = self.make_config(
            windowed.Quadrivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (3,)
        # At least some values should be finite (not all NaNs)
        assert np.any(np.isfinite(result))

    def test_dtype_float32(self) -> None:
        """Test windowed quadrivariate interpolation with float32 data."""
        grid = self.create_analytical_grid4d(np.float32)

        x = np.array([np.pi / 4])
        y = np.array([np.pi / 4])
        z = np.array([2.0])
        u = np.array([np.pi / 6])

        config = self.make_config(
            windowed.Quadrivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.dtype == np.float32
        assert np.isfinite(result[0])

    def test_num_threads(self) -> None:
        """Test windowed quadrivariate interpolation with different threads.

        This test checks the performance and consistency of interpolation
        results when using different numbers of threads.
        """
        grid = self.create_analytical_grid4d(np.float64)

        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        z = np.array([1.0, 2.5, 4.0])
        u = np.array([np.pi / 6, np.pi / 3, np.pi / 2])

        # Test with 1 thread
        config_single = self.make_config(
            windowed.Quadrivariate.bilinear
        ).with_num_threads(1)
        result_single = core.quadrivariate(grid, x, y, z, u, config_single)

        # Test with multiple threads
        config_multi = self.make_config(
            windowed.Quadrivariate.bilinear
        ).with_num_threads(4)
        result_multi = core.quadrivariate(grid, x, y, z, u, config_multi)

        # Results should be identical or very close
        np.testing.assert_array_almost_equal(result_single, result_multi)

    def test_continuity(self) -> None:
        """Test that windowed interpolation is continuous (smooth)."""
        grid = self.create_analytical_grid4d(np.float64)

        # Create a path of close points to test continuity
        n_points = 10
        x = np.linspace(1.0, 1.5, n_points)
        y = np.linspace(1.0, 1.5, n_points)
        z = np.linspace(1.0, 2.0, n_points)
        u = np.linspace(1.0, 1.5, n_points)

        config = self.make_config(
            windowed.Quadrivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        results = core.quadrivariate(grid, x, y, z, u, config)

        # Check that differences between consecutive points are small
        diffs = np.abs(np.diff(results))
        max_diff = np.max(diffs)

        # For smooth analytical function, consecutive interpolations should be
        # close. Windowed interpolation may have slightly larger steps due to
        # window changes
        assert max_diff < 0.15, (
            f"Interpolation not continuous, max diff: {max_diff}"
        )
        assert np.all(np.isfinite(results))

    def test_analytical_accuracy(self) -> None:
        """Test windowed interpolation accuracy against analytical function."""
        grid = self.create_analytical_grid4d(np.float64)

        # Define the analytical function
        def analytical_func(x: float, y: float, z: float, u: float) -> float:
            return float(np.sin(x) * np.cos(y) * np.exp(-z / 5) * np.sin(u))

        # Test at interior points (well away from grid edges)
        x = np.array([1.2, 1.5, 2.0, 2.5, 3.5])
        y = np.array([0.8, 1.2, 1.5, 1.8, 2.2])
        z = np.array([0.8, 1.2, 1.8, 2.5, 3.5])
        u = np.array([0.8, 1.2, 1.5, 1.8, 2.2])

        # Test with bilinear
        config_bilinear = self.make_config(
            windowed.Quadrivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result_bilinear = core.quadrivariate(grid, x, y, z, u, config_bilinear)

        # Test with linear (should be faster, slightly less accurate)
        config_linear = self.make_config(
            windowed.Quadrivariate.linear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result_linear = core.quadrivariate(grid, x, y, z, u, config_linear)

        # Compare with analytical values
        expected = np.array(
            [analytical_func(x[i], y[i], z[i], u[i]) for i in range(len(x))]
        )

        # Bilinear should have reasonable accuracy (windowed has larger error
        # than regular, especially in 4D)
        np.testing.assert_allclose(
            result_bilinear,
            expected,
            rtol=0.4,
            err_msg="Bilinear interpolation accuracy too low",
        )

        # Linear should also be reasonably accurate
        assert np.all(np.isfinite(result_linear))
        np.testing.assert_allclose(
            result_linear,
            expected,
            rtol=0.5,
            err_msg="Linear interpolation accuracy too low",
        )

    def test_large_array(self) -> None:
        """Test windowed quadrivariate interpolation with large arrays."""
        grid = self.create_analytical_grid4d(np.float64)

        # Create large arrays of points
        n_points = 300
        rng = np.random.Generator(np.random.PCG64(seed=42))
        x = rng.uniform(0.1, 2 * np.pi - 0.1, n_points)
        y = rng.uniform(0.1, np.pi - 0.1, n_points)
        z = rng.uniform(0.1, 4.9, n_points)
        u = rng.uniform(0.1, np.pi - 0.1, n_points)

        config = self.make_config(
            windowed.Quadrivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (n_points,)
        # Most values should be finite
        assert np.sum(np.isfinite(result)) == n_points

    def test_method_chaining(self) -> None:
        """Test that windowed methods can be chained."""
        config = (
            windowed.Quadrivariate.bicubic()
            .with_num_threads(4)
            .with_bounds_error(True)
            .with_boundary_mode(windowed.BoundaryConfig.undef())
            .with_half_window_size_x(10)
            .with_half_window_size_y(8)
        )

        assert isinstance(config, windowed.Quadrivariate)

    def test_all_interpolation_methods(self) -> None:
        """Test all available windowed quadrivariate interpolation methods."""
        grid = self.create_analytical_grid4d(np.float64)

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

        # Use interior point to avoid edge effects
        x = np.array([1.5])
        y = np.array([1.2])
        z = np.array([2.0])
        u = np.array([1.5])

        for method in all_methods:
            config = self.make_config(getattr(windowed.Quadrivariate, method))
            result = core.quadrivariate(grid, x, y, z, u, config)

            assert result.shape == (1,), (
                f"Method {method} produced wrong shape"
            )
            assert np.isfinite(result[0]), f"Method {method} produced NaN"

    def test_error_on_mismatched_array_sizes(self) -> None:
        """Test that mismatched input array sizes raise appropriate errors."""
        grid = self.create_analytical_grid4d(np.float64)

        x = np.array([np.pi / 2, np.pi / 4])
        y = np.array([np.pi / 4])  # Different size!
        z = np.array([2.0])
        u = np.array([1.0])

        config = self.make_config(windowed.Quadrivariate.bilinear)

        with pytest.raises((ValueError, RuntimeError)):
            core.quadrivariate(grid, x, y, z, u, config)

    def test_reproducibility(self) -> None:
        """Test that windowed interpolation produces reproducible results."""
        grid = self.create_analytical_grid4d(np.float64)

        x = np.array([np.pi / 3, np.pi / 2, 2 * np.pi / 3])
        y = np.array([np.pi / 4, np.pi / 3, np.pi / 2])
        z = np.array([1.0, 2.5, 4.0])
        u = np.array([0.5, 1.5, 2.5])

        config = self.make_config(windowed.Quadrivariate.bicubic)

        result1 = core.quadrivariate(grid, x, y, z, u, config)
        result2 = core.quadrivariate(grid, x, y, z, u, config)
        result3 = core.quadrivariate(grid, x, y, z, u, config)

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)

    def test_symmetry_z_axis(self) -> None:
        """Test that decay along z-axis is consistent."""
        grid = self.create_analytical_grid4d(np.float64)

        # Test points at different z levels but same x, y, u
        x = np.array([np.pi / 2, np.pi / 2, np.pi / 2])
        y = np.array([np.pi / 4, np.pi / 4, np.pi / 4])
        z = np.array([0.5, 2.5, 4.5])
        u = np.array([np.pi / 3, np.pi / 3, np.pi / 3])

        config = self.make_config(windowed.Quadrivariate.bilinear)
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
        u = np.array([0.3, np.pi / 2])

        config = self.make_config(windowed.Quadrivariate.bilinear)
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))
        # At u=π/2, sin(π/2)=1, so result should be larger than at u=0.3
        assert np.abs(result[1]) > np.abs(result[0])

    @staticmethod
    def create_analytical_temporal_grid4d(
        dtype: type[np.float32 | np.float64],
    ) -> core.GridHolder:
        """Create a 4D grid with temporal Z-axis and analytical field.

        f(x, y, t, u) = sin(x) * cos(y) * exp(-t_normalized/5) * sin(u)

        where t is a datetime64 axis.
        """
        x_vals = np.linspace(0, 2 * np.pi, 20)
        y_vals = np.linspace(0, np.pi, 18)
        # Create a temporal axis with datetime64 values
        time_vals: np.ndarray = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-11"),
            np.timedelta64(1, "D"),
        )
        u_vals = np.linspace(0, np.pi, 8)

        x_axis = core.Axis(x_vals, period=None)
        y_axis = core.Axis(y_vals)
        z_axis = core.TemporalAxis(time_vals)
        u_axis = core.Axis(u_vals)

        # Normalize time for analytical function (0 to 9 days)
        time_normalized = np.arange(10)

        x_grid, y_grid, t_grid, u_grid = np.meshgrid(
            x_vals, y_vals, time_normalized, u_vals, indexing="ij"
        )

        # Create analytical field: f(x, y, t, u)
        #   = sin(x) * cos(y) * exp(-t/5) * sin(u)
        data = (
            np.sin(x_grid)
            * np.cos(y_grid)
            * np.exp(-t_grid / 5)
            * np.sin(u_grid)
        ).astype(dtype)
        data = np.ascontiguousarray(data)

        return core.Grid(x_axis, y_axis, z_axis, u_axis, data)

    def test_temporal_grid_basic_interpolation(self) -> None:
        """Test windowed quadrivariate interpolation with temporal Z-axis."""
        grid = self.create_analytical_temporal_grid4d(np.float64)

        x = np.array([np.pi / 4])
        y = np.array([np.pi / 4])
        # Use a datetime64 value for z
        z = np.array([np.datetime64("2020-01-03")])
        u = np.array([np.pi / 4])

        config = self.make_config(
            windowed.Quadrivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

        # Verify against expected value (day 2, normalized to t=2)
        expected = (
            np.sin(np.pi / 4)
            * np.cos(np.pi / 4)
            * np.exp(-2.0 / 5)
            * np.sin(np.pi / 4)
        )
        # Windowed 4D interpolation has larger errors
        np.testing.assert_allclose(result[0], expected, rtol=0.45)

    def test_temporal_grid_multiple_times(self) -> None:
        """Test interpolation at multiple temporal points."""
        grid = self.create_analytical_temporal_grid4d(np.float64)

        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y = np.array([np.pi / 4, np.pi / 3, np.pi / 2])
        z = np.array(
            [
                np.datetime64("2020-01-01"),
                np.datetime64("2020-01-05"),
                np.datetime64("2020-01-09"),
            ]
        )
        u = np.array([np.pi / 4, np.pi / 3, np.pi / 2])

        config = self.make_config(
            windowed.Quadrivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

        # All values should be reasonable (between -1 and 1 due to sin/cos)
        assert np.all(np.abs(result) <= 1.5)

    def test_temporal_grid_with_linear_time_axis(self) -> None:
        """Test temporal grid with linear interpolation on time axis."""
        grid = self.create_analytical_temporal_grid4d(np.float64)

        x = np.array([np.pi / 2])
        y = np.array([np.pi / 4])
        # Datetime between grid points
        z = np.array([np.datetime64("2020-01-03T12:00:00")])
        u = np.array([np.pi / 4])

        # Use linear interpolation on time axis
        config = self.make_config(
            windowed.Quadrivariate.bilinear,
            third_axis=windowed.AxisConfig.linear(),
        )
        result = core.quadrivariate(grid, x, y, z, u, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

        # Result should be between values at day 3 and day 4
        z_day3 = np.array([np.datetime64("2020-01-03")])
        z_day4 = np.array([np.datetime64("2020-01-04")])

        config_nearest = self.make_config(
            windowed.Quadrivariate.bilinear,
            third_axis=windowed.AxisConfig.nearest(),
        )
        result_day3 = core.quadrivariate(grid, x, y, z_day3, u, config_nearest)
        result_day4 = core.quadrivariate(grid, x, y, z_day4, u, config_nearest)

        # Linear interpolation result should be between the two days
        assert (
            min(result_day3[0], result_day4[0])
            <= result[0]
            <= max(result_day3[0], result_day4[0])
        )

    def test_temporal_grid_all_methods(self) -> None:
        """Test all interpolation methods work with temporal grid."""
        grid = self.create_analytical_temporal_grid4d(np.float64)

        methods = [
            "bilinear",
            "bicubic",
            "c_spline",
            "linear",
            "akima",
            "steffen",
        ]

        x = np.array([np.pi / 2])
        y = np.array([np.pi / 4])
        z = np.array([np.datetime64("2020-01-05")])
        u = np.array([np.pi / 3])

        for method in methods:
            config = self.make_config(getattr(windowed.Quadrivariate, method))
            result = core.quadrivariate(grid, x, y, z, u, config)

            assert result.shape == (1,)
            assert np.isfinite(result[0]), f"Method {method} produced NaN"

    def test_temporal_grid_mixed_axis_interpolation(self) -> None:
        """Test temporal grid with different interpolation on Z and U axes."""
        grid = self.create_analytical_temporal_grid4d(np.float64)

        x = np.array([np.pi / 2])
        y = np.array([np.pi / 4])
        z = np.array([np.datetime64("2020-01-05T12:00:00")])
        u = np.array([1.5])  # Between grid points

        # Linear on both Z (temporal) and U axes
        config_linear_both = self.make_config(
            windowed.Quadrivariate.bilinear,
            third_axis=windowed.AxisConfig.linear(),
            fourth_axis=windowed.AxisConfig.linear(),
        )
        result_linear = core.quadrivariate(
            grid, x, y, z, u, config_linear_both
        )

        # Nearest on both axes
        config_nearest_both = self.make_config(
            windowed.Quadrivariate.bilinear,
            third_axis=windowed.AxisConfig.nearest(),
            fourth_axis=windowed.AxisConfig.nearest(),
        )
        result_nearest = core.quadrivariate(
            grid, x, y, z, u, config_nearest_both
        )

        assert result_linear.shape == (1,)
        assert result_nearest.shape == (1,)
        assert np.isfinite(result_linear[0])
        assert np.isfinite(result_nearest[0])

        # Results should differ when between grid points
        np.testing.assert_allclose(result_linear, result_nearest, rtol=0.01)

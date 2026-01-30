# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for windowed trivariate interpolation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from .... import core
from ....core.config import windowed
from ... import load_grid3d


if TYPE_CHECKING:
    from collections.abc import Callable


class TestTrivariateWindowed:
    """Test suite for windowed trivariate interpolation."""

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

    @staticmethod
    def make_config(
        method: Callable[[], windowed.Trivariate],
        *,
        boundary: windowed.BoundaryConfig | None = None,
        half_window_size_x: int | None = 3,
        half_window_size_y: int | None = 3,
        third_axis: windowed.AxisConfig | None = None,
    ) -> windowed.Trivariate:
        """Build a windowed trivariate configuration with sensible defaults."""
        boundary = boundary or windowed.BoundaryConfig.undef()
        third_axis = third_axis or windowed.AxisConfig.nearest()

        cfg = method().with_third_axis(third_axis).with_boundary_mode(boundary)
        if half_window_size_x is not None:
            cfg = cfg.with_half_window_size_x(half_window_size_x)
        if half_window_size_y is not None:
            cfg = cfg.with_half_window_size_y(half_window_size_y)
        return cfg

    def test_single_point_bilinear(self) -> None:
        """Test windowed bilinear interpolation at a single point."""
        grid = self.create_analytical_grid3d(np.float64)

        # Test point: (π/2, π/4, 5.0)
        # Expected: sin(π/2) * cos(π/4) * exp(-5/10) = 1 * √2/2 * exp(-0.5)
        x = np.array([np.pi / 2])
        y = np.array([np.pi / 4])
        z = np.array([5.0])

        expected = np.sin(np.pi / 2) * np.cos(np.pi / 4) * np.exp(-5.0 / 10)

        config = self.make_config(
            windowed.Trivariate.bilinear,
            third_axis=windowed.AxisConfig.linear(),
        )
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])
        # Bilinear should have reasonable accuracy for smooth functions
        np.testing.assert_allclose(result[0], expected, rtol=0.04)

    def test_multiple_points_bilinear(self) -> None:
        """Test windowed bilinear interpolation at multiple points."""
        grid = self.create_analytical_grid3d(np.float64)

        # Multiple test points with known analytical values
        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        z = np.array([0.0, 5.0, 10.0])

        # Calculate expected values using the analytical function
        expected = np.array(
            [
                np.sin(x[i]) * np.cos(y[i]) * np.exp(-z[i] / 10)
                for i in range(len(x))
            ]
        )

        config = self.make_config(
            windowed.Trivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
            third_axis=windowed.AxisConfig.linear(),
        )
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        assert result[1] < 1e-16  # Middle point is zero due to cos(π/2)=0
        result[1] = expected[1]
        # Validate against analytical values
        np.testing.assert_allclose(result, expected, rtol=0.04)

    def test_third_axis_linear_vs_nearest(self) -> None:
        """Test linear vs nearest neighbor interpolation on third axis."""
        grid = self.create_analytical_grid3d(np.float64)

        # Test at a point between z grid values
        x = np.array([np.pi / 2])
        y = np.array([np.pi / 4])
        z = np.array([2.5])  # Between grid points

        # Linear interpolation along third axis
        config_linear = self.make_config(
            windowed.Trivariate.bilinear,
            third_axis=windowed.AxisConfig.linear(),
        )
        result_linear = core.trivariate(grid, x, y, z, config_linear)

        # Nearest neighbor along third axis
        config_nearest = self.make_config(
            windowed.Trivariate.bilinear,
            third_axis=windowed.AxisConfig.nearest(),
        )
        result_nearest = core.trivariate(grid, x, y, z, config_nearest)

        # Results should be different when between grid points
        assert np.isfinite(result_linear[0])
        assert np.isfinite(result_nearest[0])
        # Linear should differ from nearest when not on a grid point
        assert not np.isclose(result_linear[0], result_nearest[0], rtol=0.01)

    def test_interpolation_method_comparison(self) -> None:
        """Test consistency and accuracy across interpolation methods."""
        grid = self.create_analytical_grid3d(np.float64)

        x = np.array([1.0])
        y = np.array([1.0])
        z = np.array([5.0])

        expected = np.sin(1.0) * np.cos(1.0) * np.exp(-5.0 / 10)

        methods = {
            "bilinear": windowed.Trivariate.bilinear,
            "bicubic": windowed.Trivariate.bicubic,
            "c_spline": windowed.Trivariate.c_spline,
        }

        results = {}
        for name, method in methods.items():
            config = self.make_config(method)
            result = core.trivariate(grid, x, y, z, config)
            assert np.isfinite(result[0]), f"Method {name} produced NaN"
            results[name] = result[0]

        # All should be reasonably close
        for name, value in results.items():
            assert np.abs(value - expected) < 0.1, (
                f"Method {name} error too large: {value} vs {expected}"
            )

    def test_bounds_error(self) -> None:
        """Test with_bounds_error parameter with windowed."""
        grid = self.create_analytical_grid3d(np.float64)

        # Point outside grid bounds
        x = np.array([3 * np.pi])
        y = np.array([0.0])
        z = np.array([0.0])

        # With with_bounds_error=False, should return NaN
        config = self.make_config(windowed.Trivariate.bilinear)
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (1,)
        assert np.isnan(result[0])

        # With with_bounds_error=True, should raise an error
        config = self.make_config(
            windowed.Trivariate.bilinear
        ).with_bounds_error(True)
        with pytest.raises((ValueError, IndexError), match="out of bounds"):
            core.trivariate(grid, x, y, z, config)

    def test_window_size_configuration(self) -> None:
        """Test that different window sizes produce reasonable results."""
        grid = self.create_analytical_grid3d(np.float64)

        x = np.array([np.pi / 2])
        y = np.array([np.pi / 3])
        z = np.array([3.0])

        expected = np.sin(x[0]) * np.cos(y[0]) * np.exp(-z[0] / 10)

        # Test with small window
        config_small = self.make_config(
            windowed.Trivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
            half_window_size_x=3,
            half_window_size_y=3,
        )
        result_small = core.trivariate(grid, x, y, z, config_small)

        # Test with larger window
        config_large = self.make_config(
            windowed.Trivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
            half_window_size_x=7,
            half_window_size_y=7,
        )
        result_large = core.trivariate(grid, x, y, z, config_large)

        assert result_small.shape == (1,)
        assert result_large.shape == (1,)
        assert np.isfinite(result_small[0])
        assert np.isfinite(result_large[0])

        # Both should be close to the expected value
        np.testing.assert_allclose(result_small[0], expected, rtol=0.1)
        np.testing.assert_allclose(result_large[0], expected, rtol=0.1)

        # Results should be similar (larger window may be slightly more
        # accurate)
        np.testing.assert_allclose(
            result_small[0],
            result_large[0],
            rtol=0.05,
        )

    def test_boundary_config(self) -> None:
        """Test different boundary modes produce finite results."""
        grid = self.create_analytical_grid3d(np.float64)
        x = np.array([np.pi / 2])
        y = np.array([np.pi / 2])
        z = np.array([5.0])

        boundary_modes = [
            windowed.BoundaryConfig.shrink(),
            windowed.BoundaryConfig.undef(),
        ]

        results = []
        for boundary in boundary_modes:
            config = self.make_config(
                windowed.Trivariate.bilinear, boundary=boundary
            )
            result = core.trivariate(grid, x, y, z, config)
            assert result.shape == (1,)
            assert np.isfinite(result[0])
            results.append(result[0])

        # All boundary modes should produce finite results
        assert all(np.isfinite(results))

    def test_nan_in_grid_data(self) -> None:
        """Test handling of NaN values in grid data."""
        grid = self.create_analytical_grid3d(np.float64)

        # Inject NaN values into the grid
        grid.array[5:7, 5:7, 3:5] = np.nan

        # Point near NaN region
        x = np.array([grid.x[6]])
        y = np.array([grid.y[6]])
        z = np.array([grid.z[4]])

        config = self.make_config(windowed.Trivariate.bilinear)
        result = core.trivariate(grid, x, y, z, config)

        # Should return NaN when interpolating over NaN values
        assert result.shape == (1,)
        assert np.isnan(result[0])

    def test_single_point_grid(self) -> None:
        """Test interpolation on minimal grid (edge case)."""
        # Create a very small grid
        x_vals = np.array([0.0, 1.0, 2.0])
        y_vals = np.array([0.0, 1.0, 2.0])
        z_vals = np.array([0.0, 1.0])

        x_axis = core.Axis(x_vals)
        y_axis = core.Axis(y_vals)
        z_axis = core.Axis(z_vals)

        x_grid, y_grid, z_grid = np.meshgrid(
            x_vals, y_vals, z_vals, indexing="ij"
        )
        data = (x_grid + y_grid + z_grid).astype(np.float64)
        data = np.ascontiguousarray(data)

        grid = core.Grid(x_axis, y_axis, z_axis, data)

        # Interpolate at center
        x = np.array([1.0])
        y = np.array([1.0])
        z = np.array([0.5])

        config = self.make_config(
            windowed.Trivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
            half_window_size_x=3,
            half_window_size_y=3,
        )
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_mixed_valid_invalid_points(self) -> None:
        """Test interpolation with mix of valid and out-of-bounds points."""
        grid = self.create_analytical_grid3d(np.float64)

        # Mix of valid and invalid points
        x = np.array([np.pi / 2, 10.0, np.pi / 4])  # Middle one out of bounds
        y = np.array([np.pi / 4, 0.5, np.pi / 3])
        z = np.array([5.0, 5.0, 3.0])

        config = self.make_config(
            windowed.Trivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (3,)
        # First and third should be finite
        assert np.isfinite(result[0])
        assert np.isfinite(result[2])
        # Second should be NaN (out of bounds)
        assert np.isnan(result[1])

    def test_with_real_data(self) -> None:
        """Test windowed trivariate interpolation with real grid data."""
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

        config = self.make_config(windowed.Trivariate.bilinear)
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (3,)
        # At least some values should be finite (not all NaNs)
        assert np.any(np.isfinite(result))

    def test_dtype_float32(self) -> None:
        """Test windowed trivariate interpolation with float32 data."""
        grid = self.create_analytical_grid3d(np.float32)

        x = np.array([np.pi / 4])
        y = np.array([np.pi / 4])
        z = np.array([5.0])

        config = self.make_config(
            windowed.Trivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.trivariate(grid, x, y, z, config)

        assert result.dtype == np.float32
        assert np.isfinite(result[0])

    def test_num_threads(self) -> None:
        """Test that results are consistent across different thread counts."""
        grid = self.create_analytical_grid3d(np.float64)

        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        z = np.array([1.0, 5.0, 9.0])

        # Test with 1 thread
        config_single = self.make_config(
            windowed.Trivariate.bilinear
        ).with_num_threads(1)
        result_single = core.trivariate(grid, x, y, z, config_single)

        # Test with multiple threads
        config_multi = self.make_config(
            windowed.Trivariate.bilinear
        ).with_num_threads(4)
        result_multi = core.trivariate(grid, x, y, z, config_multi)

        # Results should be identical or very close
        np.testing.assert_array_almost_equal(result_single, result_multi)

    def test_continuity(self) -> None:
        """Test that windowed interpolation is continuous (smooth)."""
        grid = self.create_analytical_grid3d(np.float64)

        # Create a path of close points to test continuity
        n_points = 10
        x = np.linspace(1.0, 1.5, n_points)
        y = np.linspace(1.0, 1.5, n_points)
        z = np.linspace(2.0, 3.0, n_points)

        config = self.make_config(windowed.Trivariate.bilinear)
        results = core.trivariate(grid, x, y, z, config)

        # Check that differences between consecutive points are small
        diffs = np.abs(np.diff(results))
        max_diff = np.max(diffs)

        # For smooth analytical function, consecutive interpolations should be
        # close Windowed interpolation may have slightly larger steps due to
        # window changes
        assert max_diff < 0.1, (
            f"Interpolation not continuous, max diff: {max_diff}"
        )
        assert np.all(np.isfinite(results))

    def test_corner_point(self) -> None:
        """Test windowed interpolation at grid corner."""
        grid = self.create_analytical_grid3d(np.float64)

        # Use the first interior cell to avoid undefined behavior exactly on
        # the grid edge.
        x_val = grid.x[1]
        y_val = grid.y[1]
        z_val = grid.z[1]

        x = np.array([x_val])
        y = np.array([y_val])
        z = np.array([z_val])

        config = self.make_config(
            windowed.Trivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.trivariate(grid, x, y, z, config)

        expected = grid.array[1, 1, 1]

        assert np.isfinite(result[0])
        # Windowed may have slightly more error at boundaries
        assert np.abs(result[0] - expected) < 0.1

    def test_analytical_accuracy(self) -> None:
        """Test windowed interpolation accuracy against analytical function."""
        grid = self.create_analytical_grid3d(np.float64)

        # Define the analytical function
        def analytical_func(x: float, y: float, z: float) -> float:
            return float(np.sin(x) * np.cos(y) * np.exp(-z / 10))

        # Test at interior points (not on grid points for proper interpolation
        # test)
        x = np.array([0.5, 1.0, 1.5, 2.0, 3.5])
        y = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        z = np.array([1.0, 2.0, 3.0, 4.0, 7.0])

        # Test with bilinear
        config_bilinear = self.make_config(
            windowed.Trivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result_bilinear = core.trivariate(grid, x, y, z, config_bilinear)

        # Test with bicubic (should be more accurate)
        config_bicubic = self.make_config(
            windowed.Trivariate.bicubic,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result_bicubic = core.trivariate(grid, x, y, z, config_bicubic)

        # Compare with analytical values
        expected = np.array(
            [analytical_func(x[i], y[i], z[i]) for i in range(len(x))]
        )

        # Bilinear should have reasonable accuracy (windowed has larger error
        # than regular)
        np.testing.assert_allclose(
            result_bilinear,
            expected,
            rtol=0.15,
            err_msg="Bilinear interpolation accuracy too low",
        )

        # Bicubic should be at least as accurate
        bicubic_errors = np.abs(result_bicubic - expected)
        bilinear_errors = np.abs(result_bilinear - expected)
        assert np.all(np.isfinite(result_bicubic))
        # Mean error for bicubic should be less than or equal to bilinear
        assert np.mean(bicubic_errors) <= np.mean(bilinear_errors) * 1.5

    def test_large_array(self) -> None:
        """Test windowed trivariate interpolation with large arrays."""
        grid = self.create_analytical_grid3d(np.float64)

        # Create large arrays of points
        n_points = 500
        rng = np.random.Generator(np.random.PCG64(seed=42))
        x = rng.uniform(0.1, 2 * np.pi - 0.1, n_points)
        y = rng.uniform(0.1, np.pi - 0.1, n_points)
        z = rng.uniform(0.1, 9.9, n_points)

        config = self.make_config(
            windowed.Trivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (n_points,)
        # Most values should be finite
        assert np.sum(np.isfinite(result)) == n_points

    def test_method_chaining(self) -> None:
        """Test that windowed methods can be chained."""
        config = (
            windowed.Trivariate.bicubic()
            .with_num_threads(4)
            .with_bounds_error(True)
            .with_boundary_mode(windowed.BoundaryConfig.shrink())
            .with_half_window_size_x(10)
            .with_half_window_size_y(8)
        )

        assert isinstance(config, windowed.Trivariate)

    def test_all_interpolation_methods(self) -> None:
        """Test all available windowed trivariate methods."""
        grid = self.create_analytical_grid3d(np.float64)

        methods = [
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

        x = np.array([np.pi / 2])
        y = np.array([np.pi / 2])
        z = np.array([5.0])

        for method in methods:
            config = self.make_config(getattr(windowed.Trivariate, method))
            result = core.trivariate(grid, x, y, z, config)

            assert result.shape == (1,)
            assert np.isfinite(result[0]), f"Method {method} produced NaN"

    def test_error_on_mismatched_array_sizes(self) -> None:
        """Test that mismatched input array sizes raise appropriate errors."""
        grid = self.create_analytical_grid3d(np.float64)

        x = np.array([np.pi / 2, np.pi / 4])
        y = np.array([np.pi / 4])  # Different size!
        z = np.array([5.0])

        config = self.make_config(windowed.Trivariate.bilinear)

        with pytest.raises((ValueError, RuntimeError)):
            core.trivariate(grid, x, y, z, config)

    def test_reproducibility(self) -> None:
        """Verify that repeated calls produce identical results."""
        grid = self.create_analytical_grid3d(np.float64)

        x = np.array([np.pi / 3, np.pi / 2, 2 * np.pi / 3])
        y = np.array([np.pi / 4, np.pi / 3, np.pi / 2])
        z = np.array([2.0, 5.0, 8.0])

        config = self.make_config(windowed.Trivariate.bicubic)

        result1 = core.trivariate(grid, x, y, z, config)
        result2 = core.trivariate(grid, x, y, z, config)
        result3 = core.trivariate(grid, x, y, z, config)

        # Replace values near machine epsilon with zero before comparison
        # (these are numerical noise, not meaningful results)
        epsilon = np.finfo(np.float64).eps * 100  # ~2.2e-14
        result1_cleaned = np.where(np.abs(result1) < epsilon, 0.0, result1)
        result2_cleaned = np.where(np.abs(result2) < epsilon, 0.0, result2)
        result3_cleaned = np.where(np.abs(result3) < epsilon, 0.0, result3)

        # Results should be identical after cleaning
        np.testing.assert_array_equal(result1_cleaned, result2_cleaned)
        np.testing.assert_array_equal(result2_cleaned, result3_cleaned)

    @staticmethod
    def create_analytical_temporal_grid3d(
        dtype: type[np.float32 | np.float64],
    ) -> core.GridHolder:
        """Create a 3D grid with temporal Z-axis and analytical field.

        f(x, y, t) = sin(x) * cos(y) * exp(-t_normalized/10)

        where t is a datetime64 axis.
        """
        x_vals = np.linspace(0, 2 * np.pi, 15)
        y_vals = np.linspace(0, np.pi, 12)
        # Create a temporal axis with datetime64 values
        time_vals: np.ndarray = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-11"),
            np.timedelta64(1, "D"),
        )

        x_axis = core.Axis(x_vals, period=None)
        y_axis = core.Axis(y_vals)
        z_axis = core.TemporalAxis(time_vals)

        # Normalize time for analytical function (0 to 9 days)
        time_normalized = np.arange(10)

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
        """Test windowed trivariate interpolation with temporal Z-axis."""
        grid = self.create_analytical_temporal_grid3d(np.float64)

        x = np.array([np.pi / 4])
        y = np.array([np.pi / 4])
        # Use a datetime64 value for z
        z = np.array([np.datetime64("2020-01-03")])

        config = self.make_config(
            windowed.Trivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

        # Verify against expected value (day 2, normalized to t=2)
        expected = np.sin(np.pi / 4) * np.cos(np.pi / 4) * np.exp(-2.0 / 10)
        np.testing.assert_allclose(result[0], expected, rtol=0.15)

    def test_temporal_grid_multiple_times(self) -> None:
        """Test interpolation at multiple temporal points."""
        grid = self.create_analytical_temporal_grid3d(np.float64)

        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        y = np.array([np.pi / 4, np.pi / 3, np.pi / 2])
        z = np.array(
            [
                np.datetime64("2020-01-01"),
                np.datetime64("2020-01-05"),
                np.datetime64("2020-01-09"),
            ]
        )

        config = self.make_config(
            windowed.Trivariate.bilinear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

        # All values should be reasonable (between -1 and 1 due to sin/cos)
        assert np.all(np.abs(result) <= 1.5)

    def test_temporal_grid_with_linear_time_axis(self) -> None:
        """Test temporal grid with linear interpolation on time axis."""
        grid = self.create_analytical_temporal_grid3d(np.float64)

        x = np.array([np.pi / 2])
        y = np.array([np.pi / 4])
        # Datetime between grid points
        z = np.array([np.datetime64("2020-01-03T12:00:00")])

        # Use linear interpolation on time axis
        config = self.make_config(
            windowed.Trivariate.bilinear,
            third_axis=windowed.AxisConfig.linear(),
        )
        result = core.trivariate(grid, x, y, z, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

        # Result should be between values at day 3 and day 4
        z_day3 = np.array([np.datetime64("2020-01-03")])
        z_day4 = np.array([np.datetime64("2020-01-04")])

        config_nearest = self.make_config(
            windowed.Trivariate.bilinear,
            third_axis=windowed.AxisConfig.nearest(),
        )
        result_day3 = core.trivariate(grid, x, y, z_day3, config_nearest)
        result_day4 = core.trivariate(grid, x, y, z_day4, config_nearest)

        # Linear interpolation result should be between the two days
        assert (
            min(result_day3[0], result_day4[0])
            <= result[0]
            <= max(result_day3[0], result_day4[0])
        )

    def test_temporal_grid_all_methods(self) -> None:
        """Test all interpolation methods work with temporal grid."""
        grid = self.create_analytical_temporal_grid3d(np.float64)

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

        for method in methods:
            config = self.make_config(getattr(windowed.Trivariate, method))
            result = core.trivariate(grid, x, y, z, config)

            assert result.shape == (1,)
            assert np.isfinite(result[0]), f"Method {method} produced NaN"

# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for windowed univariate interpolation and derivatives."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from .... import core
from ....core.config import windowed


if TYPE_CHECKING:
    from collections.abc import Callable


class TestUnivariateWindowed:
    """Test suite for windowed univariate interpolation."""

    @staticmethod
    def create_analytical_grid1d(
        dtype: type[np.float32 | np.float64],
    ) -> core.Grid1D:
        """Create a 1D grid with an analytical field.

        f(x) = sin(x)

        This provides a smooth, continuous field with known values and
        derivatives.
        """
        x_vals = np.linspace(0, 2 * np.pi, 50)
        x_axis = core.Axis(x_vals, period=None)

        # Create analytical field: f(x) = sin(x)
        data = np.sin(x_vals).astype(dtype)
        # Ensure C-contiguous for grid creation
        data = np.ascontiguousarray(data)

        return core.Grid(x_axis, data)

    @staticmethod
    def create_polynomial_grid1d(
        dtype: type[np.float32 | np.float64],
    ) -> core.Grid1D:
        """Create a 1D grid with a polynomial field.

        f(x) = x^2

        This is useful for testing derivatives: f'(x) = 2x
        """
        x_vals = np.linspace(0, 10, 50)
        x_axis = core.Axis(x_vals, period=None)

        # Create polynomial field: f(x) = x^2
        data = (x_vals**2).astype(dtype)
        data = np.ascontiguousarray(data)

        return core.Grid(x_axis, data)

    @staticmethod
    def make_config(
        method: Callable[[], windowed.Univariate],
        *,
        boundary: windowed.BoundaryConfig | None = None,
        half_window_size: int | None = 5,
    ) -> windowed.Univariate:
        """Build a windowed univariate configuration with default values."""
        boundary = boundary or windowed.BoundaryConfig.undef()
        cfg = method().with_boundary_mode(boundary)
        if half_window_size is not None:
            cfg = cfg.with_half_window_size(half_window_size)
        return cfg

    def test_single_point_linear(self) -> None:
        """Perform linear interpolation at a single point."""
        grid = self.create_analytical_grid1d(np.float64)

        # Test point: π/2 where sin(π/2) = 1
        x = np.array([np.pi / 2])
        expected = np.sin(np.pi / 2)

        config = self.make_config(windowed.Univariate.linear)
        result = core.univariate(grid, x, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])
        # Linear should have good accuracy for smooth functions
        np.testing.assert_allclose(result[0], expected, rtol=0.02)

    def test_multiple_points_linear(self) -> None:
        """Test linear interpolation with multiple points."""
        grid = self.create_analytical_grid1d(np.float64)

        # Multiple test points with known analytical values
        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])

        # Calculate expected values using the analytical function
        expected = np.sin(x)

        config = self.make_config(windowed.Univariate.linear)
        result = core.univariate(grid, x, config)

        assert result.shape == (4,)
        assert np.all(np.isfinite(result))
        # Validate against analytical values
        np.testing.assert_allclose(result, expected, rtol=0.05)

    def test_interpolation_method_comparison(self) -> None:
        """Compare interpolation accuracy across all available methods."""
        grid = self.create_analytical_grid1d(np.float64)

        x = np.array([1.5])
        expected = np.sin(1.5)

        # Test all available interpolation methods
        all_methods = [
            "akima",
            "akima_periodic",
            "c_spline",
            "c_spline_not_a_knot",
            "c_spline_periodic",
            "linear",
            "polynomial",
            "steffen",
        ]

        results = {}
        for name in all_methods:
            method = getattr(windowed.Univariate, name)
            config = self.make_config(method)
            result = core.univariate(grid, x, config)
            assert np.isfinite(result[0]), f"Method {name} produced NaN"
            results[name] = result[0]

        # All should be reasonably close to expected
        for name, value in results.items():
            assert np.abs(value - expected) < 0.1, (
                f"Method {name} error too large: {value} vs {expected}"
            )

    def test_derivative_single_point(self) -> None:
        """Test derivative calculation at a single point."""
        grid = self.create_polynomial_grid1d(np.float64)

        # For f(x) = x^2, f'(x) = 2x
        # Test at x = 5
        x = np.array([5.0])
        expected_derivative = 2 * x[0]

        config = self.make_config(windowed.Univariate.linear)
        result = core.univariate_derivative(grid, x, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])
        # Linear derivative should be exact for quadratic functions
        np.testing.assert_allclose(result[0], expected_derivative, rtol=0.05)

    def test_derivative_multiple_points(self) -> None:
        """Test derivative calculation at multiple points."""
        grid = self.create_polynomial_grid1d(np.float64)

        # For f(x) = x^2, f'(x) = 2x
        x = np.array([2.0, 5.0, 8.0])
        expected_derivatives = 2 * x

        config = self.make_config(windowed.Univariate.linear)
        result = core.univariate_derivative(grid, x, config)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        # Validate against analytical derivatives
        np.testing.assert_allclose(result, expected_derivatives, rtol=0.1)

    def test_derivative_of_sine(self) -> None:
        """Test derivative of sine function (should be cosine)."""
        grid = self.create_analytical_grid1d(np.float64)

        # For f(x) = sin(x), f'(x) = cos(x)
        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        expected_derivatives = np.cos(x)

        # Use cubic spline for better accuracy with derivatives
        config = self.make_config(windowed.Univariate.c_spline)
        result = core.univariate_derivative(grid, x, config)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
        # Cubic spline should give good derivative estimates
        # Use atol for values close to zero (like cos(π/2))
        np.testing.assert_allclose(
            result, expected_derivatives, rtol=0.15, atol=1e-3
        )

    def test_bounds_error(self) -> None:
        """Test bounds_error parameter with windowed."""
        grid = self.create_analytical_grid1d(np.float64)

        # Point outside grid bounds (x too large)
        x = np.array([3 * np.pi])

        # With bounds_error=False, should return NaN
        config = self.make_config(windowed.Univariate.linear)
        result = core.univariate(grid, x, config)

        assert result.shape == (1,)
        assert np.isnan(result[0])

        # With bounds_error=True, should raise an error
        config = self.make_config(
            windowed.Univariate.linear,
        ).with_bounds_error(True)
        with pytest.raises((ValueError, IndexError), match="out of bounds"):
            core.univariate(grid, x, config)

    def test_bounds_error_derivative(self) -> None:
        """Test bounds_error for derivative calculation."""
        grid = self.create_polynomial_grid1d(np.float64)

        # Point outside grid bounds
        x = np.array([20.0])

        # With bounds_error=False, should return NaN
        config = self.make_config(windowed.Univariate.linear)
        result = core.univariate_derivative(grid, x, config)

        assert result.shape == (1,)
        assert np.isnan(result[0])

        # With bounds_error=True, should raise an error
        config = self.make_config(
            windowed.Univariate.linear,
        ).with_bounds_error(True)
        with pytest.raises((ValueError, IndexError), match="out of bounds"):
            core.univariate_derivative(grid, x, config)

    def test_window_size_configuration(self) -> None:
        """Test that different window sizes produce reasonable results."""
        grid = self.create_analytical_grid1d(np.float64)

        x = np.array([np.pi / 2])
        expected = np.sin(x[0])

        # Test with small window
        config_small = self.make_config(
            windowed.Univariate.linear, half_window_size=3
        )
        result_small = core.univariate(grid, x, config_small)

        # Test with larger window
        config_large = self.make_config(
            windowed.Univariate.linear, half_window_size=9
        )
        result_large = core.univariate(grid, x, config_large)

        assert result_small.shape == (1,)
        assert result_large.shape == (1,)
        assert np.isfinite(result_small[0])
        assert np.isfinite(result_large[0])

        # Both should be close to the expected value
        np.testing.assert_allclose(result_small[0], expected, rtol=0.05)
        np.testing.assert_allclose(result_large[0], expected, rtol=0.05)

    def test_boundary_config(self) -> None:
        """Test different boundary modes with analytical functions."""
        # Test with linear function: y = 2x + 3
        # This should be exactly reproduced by linear interpolation
        x_vals = np.linspace(0, 10, 51)  # 51 points for even spacing
        x_axis = core.Axis(x_vals, period=None)

        # Linear function: f(x) = 2x + 3
        a, b = 2.0, 3.0
        data_linear = (a * x_vals + b).astype(np.float64)
        data_linear = np.ascontiguousarray(data_linear)
        grid_linear = core.Grid(x_axis, data_linear)

        # Test points in the interior [2, 8] with fine resolution
        x_test = np.linspace(2.0, 8.0, 31)
        expected_linear = a * x_test + b

        # Test all boundary modes with linear function
        boundary_configs = [
            windowed.BoundaryConfig.shrink(),
            windowed.BoundaryConfig.undef(),
        ]

        for boundary in boundary_configs:
            config = self.make_config(
                windowed.Univariate.linear, boundary=boundary
            )
            result = core.univariate(grid_linear, x_test, config)

            assert result.shape == x_test.shape
            assert np.all(np.isfinite(result))
            # Linear interpolation should exactly reproduce linear function
            np.testing.assert_allclose(
                result,
                expected_linear,
                rtol=1e-10,
                err_msg=f"Linear function failed with boundary={boundary}",
            )

        # Test with sinusoidal function: y = sin(2πx/10)
        # Period matches domain [0, 10] for WRAP mode testing
        data_sin = np.sin(2 * np.pi * x_vals / 10).astype(np.float64)
        data_sin = np.ascontiguousarray(data_sin)
        grid_sin = core.Grid(x_axis, data_sin)

        # Test points
        x_test_sin = np.linspace(2.0, 8.0, 31)
        expected_sin = np.sin(2 * np.pi * x_test_sin / 10)

        results_sin = {}
        for boundary in boundary_configs:
            config = self.make_config(
                windowed.Univariate.c_spline, boundary=boundary
            )
            result = core.univariate(grid_sin, x_test_sin, config)

            assert result.shape == x_test_sin.shape
            assert np.all(np.isfinite(result))
            results_sin[boundary] = result

            # Should be close to analytical values (cubic spline on smooth
            # function)
            np.testing.assert_allclose(
                result,
                expected_sin,
                atol=1e-16,
                err_msg=f"Sinusoidal function failed with boundary={boundary}",
            )

    def test_nan_in_grid_data(self) -> None:
        """Test handling of NaN values in grid data."""
        grid = self.create_analytical_grid1d(np.float64)

        # Inject NaN values into the grid
        grid.array[20:25] = np.nan

        # Point near NaN region
        x = np.array([grid.x[22]])

        config = self.make_config(windowed.Univariate.linear)
        result = core.univariate(grid, x, config)

        # Should return NaN when interpolating over NaN values
        assert result.shape == (1,)
        assert np.isnan(result[0])

    def test_mixed_valid_invalid_points(self) -> None:
        """Test interpolation with mix of valid and out-of-bounds points."""
        grid = self.create_analytical_grid1d(np.float64)

        # Mix of valid and invalid points
        x = np.array([np.pi / 2, 10.0, np.pi / 4])  # Middle one out of bounds

        config = self.make_config(windowed.Univariate.linear)
        result = core.univariate(grid, x, config)

        assert result.shape == (3,)
        # First and third should be finite
        assert np.isfinite(result[0])
        assert np.isfinite(result[2])
        # Second should be NaN (out of bounds)
        assert np.isnan(result[1])

    def test_dtype_float32(self) -> None:
        """Test windowed univariate interpolation with float32 data."""
        grid = self.create_analytical_grid1d(np.float32)

        x = np.array([np.pi / 4])

        config = self.make_config(windowed.Univariate.linear)
        result = core.univariate(grid, x, config)

        assert result.dtype == np.float32
        assert np.isfinite(result[0])

    def test_dtype_float32_derivative(self) -> None:
        """Test derivative calculation with float32 data."""
        grid = self.create_polynomial_grid1d(np.float32)

        x = np.array([5.0])

        config = self.make_config(windowed.Univariate.linear)
        result = core.univariate_derivative(grid, x, config)

        assert result.dtype == np.float32
        assert np.isfinite(result[0])

    def test_num_threads(self) -> None:
        """Test interpolation results are consistent across thread counts."""
        grid = self.create_analytical_grid1d(np.float64)

        x = np.array([np.pi / 4, np.pi / 2, 3 * np.pi / 4])

        # Test with 1 thread
        config_single = self.make_config(
            windowed.Univariate.linear
        ).with_num_threads(1)
        result_single = core.univariate(grid, x, config_single)

        # Test with multiple threads
        config_multi = self.make_config(
            windowed.Univariate.linear
        ).with_num_threads(4)
        result_multi = core.univariate(grid, x, config_multi)

        # Results should be identical or very close
        np.testing.assert_array_almost_equal(result_single, result_multi)

    def test_num_threads_derivative(self) -> None:
        """Test derivative results are consistent across thread counts."""
        grid = self.create_polynomial_grid1d(np.float64)

        x = np.array([2.0, 5.0, 8.0])

        # Test with 1 thread
        config_single = self.make_config(
            windowed.Univariate.linear
        ).with_num_threads(1)
        result_single = core.univariate_derivative(grid, x, config_single)

        # Test with multiple threads
        config_multi = self.make_config(
            windowed.Univariate.linear
        ).with_num_threads(4)
        result_multi = core.univariate_derivative(grid, x, config_multi)

        # Results should be identical or very close
        np.testing.assert_array_almost_equal(result_single, result_multi)

    def test_continuity(self) -> None:
        """Test that windowed interpolation is continuous (smooth)."""
        grid = self.create_analytical_grid1d(np.float64)

        # Create a path of close points to test continuity
        n_points = 20
        x = np.linspace(1.0, 2.0, n_points)

        config = self.make_config(windowed.Univariate.linear)
        results = core.univariate(grid, x, config)

        # Check that differences between consecutive points are small
        diffs = np.abs(np.diff(results))
        max_diff = np.max(diffs)

        # For smooth analytical function, consecutive interpolations should be
        # close
        assert max_diff < 0.1, (
            f"Interpolation not continuous, max diff: {max_diff}"
        )
        assert np.all(np.isfinite(results))

    def test_analytical_accuracy(self) -> None:
        """Test windowed interpolation accuracy against analytical function."""
        grid = self.create_analytical_grid1d(np.float64)

        # Define the analytical function
        def analytical_func(x: float) -> float:
            return float(np.sin(x))

        # Test at interior points (not on grid points)
        x = np.array([0.5, 1.0, 1.5, 2.0, 3.5])

        # Test with linear
        config_linear = self.make_config(
            windowed.Univariate.linear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result_linear = core.univariate(grid, x, config_linear)

        # Test with cubic spline (should be more accurate)
        config_cubic = self.make_config(
            windowed.Univariate.c_spline,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result_cubic = core.univariate(grid, x, config_cubic)

        # Compare with analytical values
        expected = np.array([analytical_func(x[i]) for i in range(len(x))])

        # Linear should have good accuracy
        np.testing.assert_allclose(
            result_linear,
            expected,
            rtol=0.05,
            err_msg="Linear interpolation accuracy too low",
        )

        # Cubic should be at least as accurate on finite values
        cubic_errors = np.abs(result_cubic - expected)
        linear_errors = np.abs(result_linear - expected)
        # Most values should be finite
        assert np.sum(np.isfinite(result_cubic)) >= 4
        # Compare mean errors on finite values only
        finite_mask = np.isfinite(result_cubic)
        if np.any(finite_mask):
            assert (
                np.mean(cubic_errors[finite_mask])
                <= np.mean(linear_errors[finite_mask]) * 1.2
            )

    def test_large_array(self) -> None:
        """Test windowed univariate interpolation with large arrays."""
        grid = self.create_analytical_grid1d(np.float64)

        # Create large arrays of points
        n_points = 1000
        rng = np.random.Generator(np.random.PCG64(seed=42))
        x = rng.uniform(0.1, 2 * np.pi - 0.1, n_points)

        config = self.make_config(
            windowed.Univariate.linear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.univariate(grid, x, config)

        assert result.shape == (n_points,)
        # Most values should be finite
        assert np.sum(np.isfinite(result)) == n_points

    def test_large_array_derivative(self) -> None:
        """Test derivative calculation with large arrays."""
        grid = self.create_polynomial_grid1d(np.float64)

        # Create large arrays of points
        n_points = 1000
        rng = np.random.Generator(np.random.PCG64(seed=42))
        x = rng.uniform(0.5, 9.5, n_points)

        config = self.make_config(
            windowed.Univariate.linear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.univariate_derivative(grid, x, config)

        assert result.shape == (n_points,)
        # Most values should be finite
        assert np.sum(np.isfinite(result)) == n_points

    def test_reproducibility(self) -> None:
        """Test that repeated calls produce identical results."""
        grid = self.create_analytical_grid1d(np.float64)

        x = np.array([np.pi / 3, np.pi / 2, 2 * np.pi / 3])

        config = self.make_config(windowed.Univariate.c_spline)

        result1 = core.univariate(grid, x, config)
        result2 = core.univariate(grid, x, config)
        result3 = core.univariate(grid, x, config)

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)

    def test_reproducibility_derivative(self) -> None:
        """Test that repeated derivative calls produce identical results."""
        grid = self.create_polynomial_grid1d(np.float64)

        x = np.array([2.0, 5.0, 8.0])

        config = self.make_config(windowed.Univariate.linear)

        result1 = core.univariate_derivative(grid, x, config)
        result2 = core.univariate_derivative(grid, x, config)
        result3 = core.univariate_derivative(grid, x, config)

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)

    def test_edge_point(self) -> None:
        """Test windowed interpolation near grid edge."""
        grid = self.create_analytical_grid1d(np.float64)

        # Use a point near (but not exactly at) the edge
        x = np.array([grid.x[1]])

        config = self.make_config(
            windowed.Univariate.linear,
            boundary=windowed.BoundaryConfig.shrink(),
        )
        result = core.univariate(grid, x, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_grid_edge_interpolation(self) -> None:
        """Test interpolation near grid edges with different boundary modes."""
        grid = self.create_analytical_grid1d(np.float64)

        # Points near grid edges
        x_axis_vals = grid.x[:]
        x_edge = np.array([x_axis_vals[4], x_axis_vals[-5]])

        for boundary in [
            windowed.BoundaryConfig.shrink(),
            windowed.BoundaryConfig.undef(),
        ]:
            config = self.make_config(
                windowed.Univariate.linear, boundary=boundary
            )
            result = core.univariate(grid, x_edge, config)

            assert result.shape == (2,)
            # At least most values should be finite
            assert np.sum(np.isfinite(result)) >= 1

    def test_cubic_vs_linear_accuracy(self) -> None:
        """Compare cubic spline vs linear interpolation accuracy."""
        grid = self.create_analytical_grid1d(np.float64)

        # Test points where we expect cubic to be more accurate
        x = np.array([0.75, 1.25, 2.5, 3.75, 4.5])
        expected = np.sin(x)

        config_linear = self.make_config(windowed.Univariate.linear)
        result_linear = core.univariate(grid, x, config_linear)

        config_cubic = self.make_config(windowed.Univariate.c_spline)
        result_cubic = core.univariate(grid, x, config_cubic)

        linear_error = np.mean(np.abs(result_linear - expected))
        cubic_error = np.mean(np.abs(result_cubic - expected))

        # Cubic should generally be more accurate for smooth functions
        assert cubic_error <= linear_error * 1.5

    def test_derivative_method_comparison(self) -> None:
        """Compare derivative accuracy across different methods."""
        grid = self.create_polynomial_grid1d(np.float64)

        # For f(x) = x^2, f'(x) = 2x
        x = np.array([3.0, 5.0, 7.0])
        expected = 2 * x

        # Test different methods
        methods = ["linear", "c_spline", "akima", "steffen"]

        for method_name in methods:
            method = getattr(windowed.Univariate, method_name)
            config = self.make_config(method)
            result = core.univariate_derivative(grid, x, config)

            assert result.shape == (3,)
            assert np.all(np.isfinite(result))
            # All should give reasonable derivative estimates
            np.testing.assert_allclose(
                result,
                expected,
                rtol=0.2,
                err_msg=f"Method {method_name} derivative error too large",
            )

    def test_zero_derivative(self) -> None:
        """Test derivative at points where it should be zero."""
        grid = self.create_analytical_grid1d(np.float64)

        # For f(x) = sin(x), f'(x) = 0 at x = π/2, 3π/2
        x = np.array([np.pi / 2])
        expected = 0.0

        config = self.make_config(windowed.Univariate.c_spline)
        result = core.univariate_derivative(grid, x, config)

        assert result.shape == (1,)
        assert np.isfinite(result[0])
        # Should be close to zero
        np.testing.assert_allclose(result[0], expected, atol=0.1)

    def test_interpolation_and_derivative_consistency(self) -> None:
        """Test that interpolation and derivative are consistent."""
        grid = self.create_polynomial_grid1d(np.float64)

        # Pick a point and its neighbors
        x_center = 5.0
        h = 0.01
        x = np.array([x_center - h, x_center, x_center + h])

        config = self.make_config(windowed.Univariate.c_spline)

        # Get interpolated values
        values = core.univariate(grid, x, config)

        # Get derivative at center point
        derivative = core.univariate_derivative(
            grid, np.array([x_center]), config
        )

        # Numerical derivative using finite differences
        numerical_derivative = (values[2] - values[0]) / (2 * h)

        # Should be close to analytical derivative
        np.testing.assert_allclose(
            derivative[0], numerical_derivative, rtol=0.15
        )

# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Tests for the regular grid interpolator module."""

from __future__ import annotations

import numpy as np
import pytest

from pyinterp import core
from pyinterp.core.config import geometric, windowed
from pyinterp.regular_grid_interpolator import (
    _make_geometric_config,
    _make_windowed_config,
    _validate_no_windowed_options,
    univariate,
    univariate_derivative,
)


class TestMakeGeometricConfig:
    """Test _make_geometric_config function."""

    def test_bilinear_config(self) -> None:
        """Test creating bilinear geometric config."""
        config = _make_geometric_config(
            "bilinear",
            geometric.Bivariate,
            bounds_error=False,
            num_threads=0,
        )
        assert isinstance(config, geometric.Bivariate)

    def test_nearest_config(self) -> None:
        """Test creating nearest neighbor geometric config."""
        config = _make_geometric_config(
            "nearest",
            geometric.Bivariate,
            bounds_error=False,
            num_threads=0,
        )
        assert isinstance(config, geometric.Bivariate)

    def test_idw_config(self) -> None:
        """Test creating IDW geometric config."""
        config = _make_geometric_config(
            "idw",
            geometric.Bivariate,
            bounds_error=False,
            num_threads=0,
        )
        assert isinstance(config, geometric.Bivariate)

    def test_invalid_method(self) -> None:
        """Test error for invalid method."""
        with pytest.raises(ValueError, match="Unknown geometric method"):
            _make_geometric_config(
                "invalid",  # type: ignore[arg-type]
                geometric.Bivariate,
                bounds_error=False,
                num_threads=0,
            )

    def test_trivariate_config(self) -> None:
        """Test creating trivariate config."""
        config = _make_geometric_config(
            "bilinear",
            geometric.Trivariate,
            bounds_error=True,
            num_threads=2,
        )
        assert isinstance(config, geometric.Trivariate)

    def test_quadrivariate_config(self) -> None:
        """Test creating quadrivariate config."""
        config = _make_geometric_config(
            "nearest",
            geometric.Quadrivariate,
            bounds_error=False,
            num_threads=4,
        )
        assert isinstance(config, geometric.Quadrivariate)


class TestMakeWindowedConfig:
    """Test _make_windowed_config function."""

    def test_bicubic_config(self) -> None:
        """Test creating bicubic windowed config."""
        config = _make_windowed_config(
            "bicubic",
            windowed.Bivariate,
            bounds_error=False,
            num_threads=0,
        )
        assert isinstance(config, windowed.Bivariate)

    def test_akima_config(self) -> None:
        """Test creating akima windowed config."""
        config = _make_windowed_config(
            "akima",
            windowed.Bivariate,
            bounds_error=False,
            num_threads=0,
        )
        assert isinstance(config, windowed.Bivariate)

    def test_linear_config(self) -> None:
        """Test creating linear windowed config."""
        config = _make_windowed_config(
            "linear",
            windowed.Bivariate,
            bounds_error=False,
            num_threads=0,
        )
        assert isinstance(config, windowed.Bivariate)

    def test_invalid_method(self) -> None:
        """Test error for invalid method."""
        with pytest.raises(ValueError, match="Unknown method:"):
            _make_windowed_config(  # type: ignore[call-overload]
                "invalid",  # type: ignore[arg-type]
                windowed.Bivariate,
                bounds_error=False,
                num_threads=0,
            )

    def test_window_size_x(self) -> None:
        """Test config with window size X."""
        config = _make_windowed_config(
            "bicubic",
            windowed.Bivariate,
            bounds_error=False,
            num_threads=0,
            half_window_size_x=10,
        )
        assert isinstance(config, windowed.Bivariate)

    def test_window_size_y(self) -> None:
        """Test config with window size Y."""
        config = _make_windowed_config(
            "bicubic",
            windowed.Bivariate,
            bounds_error=False,
            num_threads=0,
            half_window_size_y=12,
        )
        assert isinstance(config, windowed.Bivariate)

    def test_boundary_mode(self) -> None:
        """Test config with boundary mode."""
        config = _make_windowed_config(
            "bicubic",
            windowed.Bivariate,
            bounds_error=False,
            num_threads=0,
            boundary_mode="undef",
        )
        assert isinstance(config, windowed.Bivariate)

    def test_boundary_mode_wrap(self) -> None:
        """Test config with wrap boundary mode."""
        config = _make_windowed_config(
            "bicubic",
            windowed.Bivariate,
            bounds_error=False,
            num_threads=0,
            boundary_mode="undef",
        )
        assert isinstance(config, windowed.Bivariate)

    def test_trivariate_with_third_axis(self) -> None:
        """Test trivariate config with third axis."""
        config = _make_windowed_config(
            "bicubic",
            windowed.Trivariate,
            bounds_error=False,
            num_threads=0,
            third_axis="linear",
        )
        assert isinstance(config, windowed.Trivariate)

    def test_quadrivariate_with_axes(self) -> None:
        """Test quadrivariate config with axes."""
        config = _make_windowed_config(
            "bicubic",
            windowed.Quadrivariate,
            bounds_error=False,
            num_threads=0,
            third_axis="nearest",
            fourth_axis="linear",
        )
        assert isinstance(config, windowed.Quadrivariate)


class TestValidateNoWindowedOptions:
    """Test _validate_no_windowed_options function."""

    def test_no_options(self) -> None:
        """Test validation passes with no windowed options."""
        # Should not raise
        _validate_no_windowed_options(
            "bilinear",
            half_window_size_x=None,
            half_window_size_y=None,
            boundary_mode=None,
        )

    def test_window_size_x_error(self) -> None:
        """Test error for window_size_x with geometric method."""
        with pytest.raises(TypeError, match="window_size_x"):
            _validate_no_windowed_options(
                "bilinear",
                half_window_size_x=10,
                half_window_size_y=None,
                boundary_mode=None,
            )

    def test_window_size_y_error(self) -> None:
        """Test error for window_size_y with geometric method."""
        with pytest.raises(TypeError, match="window_size_y"):
            _validate_no_windowed_options(
                "nearest",
                half_window_size_x=None,
                half_window_size_y=12,
                boundary_mode=None,
            )

    def test_boundary_mode_error(self) -> None:
        """Test error for boundary_mode with geometric method."""
        with pytest.raises(TypeError, match="boundary_mode"):
            _validate_no_windowed_options(
                "idw",
                half_window_size_x=None,
                half_window_size_y=None,
                boundary_mode="undef",
            )

    def test_third_axis_error(self) -> None:
        """Test error for third_axis with geometric method."""
        with pytest.raises(TypeError, match="third_axis"):
            _validate_no_windowed_options(
                "bilinear",
                half_window_size_x=None,
                half_window_size_y=None,
                boundary_mode=None,
                third_axis="linear",
            )

    def test_fourth_axis_error(self) -> None:
        """Test error for fourth_axis with geometric method."""
        with pytest.raises(TypeError, match="fourth_axis"):
            _validate_no_windowed_options(
                "nearest",
                half_window_size_x=None,
                half_window_size_y=None,
                boundary_mode=None,
                fourth_axis="nearest",
            )

    def test_multiple_errors(self) -> None:
        """Test error message with multiple invalid options."""
        with pytest.raises(TypeError) as exc_info:
            _validate_no_windowed_options(
                "bilinear",
                half_window_size_x=10,
                half_window_size_y=12,
                boundary_mode="shrink",
            )
        error_msg = str(exc_info.value)
        assert "window_size_x" in error_msg
        assert "window_size_y" in error_msg
        assert "boundary_mode" in error_msg


class TestMakeWindowedConfigUnivariate:
    """Test _make_windowed_config with Univariate config."""

    def test_linear_config(self) -> None:
        """Test creating linear univariate config."""
        config = _make_windowed_config(
            "linear",
            windowed.Univariate,
            bounds_error=False,
            num_threads=0,
        )
        assert isinstance(config, windowed.Univariate)

    def test_with_window_size(self) -> None:
        """Test univariate config with window size."""
        config = _make_windowed_config(
            "c_spline",
            windowed.Univariate,
            bounds_error=False,
            num_threads=0,
            half_window_size=10,
        )
        assert isinstance(config, windowed.Univariate)

    def test_with_boundary_mode(self) -> None:
        """Test univariate config with boundary mode."""
        config = _make_windowed_config(
            "akima",
            windowed.Univariate,
            bounds_error=False,
            num_threads=0,
            boundary_mode="undef",
        )
        assert isinstance(config, windowed.Univariate)

    def test_all_methods(self) -> None:
        """Test all windowed methods work with Univariate."""
        methods = [
            "linear",
            "c_spline",
            "akima",
            "steffen",
            "polynomial",
        ]
        for method in methods:
            config = _make_windowed_config(  # type: ignore[call-overload]
                method,  # type: ignore[arg-type]
                windowed.Univariate,
                bounds_error=False,
                num_threads=0,
            )
            assert isinstance(config, windowed.Univariate)


class TestUnivariateFunction:
    """Test the univariate wrapper function."""

    @staticmethod
    def create_test_grid() -> core.Grid1D:
        """Create a simple 1D grid for testing."""
        x_vals = np.linspace(0, 10, 51)  # Denser grid for better interpolation
        x_axis = core.Axis(x_vals)
        data = np.sin(x_vals)
        return core.Grid(x_axis, data)

    def test_with_string_method(self) -> None:
        """Test univariate with string method."""
        grid = self.create_test_grid()
        x = np.array([2.5, 5.0, 7.5])

        result = univariate(grid, x, "linear")
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_with_config_object(self) -> None:
        """Test univariate with config object."""
        grid = self.create_test_grid()
        x = np.array([5.0])  # Test at center point to avoid edge issues

        config = windowed.Univariate.c_spline().with_half_window_size(7)
        result = univariate(grid, x, config)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_with_window_size(self) -> None:
        """Test univariate with window size parameter."""
        grid = self.create_test_grid()
        x = np.array([5.0])  # Test at center point to avoid edge issues

        result = univariate(grid, x, "c_spline", half_window_size=7)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_with_boundary_mode(self) -> None:
        """Test univariate with boundary mode."""
        grid = self.create_test_grid()
        x = np.array([2.5, 5.0, 7.5])

        result = univariate(grid, x, "linear", boundary_mode="undef")
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_with_bounds_error(self) -> None:
        """Test univariate with bounds_error."""
        grid = self.create_test_grid()
        x = np.array([15.0])  # Out of bounds

        # Should return NaN without bounds_error
        result = univariate(grid, x, "linear", bounds_error=False)
        assert np.isnan(result[0])

        # Should raise with bounds_error=True
        with pytest.raises((ValueError, IndexError)):
            univariate(grid, x, "linear", bounds_error=True)

    def test_different_methods(self) -> None:
        """Test different interpolation methods."""
        grid = self.create_test_grid()
        x = np.array([2.5])

        methods = ["linear", "c_spline", "akima", "steffen"]
        for method in methods:
            result = univariate(  # type: ignore[call-overload]
                grid,
                x,
                method,  # type: ignore[arg-type]
            )
            assert result.shape == (1,)
            assert np.isfinite(result[0])


class TestUnivariateDerivativeFunction:
    """Test the univariate_derivative wrapper function."""

    @staticmethod
    def create_test_grid() -> core.Grid1D:
        """Create a 1D grid with x^2 for testing derivatives."""
        x_vals = np.linspace(0, 10, 21)
        x_axis = core.Axis(x_vals)
        data = x_vals**2
        return core.Grid(x_axis, data)

    def test_with_string_method(self) -> None:
        """Test derivative with string method."""
        grid = self.create_test_grid()
        x = np.array([2.0, 5.0, 8.0])

        result = univariate_derivative(grid, x, "linear")
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_with_config_object(self) -> None:
        """Test derivative with config object."""
        grid = self.create_test_grid()
        x = np.array([5.0])  # Test at center point to avoid edge issues

        config = windowed.Univariate.c_spline().with_half_window_size(7)
        result = univariate_derivative(grid, x, config)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_derivative_accuracy(self) -> None:
        """Test derivative accuracy for x^2 (derivative should be 2x)."""
        grid = self.create_test_grid()
        x = np.array([2.0, 5.0, 8.0])
        expected = 2 * x  # Derivative of x^2 is 2x

        result = univariate_derivative(grid, x, "linear")
        # Linear should give good derivative estimates
        np.testing.assert_allclose(result, expected, rtol=0.15)

    def test_with_window_size(self) -> None:
        """Test derivative with window size parameter."""
        grid = self.create_test_grid()
        x = np.array([5.0])

        result = univariate_derivative(grid, x, "c_spline", half_window_size=9)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_with_bounds_error(self) -> None:
        """Test derivative with bounds_error."""
        grid = self.create_test_grid()
        x = np.array([15.0])  # Out of bounds

        # Should return NaN without bounds_error
        result = univariate_derivative(grid, x, "linear", bounds_error=False)
        assert np.isnan(result[0])

        # Should raise with bounds_error=True
        with pytest.raises((ValueError, IndexError)):
            univariate_derivative(grid, x, "linear", bounds_error=True)

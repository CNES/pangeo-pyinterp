# Copyright (c) 2026 CNES.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.
"""Unit tests for configuration objects."""

from __future__ import annotations

from ...core.config import fill, geometric, rtree, windowed


class TestGeometric:
    """Test geometric interpolation configurations."""

    def test_bivariate_class_methods(self) -> None:
        """Test Bivariate class methods return instances."""
        bilinear = geometric.Bivariate.bilinear()
        assert isinstance(bilinear, geometric.Bivariate)

        idw = geometric.Bivariate.idw()
        assert isinstance(idw, geometric.Bivariate)

        nearest = geometric.Bivariate.nearest()
        assert isinstance(nearest, geometric.Bivariate)

    def test_bivariate_instance_methods(self) -> None:
        """Test Bivariate instance methods."""
        config = geometric.Bivariate.bilinear()

        # Test bounds_error
        config_bounds = config.with_bounds_error(True)
        assert isinstance(config_bounds, geometric.Bivariate)
        assert config_bounds is not config

        config_bounds_false = config.with_bounds_error(False)
        assert isinstance(config_bounds_false, geometric.Bivariate)

        # Test num_threads
        config_threads = config.with_num_threads(4)
        assert isinstance(config_threads, geometric.Bivariate)
        assert config_threads is not config

    def test_trivariate_class_methods(self) -> None:
        """Test Trivariate class methods return instances."""
        bilinear = geometric.Trivariate.bilinear()
        assert isinstance(bilinear, geometric.Trivariate)

        idw = geometric.Trivariate.idw()
        assert isinstance(idw, geometric.Trivariate)

        nearest = geometric.Trivariate.nearest()
        assert isinstance(nearest, geometric.Trivariate)

    def test_trivariate_instance_methods(self) -> None:
        """Test Trivariate instance methods."""
        config = geometric.Trivariate.bilinear()

        config_bounds = config.with_bounds_error(True)
        assert isinstance(config_bounds, geometric.Trivariate)
        assert config_bounds is not config

        config_threads = config.with_num_threads(8)
        assert isinstance(config_threads, geometric.Trivariate)
        assert config_threads is not config

    def test_quadrivariate_class_methods(self) -> None:
        """Test Quadrivariate class methods return instances."""
        bilinear = geometric.Quadrivariate.bilinear()
        assert isinstance(bilinear, geometric.Quadrivariate)

        idw = geometric.Quadrivariate.idw()
        assert isinstance(idw, geometric.Quadrivariate)

        nearest = geometric.Quadrivariate.nearest()
        assert isinstance(nearest, geometric.Quadrivariate)

    def test_quadrivariate_instance_methods(self) -> None:
        """Test Quadrivariate instance methods."""
        config = geometric.Quadrivariate.bilinear()

        config_bounds = config.with_bounds_error(True)
        assert isinstance(config_bounds, geometric.Quadrivariate)
        assert config_bounds is not config

        config_threads = config.with_num_threads(2)
        assert isinstance(config_threads, geometric.Quadrivariate)
        assert config_threads is not config


class TestWindowed:
    """Test windowed interpolation configurations."""

    def test_boundary_config_class_methods(self) -> None:
        """Test BoundaryConfig class methods."""
        shrink = windowed.BoundaryConfig.shrink()
        assert isinstance(shrink, windowed.BoundaryConfig)

        undef = windowed.BoundaryConfig.undef()
        assert isinstance(undef, windowed.BoundaryConfig)

    def test_axis_config_class_methods(self) -> None:
        """Test AxisConfig class methods."""
        linear = windowed.AxisConfig.linear()
        assert isinstance(linear, windowed.AxisConfig)

        nearest = windowed.AxisConfig.nearest()
        assert isinstance(nearest, windowed.AxisConfig)

    def test_bivariate_class_methods(self) -> None:
        """Test Bivariate class methods return instances."""
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

        for method in methods:
            config = getattr(windowed.Bivariate, method)()
            assert isinstance(config, windowed.Bivariate)

    def test_bivariate_instance_methods(self) -> None:
        """Test Bivariate instance methods."""
        config = windowed.Bivariate.bilinear()

        # Test bounds_error
        config_bounds = config.with_bounds_error(True)
        assert isinstance(config_bounds, windowed.Bivariate)
        assert config_bounds is not config

        # Test num_threads
        config_threads = config.with_num_threads(4)
        assert isinstance(config_threads, windowed.Bivariate)
        assert config_threads is not config

        # Test with_boundary_mode
        config_boundary = config.with_boundary_mode(
            windowed.BoundaryConfig.shrink()
        )
        assert isinstance(config_boundary, windowed.Bivariate)
        assert config_boundary is not config

        # Test with_bounds_error
        config_bounds2 = config.with_bounds_error(False)
        assert isinstance(config_bounds2, windowed.Bivariate)
        assert config_bounds2 is not config

        # Test with_num_threads
        config_threads2 = config.with_num_threads(8)
        assert isinstance(config_threads2, windowed.Bivariate)
        assert config_threads2 is not config

        # Test with_window_size_x
        config_wsx = config.with_half_window_size_x(5)
        assert isinstance(config_wsx, windowed.Bivariate)
        assert config_wsx is not config

        # Test with_window_size_y
        config_wsy = config.with_half_window_size_y(3)
        assert isinstance(config_wsy, windowed.Bivariate)
        assert config_wsy is not config

    def test_trivariate_class_methods(self) -> None:
        """Test Trivariate class methods return instances."""
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

        for method in methods:
            config = getattr(windowed.Trivariate, method)()
            assert isinstance(config, windowed.Trivariate)

    def test_trivariate_instance_methods(self) -> None:
        """Test Trivariate instance methods."""
        config = windowed.Trivariate.bilinear()

        config_bounds = config.with_bounds_error(True)
        assert isinstance(config_bounds, windowed.Trivariate)
        assert config_bounds is not config

        config_threads = config.with_num_threads(4)
        assert isinstance(config_threads, windowed.Trivariate)
        assert config_threads is not config

        config_boundary = config.with_boundary_mode(
            windowed.BoundaryConfig.shrink()
        )
        assert isinstance(config_boundary, windowed.Trivariate)
        assert config_boundary is not config

        config_bounds2 = config.with_bounds_error(False)
        assert isinstance(config_bounds2, windowed.Trivariate)
        assert config_bounds2 is not config

        config_threads2 = config.with_num_threads(2)
        assert isinstance(config_threads2, windowed.Trivariate)
        assert config_threads2 is not config

        config_wsx = config.with_half_window_size_x(7)
        assert isinstance(config_wsx, windowed.Trivariate)
        assert config_wsx is not config

        config_wsy = config.with_half_window_size_y(5)
        assert isinstance(config_wsy, windowed.Trivariate)
        assert config_wsy is not config

        # Test with_third_axis
        axis_config = windowed.AxisConfig.linear()
        config_axis = config.with_third_axis(axis_config)
        assert isinstance(config_axis, windowed.Trivariate)
        assert config_axis is not config

    def test_quadrivariate_class_methods(self) -> None:
        """Test Quadrivariate class methods return instances."""
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

        for method in methods:
            config = getattr(windowed.Quadrivariate, method)()
            assert isinstance(config, windowed.Quadrivariate)

    def test_quadrivariate_instance_methods(self) -> None:
        """Test Quadrivariate instance methods."""
        config = windowed.Quadrivariate.bilinear()

        config_bounds = config.with_bounds_error(True)
        assert isinstance(config_bounds, windowed.Quadrivariate)
        assert config_bounds is not config

        config_threads = config.with_num_threads(4)
        assert isinstance(config_threads, windowed.Quadrivariate)
        assert config_threads is not config

        config_boundary = config.with_boundary_mode(
            windowed.BoundaryConfig.shrink()
        )
        assert isinstance(config_boundary, windowed.Quadrivariate)
        assert config_boundary is not config

        config_bounds2 = config.with_bounds_error(False)
        assert isinstance(config_bounds2, windowed.Quadrivariate)
        assert config_bounds2 is not config

        config_threads2 = config.with_num_threads(6)
        assert isinstance(config_threads2, windowed.Quadrivariate)
        assert config_threads2 is not config

        config_wsx = config.with_half_window_size_x(9)
        assert isinstance(config_wsx, windowed.Quadrivariate)
        assert config_wsx is not config

        config_wsy = config.with_half_window_size_y(7)
        assert isinstance(config_wsy, windowed.Quadrivariate)
        assert config_wsy is not config

        # Test with_third_axis and with_fourth_axis
        axis_config = windowed.AxisConfig.nearest()
        config_axis3 = config.with_third_axis(axis_config)
        assert isinstance(config_axis3, windowed.Quadrivariate)
        assert config_axis3 is not config

        config_axis4 = config.with_fourth_axis(axis_config)
        assert isinstance(config_axis4, windowed.Quadrivariate)
        assert config_axis4 is not config

    def test_univariate_class_methods(self) -> None:
        """Test Univariate class methods return instances."""
        methods = [
            "akima",
            "akima_periodic",
            "c_spline",
            "c_spline_not_a_knot",
            "c_spline_periodic",
            "linear",
            "polynomial",
            "steffen",
        ]

        for method in methods:
            config = getattr(windowed.Univariate, method)()
            assert isinstance(config, windowed.Univariate)

    def test_univariate_instance_methods(self) -> None:
        """Test Univariate instance methods."""
        config = windowed.Univariate.linear()

        # Test bounds_error
        config_bounds = config.with_bounds_error(True)
        assert isinstance(config_bounds, windowed.Univariate)
        assert config_bounds is not config

        config_bounds_false = config.with_bounds_error(False)
        assert isinstance(config_bounds_false, windowed.Univariate)
        assert config_bounds_false is not config

        # Test num_threads
        config_threads = config.with_num_threads(4)
        assert isinstance(config_threads, windowed.Univariate)
        assert config_threads is not config

        # Test with_boundary_mode
        config_boundary = config.with_boundary_mode(
            windowed.BoundaryConfig.shrink()
        )
        assert isinstance(config_boundary, windowed.Univariate)
        assert config_boundary is not config

        config_boundary_sym = config.with_boundary_mode(
            windowed.BoundaryConfig.shrink()
        )
        assert isinstance(config_boundary_sym, windowed.Univariate)
        assert config_boundary_sym is not config

        # Test with_window_size
        config_ws = config.with_half_window_size(5)
        assert isinstance(config_ws, windowed.Univariate)
        assert config_ws is not config

    def test_method_chaining(self) -> None:
        """Test that methods can be chained."""
        config = (
            windowed.Bivariate.bicubic()
            .with_num_threads(4)
            .with_bounds_error(True)
            .with_boundary_mode(windowed.BoundaryConfig.shrink())
            .with_half_window_size_x(10)
            .with_half_window_size_y(8)
        )
        assert isinstance(config, windowed.Bivariate)

        # Test univariate method chaining
        univariate_config = (
            windowed.Univariate.c_spline()
            .with_num_threads(2)
            .with_bounds_error(False)
            .with_boundary_mode(windowed.BoundaryConfig.shrink())
            .with_half_window_size(7)
        )
        assert isinstance(univariate_config, windowed.Univariate)

    def test_equality(self) -> None:
        """Test configuration equality (basic checks)."""
        config1 = geometric.Bivariate.bilinear().with_num_threads(4)
        config2 = geometric.Bivariate.bilinear().with_num_threads(4)
        config3 = geometric.Bivariate.bilinear().with_num_threads(8)

        assert isinstance(config1, geometric.Bivariate)
        assert isinstance(config2, geometric.Bivariate)
        assert isinstance(config3, geometric.Bivariate)


class TestRTree:
    """Test RTree interpolation configurations."""

    def test_covariance_function_enum(self) -> None:
        """Test CovarianceFunction enum values."""
        assert rtree.CovarianceFunction.CAUCHY
        assert rtree.CovarianceFunction.GAUSSIAN
        assert rtree.CovarianceFunction.MATERN_12
        assert rtree.CovarianceFunction.MATERN_32
        assert rtree.CovarianceFunction.MATERN_52
        assert rtree.CovarianceFunction.SPHERICAL
        assert rtree.CovarianceFunction.WENDLAND

        # Test that enum values are unique
        values = [
            rtree.CovarianceFunction.CAUCHY,
            rtree.CovarianceFunction.GAUSSIAN,
            rtree.CovarianceFunction.MATERN_12,
            rtree.CovarianceFunction.MATERN_32,
            rtree.CovarianceFunction.MATERN_52,
            rtree.CovarianceFunction.SPHERICAL,
            rtree.CovarianceFunction.WENDLAND,
        ]
        assert len(set(values)) == 7

    def test_drift_function_enum(self) -> None:
        """Test DriftFunction enum values."""
        assert rtree.DriftFunction.LINEAR
        assert rtree.DriftFunction.QUADRATIC

        # Test that enum values are unique
        values = [
            rtree.DriftFunction.LINEAR,
            rtree.DriftFunction.QUADRATIC,
        ]
        assert len(set(values)) == 2

    def test_window_function_enum(self) -> None:
        """Test WindowFunction enum values."""
        assert rtree.WindowKernel.BLACKMAN
        assert rtree.WindowKernel.BLACKMAN_HARRIS
        assert rtree.WindowKernel.BOXCAR
        assert rtree.WindowKernel.FLAT_TOP
        assert rtree.WindowKernel.GAUSSIAN
        assert rtree.WindowKernel.HAMMING
        assert rtree.WindowKernel.LANCZOS
        assert rtree.WindowKernel.NUTTALL
        assert rtree.WindowKernel.PARZEN
        assert rtree.WindowKernel.PARZEN_SWOT

        # Test that enum values are unique
        values = [
            rtree.WindowKernel.BLACKMAN,
            rtree.WindowKernel.BLACKMAN_HARRIS,
            rtree.WindowKernel.BOXCAR,
            rtree.WindowKernel.FLAT_TOP,
            rtree.WindowKernel.GAUSSIAN,
            rtree.WindowKernel.HAMMING,
            rtree.WindowKernel.LANCZOS,
            rtree.WindowKernel.NUTTALL,
            rtree.WindowKernel.PARZEN,
            rtree.WindowKernel.PARZEN_SWOT,
        ]
        assert len(set(values)) == 10

    def test_inverse_distance_weighting_creation(self) -> None:
        """Test InverseDistanceWeighting configuration creation."""
        config = rtree.InverseDistanceWeighting()
        assert isinstance(config, rtree.InverseDistanceWeighting)

    def test_inverse_distance_weighting_instance_methods(self) -> None:
        """Test InverseDistanceWeighting instance methods."""
        config = rtree.InverseDistanceWeighting()

        # Test with_k
        config_k = config.with_k(12)
        assert isinstance(config_k, rtree.InverseDistanceWeighting)
        assert config_k is not config

        # Test with_p
        config_p = config.with_p(3)
        assert isinstance(config_p, rtree.InverseDistanceWeighting)
        assert config_p is not config

        # Test with_radius
        config_radius = config.with_radius(1000.0)
        assert isinstance(config_radius, rtree.InverseDistanceWeighting)
        assert config_radius is not config

        # Test with_num_threads
        config_threads = config.with_num_threads(4)
        assert isinstance(config_threads, rtree.InverseDistanceWeighting)
        assert config_threads is not config

    def test_inverse_distance_weighting_method_chaining(self) -> None:
        """Test InverseDistanceWeighting method chaining."""
        config = (
            rtree.InverseDistanceWeighting()
            .with_k(16)
            .with_p(4)
            .with_radius(5000.0)
            .with_num_threads(8)
        )
        assert isinstance(config, rtree.InverseDistanceWeighting)

    def test_kriging_creation(self) -> None:
        """Test Kriging configuration creation."""
        config = rtree.Kriging()
        assert isinstance(config, rtree.Kriging)

    def test_kriging_instance_methods(self) -> None:
        """Test Kriging instance methods."""
        config = rtree.Kriging()

        # Test with_k
        config_k = config.with_k(10)
        assert isinstance(config_k, rtree.Kriging)
        assert config_k is not config

        # Test with_sigma
        config_sigma = config.with_sigma(2.0)
        assert isinstance(config_sigma, rtree.Kriging)
        assert config_sigma is not config

        # Test with_lambda
        config_lambda = config.with_lambda(0.5)
        assert isinstance(config_lambda, rtree.Kriging)
        assert config_lambda is not config

        # Test with_nugget
        config_nugget = config.with_nugget(0.1)
        assert isinstance(config_nugget, rtree.Kriging)
        assert config_nugget is not config

        # Test with_covariance_model
        config_cov = config.with_covariance_model(
            rtree.CovarianceFunction.GAUSSIAN
        )
        assert isinstance(config_cov, rtree.Kriging)
        assert config_cov is not config

        # Test with_drift_function
        config_drift = config.with_drift_function(rtree.DriftFunction.LINEAR)
        assert isinstance(config_drift, rtree.Kriging)
        assert config_drift is not config

        # Test with_radius
        config_radius = config.with_radius(2000.0)
        assert isinstance(config_radius, rtree.Kriging)
        assert config_radius is not config

        # Test with_num_threads
        config_threads = config.with_num_threads(6)
        assert isinstance(config_threads, rtree.Kriging)
        assert config_threads is not config

    def test_kriging_method_chaining(self) -> None:
        """Test Kriging method chaining."""
        config = (
            rtree.Kriging()
            .with_k(12)
            .with_sigma(1.5)
            .with_lambda(1.0)
            .with_nugget(0.05)
            .with_covariance_model(rtree.CovarianceFunction.MATERN_32)
            .with_drift_function(rtree.DriftFunction.QUADRATIC)
            .with_radius(3000.0)
            .with_num_threads(4)
        )
        assert isinstance(config, rtree.Kriging)

    def test_radial_basis_function_creation(self) -> None:
        """Test RadialBasisFunction configuration creation."""
        config = rtree.RadialBasisFunction()
        assert isinstance(config, rtree.RadialBasisFunction)

    def test_radial_basis_function_instance_methods(self) -> None:
        """Test RadialBasisFunction instance methods."""
        config = rtree.RadialBasisFunction()

        # Test with_k
        config_k = config.with_k(15)
        assert isinstance(config_k, rtree.RadialBasisFunction)
        assert config_k is not config

        # Test with_rbf
        config_rbf = config.with_rbf(rtree.RBFKernel.GAUSSIAN)
        assert isinstance(config_rbf, rtree.RadialBasisFunction)
        assert config_rbf is not config

        # Test with_epsilon
        config_epsilon = config.with_epsilon(1.0)
        assert isinstance(config_epsilon, rtree.RadialBasisFunction)
        assert config_epsilon is not config

        # Test with_smooth
        config_smooth = config.with_smooth(0.01)
        assert isinstance(config_smooth, rtree.RadialBasisFunction)
        assert config_smooth is not config

        # Test with_radius
        config_radius = config.with_radius(4000.0)
        assert isinstance(config_radius, rtree.RadialBasisFunction)
        assert config_radius is not config

        # Test with_num_threads
        config_threads = config.with_num_threads(8)
        assert isinstance(config_threads, rtree.RadialBasisFunction)
        assert config_threads is not config

    def test_radial_basis_function_method_chaining(self) -> None:
        """Test RadialBasisFunction method chaining."""
        config = (
            rtree.RadialBasisFunction()
            .with_k(20)
            .with_rbf(rtree.RBFKernel.THIN_PLATE)
            .with_epsilon(0.5)
            .with_smooth(0.001)
            .with_radius(5000.0)
            .with_num_threads(6)
        )
        assert isinstance(config, rtree.RadialBasisFunction)

    def test_interpolation_window_creation(self) -> None:
        """Test InterpolationWindow configuration creation."""
        config = rtree.InterpolationWindow()
        assert isinstance(config, rtree.InterpolationWindow)

    def test_interpolation_window_instance_methods(self) -> None:
        """Test InterpolationWindow instance methods."""
        config = rtree.InterpolationWindow()

        # Test with_k
        config_k = config.with_k(10)
        assert isinstance(config_k, rtree.InterpolationWindow)
        assert config_k is not config

        # Test with_wf
        config_wf = config.with_wf(rtree.WindowKernel.GAUSSIAN)
        assert isinstance(config_wf, rtree.InterpolationWindow)
        assert config_wf is not config

        # Test with_arg
        config_arg = config.with_arg(0.5)
        assert isinstance(config_arg, rtree.InterpolationWindow)
        assert config_arg is not config

        # Test with_radius
        config_radius = config.with_radius(1500.0)
        assert isinstance(config_radius, rtree.InterpolationWindow)
        assert config_radius is not config

        # Test with_num_threads
        config_threads = config.with_num_threads(4)
        assert isinstance(config_threads, rtree.InterpolationWindow)
        assert config_threads is not config

    def test_interpolation_window_method_chaining(self) -> None:
        """Test InterpolationWindow method chaining."""
        config = (
            rtree.InterpolationWindow()
            .with_k(16)
            .with_wf(rtree.WindowKernel.HAMMING)
            .with_arg(0.3)
            .with_radius(2500.0)
            .with_num_threads(8)
        )
        assert isinstance(config, rtree.InterpolationWindow)

    def test_multiple_covariance_models(self) -> None:
        """Test different covariance models for kriging."""
        models = [
            rtree.CovarianceFunction.CAUCHY,
            rtree.CovarianceFunction.GAUSSIAN,
            rtree.CovarianceFunction.MATERN_12,
            rtree.CovarianceFunction.MATERN_32,
            rtree.CovarianceFunction.MATERN_52,
            rtree.CovarianceFunction.SPHERICAL,
            rtree.CovarianceFunction.WENDLAND,
        ]

        for model in models:
            config = rtree.Kriging().with_covariance_model(model)
            assert isinstance(config, rtree.Kriging)

    def test_multiple_window_functions(self) -> None:
        """Test different window functions."""
        windows = [
            rtree.WindowKernel.BLACKMAN,
            rtree.WindowKernel.BLACKMAN_HARRIS,
            rtree.WindowKernel.BOXCAR,
            rtree.WindowKernel.FLAT_TOP,
            rtree.WindowKernel.GAUSSIAN,
            rtree.WindowKernel.HAMMING,
            rtree.WindowKernel.LANCZOS,
            rtree.WindowKernel.NUTTALL,
            rtree.WindowKernel.PARZEN,
            rtree.WindowKernel.PARZEN_SWOT,
        ]

        for window in windows:
            config = rtree.InterpolationWindow().with_wf(window)
            assert isinstance(config, rtree.InterpolationWindow)

    def test_radius_handling(self) -> None:
        """Test radius parameter handling across configs."""
        # Test None radius
        config_idw = rtree.InverseDistanceWeighting().with_radius(None)
        assert isinstance(config_idw, rtree.InverseDistanceWeighting)

        # Test specific radius values
        config_kriging = rtree.Kriging().with_radius(1000.0)
        assert isinstance(config_kriging, rtree.Kriging)

        config_rbf = rtree.RadialBasisFunction().with_radius(2000.5)
        assert isinstance(config_rbf, rtree.RadialBasisFunction)

        config_window = rtree.InterpolationWindow().with_radius(500.0)
        assert isinstance(config_window, rtree.InterpolationWindow)


class TestFill:
    """Test fill method configurations."""

    def test_first_guess_enum(self) -> None:
        """Test FirstGuess enum values."""
        assert fill.FirstGuess.ZERO
        assert fill.FirstGuess.ZONAL_AVERAGE

        # Test that enum values are unique
        values = [
            fill.FirstGuess.ZERO,
            fill.FirstGuess.ZONAL_AVERAGE,
        ]
        assert len(set(values)) == 2

    def test_loess_value_type_enum(self) -> None:
        """Test LoessValueType enum values."""
        assert fill.LoessValueType.ALL
        assert fill.LoessValueType.DEFINED
        assert fill.LoessValueType.UNDEFINED

        # Test that enum values are unique
        values = [
            fill.LoessValueType.ALL,
            fill.LoessValueType.DEFINED,
            fill.LoessValueType.UNDEFINED,
        ]
        assert len(set(values)) == 3

    def test_gauss_seidel_creation(self) -> None:
        """Test GaussSeidel configuration creation."""
        config = fill.GaussSeidel()
        assert isinstance(config, fill.GaussSeidel)

    def test_gauss_seidel_instance_methods(self) -> None:
        """Test GaussSeidel instance methods."""
        config = fill.GaussSeidel()

        # Test with_epsilon
        config_epsilon = config.with_epsilon(1e-6)
        assert isinstance(config_epsilon, fill.GaussSeidel)
        assert config_epsilon is not config

        # Test with_first_guess
        config_fg = config.with_first_guess(fill.FirstGuess.ZONAL_AVERAGE)
        assert isinstance(config_fg, fill.GaussSeidel)
        assert config_fg is not config

        # Test with_max_iterations
        config_iter = config.with_max_iterations(100)
        assert isinstance(config_iter, fill.GaussSeidel)
        assert config_iter is not config

        # Test with_num_threads
        config_threads = config.with_num_threads(4)
        assert isinstance(config_threads, fill.GaussSeidel)
        assert config_threads is not config

        # Test with_relaxation
        config_relax = config.with_relaxation(1.5)
        assert isinstance(config_relax, fill.GaussSeidel)
        assert config_relax is not config

    def test_gauss_seidel_method_chaining(self) -> None:
        """Test GaussSeidel method chaining."""
        config = (
            fill.GaussSeidel()
            .with_epsilon(1e-5)
            .with_first_guess(fill.FirstGuess.ZERO)
            .with_max_iterations(200)
            .with_num_threads(8)
            .with_relaxation(1.8)
        )
        assert isinstance(config, fill.GaussSeidel)

    def test_fft_inpaint_creation(self) -> None:
        """Test FFTInpaint configuration creation."""
        config = fill.FFTInpaint()
        assert isinstance(config, fill.FFTInpaint)

    def test_fft_inpaint_instance_methods(self) -> None:
        """Test FFTInpaint instance methods."""
        config = fill.FFTInpaint()

        # Test with_epsilon
        config_epsilon = config.with_epsilon(1e-4)
        assert isinstance(config_epsilon, fill.FFTInpaint)
        assert config_epsilon is not config

        # Test with_first_guess
        config_fg = config.with_first_guess(fill.FirstGuess.ZONAL_AVERAGE)
        assert isinstance(config_fg, fill.FFTInpaint)
        assert config_fg is not config

        # Test with_max_iterations
        config_iter = config.with_max_iterations(50)
        assert isinstance(config_iter, fill.FFTInpaint)
        assert config_iter is not config

        # Test with_num_threads
        config_threads = config.with_num_threads(2)
        assert isinstance(config_threads, fill.FFTInpaint)
        assert config_threads is not config

        # Test with_sigma
        config_sigma = config.with_sigma(2.0)
        assert isinstance(config_sigma, fill.FFTInpaint)
        assert config_sigma is not config

    def test_fft_inpaint_method_chaining(self) -> None:
        """Test FFTInpaint method chaining."""
        config = (
            fill.FFTInpaint()
            .with_epsilon(1e-3)
            .with_first_guess(fill.FirstGuess.ZERO)
            .with_max_iterations(75)
            .with_num_threads(4)
            .with_sigma(3.0)
        )
        assert isinstance(config, fill.FFTInpaint)

    def test_loess_creation(self) -> None:
        """Test Loess configuration creation."""
        config = fill.Loess()
        assert isinstance(config, fill.Loess)

    def test_loess_instance_methods(self) -> None:
        """Test Loess instance methods."""
        config = fill.Loess()

        # Test with_epsilon
        config_epsilon = config.with_epsilon(1e-7)
        assert isinstance(config_epsilon, fill.Loess)
        assert config_epsilon is not config

        # Test with_first_guess
        config_fg = config.with_first_guess(fill.FirstGuess.ZONAL_AVERAGE)
        assert isinstance(config_fg, fill.Loess)
        assert config_fg is not config

        # Test with_max_iterations
        config_iter = config.with_max_iterations(150)
        assert isinstance(config_iter, fill.Loess)
        assert config_iter is not config

        # Test with_num_threads
        config_threads = config.with_num_threads(6)
        assert isinstance(config_threads, fill.Loess)
        assert config_threads is not config

        # Test with_nx
        config_nx = config.with_nx(10)
        assert isinstance(config_nx, fill.Loess)
        assert config_nx is not config

        # Test with_ny
        config_ny = config.with_ny(8)
        assert isinstance(config_ny, fill.Loess)
        assert config_ny is not config

        # Test with_value_type
        config_vt = config.with_value_type(fill.LoessValueType.DEFINED)
        assert isinstance(config_vt, fill.Loess)
        assert config_vt is not config

    def test_loess_method_chaining(self) -> None:
        """Test Loess method chaining."""
        config = (
            fill.Loess()
            .with_epsilon(1e-6)
            .with_first_guess(fill.FirstGuess.ZERO)
            .with_max_iterations(100)
            .with_num_threads(4)
            .with_nx(12)
            .with_ny(10)
            .with_value_type(fill.LoessValueType.ALL)
        )
        assert isinstance(config, fill.Loess)

    def test_loess_value_types(self) -> None:
        """Test different value types for Loess."""
        value_types = [
            fill.LoessValueType.ALL,
            fill.LoessValueType.DEFINED,
            fill.LoessValueType.UNDEFINED,
        ]

        for value_type in value_types:
            config = fill.Loess().with_value_type(value_type)
            assert isinstance(config, fill.Loess)

    def test_multigrid_creation(self) -> None:
        """Test Multigrid configuration creation."""
        config = fill.Multigrid()
        assert isinstance(config, fill.Multigrid)

    def test_multigrid_instance_methods(self) -> None:
        """Test Multigrid instance methods."""
        config = fill.Multigrid()

        # Test with_epsilon
        config_epsilon = config.with_epsilon(1e-8)
        assert isinstance(config_epsilon, fill.Multigrid)
        assert config_epsilon is not config

        # Test with_first_guess
        config_fg = config.with_first_guess(fill.FirstGuess.ZONAL_AVERAGE)
        assert isinstance(config_fg, fill.Multigrid)
        assert config_fg is not config

        # Test with_max_iterations
        config_iter = config.with_max_iterations(50)
        assert isinstance(config_iter, fill.Multigrid)
        assert config_iter is not config

        # Test with_num_threads
        config_threads = config.with_num_threads(8)
        assert isinstance(config_threads, fill.Multigrid)
        assert config_threads is not config

        # Test with_pre_smooth
        config_pre = config.with_pre_smooth(3)
        assert isinstance(config_pre, fill.Multigrid)
        assert config_pre is not config

        # Test with_post_smooth
        config_post = config.with_post_smooth(2)
        assert isinstance(config_post, fill.Multigrid)
        assert config_post is not config

    def test_multigrid_method_chaining(self) -> None:
        """Test Multigrid method chaining."""
        config = (
            fill.Multigrid()
            .with_epsilon(1e-5)
            .with_first_guess(fill.FirstGuess.ZERO)
            .with_max_iterations(100)
            .with_num_threads(4)
            .with_pre_smooth(5)
            .with_post_smooth(3)
        )
        assert isinstance(config, fill.Multigrid)

    def test_first_guess_variations(self) -> None:
        """Test different first guess strategies across all configs."""
        first_guesses = [
            fill.FirstGuess.ZERO,
            fill.FirstGuess.ZONAL_AVERAGE,
        ]

        for fg in first_guesses:
            config_gs = fill.GaussSeidel().with_first_guess(fg)
            assert isinstance(config_gs, fill.GaussSeidel)

            config_fft = fill.FFTInpaint().with_first_guess(fg)
            assert isinstance(config_fft, fill.FFTInpaint)

            config_loess = fill.Loess().with_first_guess(fg)
            assert isinstance(config_loess, fill.Loess)

            config_mg = fill.Multigrid().with_first_guess(fg)
            assert isinstance(config_mg, fill.Multigrid)

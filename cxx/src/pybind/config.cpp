// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/config.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <optional>

#include "pyinterp/config/common.hpp"
#include "pyinterp/config/fill.hpp"
#include "pyinterp/config/geometric.hpp"
#include "pyinterp/config/rtree.hpp"
#include "pyinterp/config/windowed.hpp"
#include "pyinterp/math/interpolate/rbf.hpp"

namespace nb = nanobind;

namespace pyinterp {
namespace config {

template <typename Class>
auto add_common_attributes(nb::class_<Class>& pyclass) -> nb::class_<Class>& {
  pyclass
      .def("with_bounds_error", &Class::with_bounds_error, nb::arg("value"),
           "Whether to raise an error when interpolated values are "
           "requested outside the domain defined by the input "
           "data.")
      .def("with_num_threads", &Class::with_num_threads, nb::arg("value"),
           "Number of threads to use for interpolation. A value of 0 means "
           "that all available cores will be used.");
  return pyclass;
}

namespace geometric::pybind {

/// @brief Add bivariate/trivariate/quadrivariate-specific methods
template <typename Class>
inline auto add_methods(nb::class_<Class>& pyclass) -> nb::class_<Class>& {
  pyclass
      .def_static("bilinear", &Class::bilinear,
                  "Create a configuration for bilinear interpolation.",
                  nb::call_guard<nb::gil_scoped_release>())
      .def_static("idw", &Class::idw, nb::arg("exp") = 2,
                  "Create a configuration for inverse distance weighting "
                  "interpolation.",
                  nb::call_guard<nb::gil_scoped_release>())
      .def_static("nearest", &Class::nearest,
                  "Create a configuration for nearest-neighbor interpolation.",
                  nb::call_guard<nb::gil_scoped_release>());
  return pyclass;
}

inline auto bind(nb::module_& m) -> void {
  add_common_attributes(add_methods(
      nb::class_<Bivariate>(m, "Bivariate",
                            "Parameters controlling bivariate interpolation "
                            "on two-dimensional grids.")
          .def(nb::init<>(), "Default constructor.",
               nb::call_guard<nb::gil_scoped_release>())));

  add_common_attributes(add_methods(
      nb::class_<Trivariate>(m, "Trivariate",
                             "Parameters controlling trivariate interpolation "
                             "on three-dimensional grids.")
          .def(nb::init<>(), "Default constructor.",
               nb::call_guard<nb::gil_scoped_release>())));

  add_common_attributes(
      add_methods(nb::class_<Quadrivariate>(
                      m, "Quadrivariate",
                      "Parameters controlling quadrivariate interpolation "
                      "on four-dimensional grids.")
                      .def(nb::init<>(), "Default constructor.",
                           nb::call_guard<nb::gil_scoped_release>())));
}

}  // namespace geometric::pybind

namespace windowed::pybind {

template <typename Class>
auto add_windowed_methods(nb::class_<Class>& pyclass) -> nb::class_<Class>& {
  pyclass
      .def_static("akima", &Class::akima,
                  "Create a configuration for Akima spline interpolation.",
                  nb::call_guard<nb::gil_scoped_release>())
      .def_static("akima_periodic", &Class::akima_periodic,
                  "Create a configuration for Akima periodic spline "
                  "interpolation.",
                  nb::call_guard<nb::gil_scoped_release>())
      .def_static("c_spline", &Class::c_spline,
                  "Create a configuration for C spline interpolation.",
                  nb::call_guard<nb::gil_scoped_release>())
      .def_static("c_spline_not_a_knot", &Class::c_spline_not_a_knot,
                  "Create a configuration for C spline not-a-knot "
                  "interpolation.",
                  nb::call_guard<nb::gil_scoped_release>())
      .def_static("c_spline_periodic", &Class::c_spline_periodic,
                  "Create a configuration for C spline periodic "
                  "interpolation.",
                  nb::call_guard<nb::gil_scoped_release>())
      .def_static("steffen", &Class::steffen,
                  "Create a configuration for Steffen spline interpolation.",
                  nb::call_guard<nb::gil_scoped_release>())
      .def_static("linear", &Class::linear,
                  "Create a configuration for linear interpolation.",
                  nb::call_guard<nb::gil_scoped_release>())
      .def_static("polynomial", &Class::polynomial,
                  "Create a configuration for polynomial spline "
                  "interpolation.",
                  nb::call_guard<nb::gil_scoped_release>())
      .def("with_boundary_mode", &Class::with_boundary_mode,
           "Update boundary mode.", nb::arg("config"),
           nb::call_guard<nb::gil_scoped_release>());
  return pyclass;
}

template <typename Class>
auto add_methods(nb::class_<Class>& pyclass) -> nb::class_<Class>& {
  add_windowed_methods(pyclass);
  pyclass
      .def_static("bicubic", &Class::bicubic,
                  "Create a configuration for bicubic interpolation.",
                  nb::call_guard<nb::gil_scoped_release>())
      .def_static("bilinear", &Class::bilinear,
                  "Create a configuration for bilinear interpolation.",
                  nb::call_guard<nb::gil_scoped_release>())
      .def("with_half_window_size_x", &Class::with_half_window_size_x,
           "Update half window size in x direction.", nb::arg("size"),
           nb::call_guard<nb::gil_scoped_release>())
      .def("with_half_window_size_y", &Class::with_half_window_size_y,
           "Update half window size in y direction.", nb::arg("size"),
           nb::call_guard<nb::gil_scoped_release>());
  return pyclass;
}

auto bind(nb::module_& m) -> void {
  nb::class_<AxisConfig>(m, "AxisConfig",
                         "Configuration for a single-axis interpolation.")
      .def(nb::init<>(), "Default constructor.",
           nb::call_guard<nb::gil_scoped_release>())
      .def_static("linear", &AxisConfig::linear,
                  "Create a configuration for linear interpolation.",
                  nb::call_guard<nb::gil_scoped_release>())
      .def_static("nearest", &AxisConfig::nearest,
                  "Create a configuration for nearest-neighbor interpolation.",
                  nb::call_guard<nb::gil_scoped_release>());

  nb::class_<BoundaryConfig>(
      m, "BoundaryConfig",
      "Configuration for boundary handling in windowed interpolation.")
      .def_static("shrink", &BoundaryConfig::shrink,
                  "Create a configuration to shrink the window at the "
                  "boundaries.",
                  nb::call_guard<nb::gil_scoped_release>())
      .def_static("undef", &BoundaryConfig::undef,
                  "Create a configuration with undefined boundary mode.",
                  nb::call_guard<nb::gil_scoped_release>());

  // Bind windowed Univariate configuration
  add_common_attributes(
      add_windowed_methods(
          nb::class_<Univariate>(m, "Univariate",
                                 "Parameters controlling univariate windowed "
                                 "interpolation on one-dimensional signals.")
              .def(nb::init<>(), "Default constructor.",
                   nb::call_guard<nb::gil_scoped_release>()))
          .def("with_half_window_size", &Univariate::with_half_window_size,
               "Update half window size.", nb::arg("size"),
               nb::call_guard<nb::gil_scoped_release>()));

  add_common_attributes(add_methods(
      nb::class_<Bivariate>(m, "Bivariate",
                            "Parameters controlling the windowing "
                            "interpolation on two-dimensional grids.")
          .def(nb::init<>(), "Default constructor.",
               nb::call_guard<nb::gil_scoped_release>())));

  auto trivariate = add_common_attributes(add_methods(
      nb::class_<Trivariate>(m, "Trivariate",
                             "Parameters controlling the windowing "
                             "interpolation on three-dimensional grids.")
          .def(nb::init<>(), "Default constructor.",
               nb::call_guard<nb::gil_scoped_release>())));
  trivariate.def("with_third_axis", &Trivariate::with_third_axis,
                 nb::arg("config"), "Update third axis configuration.",
                 nb::call_guard<nb::gil_scoped_release>());

  auto quadrivariate = add_common_attributes(add_methods(
      nb::class_<Quadrivariate>(m, "Quadrivariate",
                                "Parameters controlling the windowing "
                                "interpolation on four-dimensional grids.")
          .def(nb::init<>(), "Default constructor.",
               nb::call_guard<nb::gil_scoped_release>())));
  quadrivariate
      .def("with_third_axis", &Quadrivariate::with_third_axis,
           nb::arg("config"), "Update third axis configuration.",
           nb::call_guard<nb::gil_scoped_release>())
      .def("with_fourth_axis", &Quadrivariate::with_fourth_axis,
           nb::arg("config"), "Update fourth axis configuration.",
           nb::call_guard<nb::gil_scoped_release>());
}

}  // namespace windowed::pybind

namespace rtree::pybind {

/// @brief Add common RTree methods (k, radius, num_threads) to a class
/// @tparam Class The configuration class type
/// @param pyclass The nanobind class wrapper
/// @return Reference to the modified class wrapper
template <typename Class>
auto add_rtree_methods(nb::class_<Class>& pyclass) -> nb::class_<Class>& {
  pyclass
      .def("with_k", &Class::with_k, nb::arg("value"),
           "Set the number of neighbors to consider for interpolation.",
           nb::call_guard<nb::gil_scoped_release>())
      .def("with_radius", &Class::with_radius, nb::arg("value") = std::nullopt,
           "Set the search radius in meters (None for unlimited).",
           nb::call_guard<nb::gil_scoped_release>())
      .def("with_boundary_check", &Class::with_boundary_check, nb::arg("value"),
           "Set the type of boundary check to apply.",
           nb::call_guard<nb::gil_scoped_release>())
      .def("with_num_threads", &Class::with_num_threads, nb::arg("value"),
           "Number of threads to use for interpolation. A value of 0 means "
           "that all available cores will be used.",
           nb::call_guard<nb::gil_scoped_release>());
  return pyclass;
}

auto bind(nb::module_& m) -> void {
  // Bind BoundaryCheck enum
  nb::enum_<geometry::BoundaryCheck>(m, "BoundaryCheck",
                                     "Type of boundary check to apply.")
      .value("NONE", geometry::BoundaryCheck::kNone,
             "Do not apply boundary check (default).")
      .value("ENVELOPE", geometry::BoundaryCheck::kEnvelope,
             "Check if the point is within the Axis Aligned Bounding Box "
             "(AABB) of the neighbors.")
      .value("CONVEX_HULL", geometry::BoundaryCheck::kConvexHull,
             "Check if the point is within the convex hull of the neighbors.");
  // Bind RadialBasisFunction enum
  nb::enum_<math::interpolate::RBFKernel>(m, "RBFKernel",
                                          "Type of radial basis kernel.")
      .value("CUBIC", math::interpolate::RBFKernel::kCubic,
             "Cubic radial basis function.")
      .value("GAUSSIAN", math::interpolate::RBFKernel::kGaussian,
             "Gaussian radial basis function.")
      .value("INVERSE_MULTIQUADRIC",
             math::interpolate::RBFKernel::kInverseMultiquadric,
             "Inverse multiquadric radial basis function.")
      .value("LINEAR", math::interpolate::RBFKernel::kLinear,
             "Linear radial basis function.")
      .value("MULTIQUADRIC", math::interpolate::RBFKernel::kMultiquadric,
             "Multiquadric radial basis function.")
      .value("THIN_PLATE", math::interpolate::RBFKernel::kThinPlate,
             "Thin plate radial basis function.");

  // Bind CovarianceFunction enum
  nb::enum_<math::interpolate::CovarianceFunction>(
      m, "CovarianceFunction",
      "Type of covariance function for Kriging interpolation.")
      .value("MATERN_12", math::interpolate::CovarianceFunction::kMatern_12,
             "Matérn :math:`\\nu = 0.5` (exponential, C⁰).")
      .value("MATERN_32", math::interpolate::CovarianceFunction::kMatern_32,
             "Matérn :math:`\\nu = 1.5` (C¹).")
      .value("MATERN_52", math::interpolate::CovarianceFunction::kMatern_52,
             "Matérn :math:`\\nu = 2.5` (C²).")
      .value("CAUCHY", math::interpolate::CovarianceFunction::kCauchy,
             "Cauchy (heavy-tailed).")
      .value("SPHERICAL", math::interpolate::CovarianceFunction::kSpherical,
             "Spherical (compact support).")
      .value("GAUSSIAN", math::interpolate::CovarianceFunction::kGaussian,
             "Gaussian (:math:`C^\\infty`, can cause numerical issues).")
      .value(
          "WENDLAND", math::interpolate::CovarianceFunction::kWendland,
          "Wendland :math:`\\phi_{3,0}` (compact support, sparse matrices).");

  // Bind window::Function enum
  nb::enum_<math::interpolate::window::Kernel>(m, "WindowKernel",
                                               "Window kernel function.")
      .value("BLACKMAN", math::interpolate::window::Kernel::kBlackman,
             "Blackman window function.")
      .value("BLACKMAN_HARRIS",
             math::interpolate::window::Kernel::kBlackmanHarris,
             "Blackman-Harris window function.")
      .value("BOXCAR", math::interpolate::window::Kernel::kBoxcar,
             "Boxcar (rectangular) window function.")
      .value("FLAT_TOP", math::interpolate::window::Kernel::kFlatTop,
             "Flat top window function (used for accurate amplitude "
             "measurements).")
      .value("GAUSSIAN", math::interpolate::window::Kernel::kGaussian,
             "Gaussian window function.")
      .value("HAMMING", math::interpolate::window::Kernel::kHamming,
             "Hamming window function.")
      .value("LANCZOS", math::interpolate::window::Kernel::kLanczos,
             "Lanczos window function.")
      .value("NUTTALL", math::interpolate::window::Kernel::kNuttall,
             "Nuttall window function.")
      .value("PARZEN", math::interpolate::window::Kernel::kParzen,
             "Parzen window function.")
      .value("PARZEN_SWOT", math::interpolate::window::Kernel::kParzenSWOT,
             "Parzen SWOT window function.");

  // Bind DriftFunction enum
  nb::enum_<math::interpolate::DriftFunction>(
      m, "DriftFunction",
      "Type of drift function for Universal Kriging interpolation.")
      .value("LINEAR", math::interpolate::DriftFunction::kLinear,
             "Constant + linear terms (4 parameters).")
      .value("QUADRATIC", math::interpolate::DriftFunction::kQuadratic,
             "Constant + linear + quadratic terms (10 parameters).");

  // Bind Query configuration
  add_rtree_methods(
      nb::class_<Query>(m, "Query", "Configuration for query operations.")
          .def(nb::init<>(), "Default constructor.",
               nb::call_guard<nb::gil_scoped_release>()));

  // Bind InverseDistanceWeighting configuration
  add_rtree_methods(
      nb::class_<InverseDistanceWeighting>(
          m, "InverseDistanceWeighting",
          "Configuration for inverse distance weighting interpolation.")
          .def(nb::init<>(), "Default constructor.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_p", &InverseDistanceWeighting::with_p, nb::arg("value"),
               "Set the power parameter (exponent) for distance weighting.",
               nb::call_guard<nb::gil_scoped_release>()));

  // Bind Kriging configuration
  add_rtree_methods(
      nb::class_<Kriging>(m, "Kriging",
                          "Configuration for kriging interpolation.")
          .def(nb::init<>(), "Default constructor.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_sigma", &Kriging::with_sigma, nb::arg("value"),
               "Set the sill parameter (variance at infinity).",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_lambda", &Kriging::with_lambda, nb::arg("value"),
               "Set the range parameter (distance scale).",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_nugget", &Kriging::with_nugget, nb::arg("value"),
               "Set the nugget effect parameter (micro-scale variance).",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_covariance_model", &Kriging::with_covariance_model,
               nb::arg("value"), "Set the covariance function type.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_drift_function", &Kriging::with_drift_function,
               nb::arg("value"), "Set the drift function.",
               nb::call_guard<nb::gil_scoped_release>()));

  // Bind RadialBasisFunction configuration
  add_rtree_methods(
      nb::class_<RadialBasisFunction>(
          m, "RadialBasisFunction",
          "Configuration for radial basis function interpolation.")
          .def(nb::init<>(), "Default constructor.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_rbf", &RadialBasisFunction::with_rbf, nb::arg("value"),
               "Set the radial basis function type.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_epsilon", &RadialBasisFunction::with_epsilon,
               nb::arg("value") = std::nullopt,
               "Set the shape parameter epsilon (None for automatic).",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_smooth", &RadialBasisFunction::with_smooth,
               nb::arg("value"), "Set the smoothing parameter.",
               nb::call_guard<nb::gil_scoped_release>()));

  // Bind InterpolationWindow configuration
  add_rtree_methods(nb::class_<InterpolationWindow>(
                        m, "InterpolationWindow",
                        "Configuration for window function interpolation.")
                        .def(nb::init<>(), "Default constructor.",
                             nb::call_guard<nb::gil_scoped_release>())
                        .def("with_wf", &InterpolationWindow::with_wf,
                             nb::arg("value"), "Set the window function type.",
                             nb::call_guard<nb::gil_scoped_release>())
                        .def("with_arg", &InterpolationWindow::with_arg,
                             nb::arg("value") = std::nullopt,
                             "Set the window function argument.",
                             nb::call_guard<nb::gil_scoped_release>()));
}

}  // namespace rtree::pybind

namespace fill::pybind {

/// @brief Add common fill methods to a configuration class
/// @tparam Class The configuration class type
/// @param pyclass The nanobind class wrapper
/// @return Reference to the modified class wrapper
template <typename Class>
auto add_fill_methods(nb::class_<Class>& pyclass) -> nb::class_<Class>& {
  pyclass
      .def("with_first_guess", &Class::with_first_guess, nb::arg("value"),
           "Set the first guess strategy.",
           nb::call_guard<nb::gil_scoped_release>())
      .def("with_max_iterations", &Class::with_max_iterations, nb::arg("value"),
           "Set the maximum number of iterations.",
           nb::call_guard<nb::gil_scoped_release>())
      .def("with_epsilon", &Class::with_epsilon, nb::arg("value"),
           "Set the convergence threshold (epsilon).",
           nb::call_guard<nb::gil_scoped_release>())
      .def("with_num_threads", &Class::with_num_threads, nb::arg("value"),
           "Number of threads to use. A value of 0 means that all available "
           "cores will be used.",
           nb::call_guard<nb::gil_scoped_release>())
      .def("with_is_periodic", &Class::with_is_periodic, nb::arg("value"),
           "Set whether the X-axis is periodic.",
           nb::call_guard<nb::gil_scoped_release>());
  return pyclass;
}

auto bind(nb::module_& m) -> void {
  // Bind FirstGuess enum
  nb::enum_<FirstGuess>(m, "FirstGuess",
                        "Initial guess strategy for iterative fill methods.")
      .value("ZONAL_AVERAGE", FirstGuess::kZonalAverage,
             "Use zonal average of defined values.")
      .value("ZERO", FirstGuess::kZero, "Use zero as initial guess.");

  // Bind LoessValueType enum
  nb::enum_<LoessValueType>(m, "LoessValueType",
                            "Type of values processed by the LOESS filter.")
      .value("UNDEFINED", LoessValueType::kUndefined,
             "Fill only undefined (NaN) values.")
      .value("DEFINED", LoessValueType::kDefined, "Smooth only defined values.")
      .value("ALL", LoessValueType::kAll, "Smooth and fill all values.");

  // Bind Loess configuration
  add_fill_methods(
      nb::class_<Loess>(m, "Loess", "Configuration for LOESS fill method.")
          .def(nb::init<>(), "Default constructor.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_value_type", &Loess::with_value_type, nb::arg("value"),
               "Set the value type to process.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_nx", &Loess::with_nx, nb::arg("value"),
               "Set the half-window size along x-axis.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_ny", &Loess::with_ny, nb::arg("value"),
               "Set the half-window size along y-axis.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("value_type", &Loess::value_type,
               "Get the value type to process.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("nx", &Loess::nx, "Get the half-window size along x-axis.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("ny", &Loess::ny, "Get the half-window size along y-axis.",
               nb::call_guard<nb::gil_scoped_release>()));

  // Bind FFTInpaint configuration
  add_fill_methods(
      nb::class_<FFTInpaint>(m, "FFTInpaint",
                             "Configuration for FFT Inpaint fill method.")
          .def(nb::init<>(), "Default constructor.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_sigma", &FFTInpaint::with_sigma, nb::arg("value"),
               "Set the sigma parameter.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("sigma", &FFTInpaint::sigma, "Get the sigma parameter.",
               nb::call_guard<nb::gil_scoped_release>()));

  // Bind GaussSeidel configuration
  add_fill_methods(
      nb::class_<GaussSeidel>(m, "GaussSeidel",
                              "Configuration for Gauss-Seidel fill method.")
          .def(nb::init<>(), "Default constructor.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_relaxation", &GaussSeidel::with_relaxation,
               nb::arg("value"), "Set the relaxation parameter.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("relaxation", &GaussSeidel::relaxation,
               "Get the relaxation parameter.",
               nb::call_guard<nb::gil_scoped_release>()));

  // Bind Multigrid configuration
  add_fill_methods(
      nb::class_<Multigrid>(m, "Multigrid",
                            "Configuration for Multigrid fill method.")
          .def(nb::init<>(), "Default constructor.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_pre_smooth", &Multigrid::with_pre_smooth, nb::arg("value"),
               "Set the number of pre-smoothing iterations.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("with_post_smooth", &Multigrid::with_post_smooth,
               nb::arg("value"), "Set the number of post-smoothing iterations.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("pre_smooth", &Multigrid::pre_smooth,
               "Get the number of pre-smoothing iterations.",
               nb::call_guard<nb::gil_scoped_release>())
          .def("post_smooth", &Multigrid::post_smooth,
               "Get the number of post-smoothing iterations.",
               nb::call_guard<nb::gil_scoped_release>()));
}

}  // namespace fill::pybind
}  // namespace config

namespace pybind {

auto init_config(nb::module_& m) -> void {
  auto config = m.def_submodule("config", "Interpolation configurations.");
  auto geometric = config.def_submodule(
      "geometric", "Configuration for geometric interpolation.");
  auto windowed = config.def_submodule(
      "windowed", "Configuration for windowed interpolation.");
  auto rtree =
      config.def_submodule("rtree", "Configuration for RTree interpolation.");
  auto fill = config.def_submodule("fill", "Configuration for fill methods.");
  config::geometric::pybind::bind(geometric);
  config::windowed::pybind::bind(windowed);
  config::rtree::pybind::bind(rtree);
  config::fill::pybind::bind(fill);
}

}  // namespace pybind
}  // namespace pyinterp

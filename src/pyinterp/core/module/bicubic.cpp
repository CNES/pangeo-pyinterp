// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/detail/math/bicubic.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cctype>

#include "pyinterp/detail/math/linear.hpp"
#include "pyinterp/detail/math/spline2d.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/frame.hpp"

namespace py = pybind11;

namespace pyinterp {

/// Parse the requested boundary option
static inline auto parse_axis_boundary(const std::string &boundary)
    -> axis::Boundary {
  if (boundary == "expand") {
    return axis::kExpand;
  }
  if (boundary == "wrap") {
    return axis::kWrap;
  }
  if (boundary == "sym") {
    return axis::kSym;
  }
  if (boundary == "undef") {
    return axis::kUndef;
  }
  throw std::invalid_argument("boundary '" + boundary + "' is not defined");
}

/// Evaluate the interpolation.
template <typename DataType, typename Interpolator>
auto bicubic(const Grid2D<DataType> &grid, const py::array_t<double> &x,
             const py::array_t<double> &y, Eigen::Index nx, Eigen::Index ny,
             const std::string &fitting_model, const std::string &boundary,
             const bool bounds_error, size_t num_threads)
    -> py::array_t<double> {
  detail::check_array_ndim("x", 1, x, "y", 1, y);
  detail::check_ndarray_shape("x", x, "y", y);

  auto boundary_type = parse_axis_boundary(boundary);
  auto size = x.size();
  auto result = py::array_t<double>(py::array::ShapeContainer{size});

  auto _x = x.template unchecked<1>();
  auto _y = y.template unchecked<1>();
  auto _result = result.template mutable_unchecked<1>();
  {
    py::gil_scoped_release release;

    // Captures the detected exceptions in the calculation function
    // (only the last exception captured is kept)
    auto except = std::exception_ptr(nullptr);

    // Access to the shared pointer outside the loop to avoid data races
    const auto is_angle = grid.x()->is_angle();

    detail::dispatch(
        [&](const size_t start, const size_t end) {
          try {
            auto frame = detail::math::Frame2D(nx, ny);
            auto interpolator = Interpolator(frame, fitting_model);

            for (size_t ix = start; ix < end; ++ix) {
              auto xi = _x(ix);
              auto yi = _y(ix);
              _result(ix) =
                  // The grid instance is accessed as a constant reference, no
                  // data race problem here.
                  load_frame(grid, xi, yi, boundary_type, bounds_error, frame)
                      ? interpolator.interpolate(
                            is_angle ? frame.normalize_angle(xi) : xi, yi,
                            frame)
                      : std::numeric_limits<double>::quiet_NaN();
            }
          } catch (...) {
            except = std::current_exception();
          }
        },
        size, num_threads);

    if (except != nullptr) {
      std::rethrow_exception(except);
    }
  }
  return result;
}

/// Evaluate the interpolation.
template <typename DataType, typename AxisType, typename Interpolator>
auto bicubic_3d(const Grid3D<DataType, AxisType> &grid,
                const py::array_t<double> &x, const py::array_t<double> &y,
                const py::array_t<AxisType> &z, Eigen::Index nx,
                Eigen::Index ny, const std::string &fitting_model,
                const std::string &boundary, const bool bounds_error,
                size_t num_threads) -> py::array_t<double> {
  detail::check_array_ndim("x", 1, x, "y", 1, y, "z", 1, z);
  detail::check_ndarray_shape("x", x, "y", y, "z", z);
  auto boundary_type = parse_axis_boundary(boundary);

  auto size = x.size();
  auto result = py::array_t<double>(py::array::ShapeContainer{size});

  auto _x = x.template unchecked<1>();
  auto _y = y.template unchecked<1>();
  auto _z = z.template unchecked<1>();
  auto _result = result.template mutable_unchecked<1>();
  {
    py::gil_scoped_release release;

    // Captures the detected exceptions in the calculation function
    // (only the last exception captured is kept)
    auto except = std::exception_ptr(nullptr);

    // Access to the shared pointer outside the loop to avoid data races
    const auto is_angle = grid.x()->is_angle();

    detail::dispatch(
        [&](const size_t start, const size_t end) {
          try {
            auto frame = detail::math::Frame3D<AxisType>(nx, ny, 1);
            auto interpolator =
                Interpolator(detail::math::Frame2D(nx, ny), fitting_model);

            for (size_t ix = start; ix < end; ++ix) {
              auto xi = _x(ix);
              auto yi = _y(ix);
              auto zi = _z(ix);

              if (load_frame<DataType, AxisType>(
                      grid, xi, yi, zi, boundary_type, bounds_error, frame)) {
                xi = is_angle ? frame.normalize_angle(xi) : xi;
                auto z0 = interpolator.interpolate(xi, yi, frame.frame_2d(0));
                auto z1 = interpolator.interpolate(xi, yi, frame.frame_2d(1));
                _result(ix) = detail::math::linear<AxisType, double>(
                    zi, frame.z(0), frame.z(1), z0, z1);
              } else {
                _result(ix) = std::numeric_limits<double>::quiet_NaN();
              }
            }
          } catch (...) {
            except = std::current_exception();
          }
        },
        size, num_threads);

    if (except != nullptr) {
      std::rethrow_exception(except);
    }
  }
  return result;
}

/// Evaluate the interpolation.
template <typename DataType, typename AxisType, typename Interpolator>
auto bicubic_4d(const Grid4D<DataType, AxisType> &grid,
                const py::array_t<double> &x, const py::array_t<double> &y,
                const py::array_t<AxisType> &z, const py::array_t<double> &u,
                Eigen::Index nx, Eigen::Index ny,
                const std::string &fitting_model, const std::string &boundary,
                const bool bounds_error, size_t num_threads)
    -> py::array_t<double> {
  detail::check_array_ndim("x", 1, x, "y", 1, y, "z", 1, z, "u", 1, u);
  detail::check_ndarray_shape("x", x, "y", y, "z", z, "u", u);
  auto boundary_type = parse_axis_boundary(boundary);

  auto size = x.size();
  auto result = py::array_t<double>(py::array::ShapeContainer{size});

  auto _x = x.template unchecked<1>();
  auto _y = y.template unchecked<1>();
  auto _z = z.template unchecked<1>();
  auto _u = u.template unchecked<1>();
  auto _result = result.template mutable_unchecked<1>();
  {
    py::gil_scoped_release release;

    // Captures the detected exceptions in the calculation function
    // (only the last exception captured is kept)
    auto except = std::exception_ptr(nullptr);

    // Access to the shared pointer outside the loop to avoid data races
    const auto is_angle = grid.x()->is_angle();

    detail::dispatch(
        [&](const size_t start, const size_t end) {
          try {
            auto frame = detail::math::Frame4D<AxisType>(nx, ny, 1, 1);
            auto interpolator =
                Interpolator(detail::math::Frame2D(nx, ny), fitting_model);

            for (size_t ix = start; ix < end; ++ix) {
              auto xi = _x(ix);
              auto yi = _y(ix);
              auto zi = _z(ix);
              auto ui = _u(ix);

              if (load_frame<DataType, AxisType>(grid, xi, yi, zi, ui,
                                                 boundary_type, bounds_error,
                                                 frame)) {
                xi = is_angle ? frame.normalize_angle(xi) : xi;
                auto z00 =
                    interpolator.interpolate(xi, yi, frame.frame_2d(0, 0));
                auto z10 =
                    interpolator.interpolate(xi, yi, frame.frame_2d(1, 0));
                auto z01 =
                    interpolator.interpolate(xi, yi, frame.frame_2d(0, 1));
                auto z11 =
                    interpolator.interpolate(xi, yi, frame.frame_2d(1, 1));
                _result(ix) = detail::math::linear<double>(
                    ui, frame.u(0), frame.u(1),
                    detail::math::linear<AxisType, double>(
                        zi, frame.z(0), frame.z(1), z00, z10),
                    detail::math::linear<AxisType, double>(
                        zi, frame.z(0), frame.z(1), z01, z11));
              } else {
                _result(ix) = std::numeric_limits<double>::quiet_NaN();
              }
            }
          } catch (...) {
            except = std::current_exception();
          }
        },
        size, num_threads);

    if (except != nullptr) {
      std::rethrow_exception(except);
    }
  }
  return result;
}

}  // namespace pyinterp

template <typename DataType, typename Interpolator>
void implement_bicubic(py::module &m, const std::string &prefix,
                       const std::string &suffix,
                       const std::string &default_fitting_model) {
  auto function_prefix = prefix;
  auto function_suffix = suffix;
  function_prefix[0] = static_cast<char>(std::tolower(function_prefix[0]));
  function_suffix[0] = static_cast<char>(std::tolower(function_suffix[0]));

  m.def((function_prefix + "_" + function_suffix).c_str(),
        &pyinterp::bicubic<DataType, Interpolator>, py::arg("grid"),
        py::arg("x"), py::arg("y"), py::arg("nx") = 3, py::arg("ny") = 3,
        py::arg("fitting_model") = default_fitting_model,
        py::arg("boundary") = "undef", py::arg("bounds_error") = false,
        py::arg("num_threads") = 0,
        (prefix + R"__doc__( gridded 2D interpolation.

Args:
    grid: Grid containing the values to be interpolated.
    x: X-values.
    y: Y-values.
    nx: The number of X coordinate values required to perform the interpolation.
        Defaults to ``3``.
    ny: The number of Y coordinate values required to perform the interpolation.
        Defaults to ``3``.
    fitting_model: Type of interpolation to be performed. Defaults to `)__doc__" +
         default_fitting_model + R"__doc__(`
    boundary: Type of axis boundary management. Defaults to ``undef``.
    bounds_error: If True, when interpolated values are requested outside of the
        domain of the input axes (x,y), a ValueError is raised. If False, then
        value is set to NaN.
    num_threads: The number of threads to use for the computation. If 0 all
        CPUs are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    Values interpolated
  )__doc__")
            .c_str());
}

template <typename DataType, typename AxisType, typename Interpolator>
void implement_bicubic_3d(py::module &m, const std::string &prefix,
                          const std::string &suffix,
                          const std::string &grid_prefix,
                          const std::string &default_fitting_model) {
  auto function_prefix = prefix;
  auto function_suffix = suffix;
  function_prefix[0] = static_cast<char>(std::tolower(function_prefix[0]));
  function_suffix[0] = static_cast<char>(std::tolower(function_suffix[0]));

  m.def(
      (function_prefix + "_" + function_suffix).c_str(),
      &pyinterp::bicubic_3d<DataType, AxisType, Interpolator>, py::arg("grid"),
      py::arg("x"), py::arg("y"), py::arg("z"), py::arg("nx") = 3,
      py::arg("ny") = 3, py::arg("fitting_model") = default_fitting_model,
      py::arg("boundary") = "undef", py::arg("bounds_error") = false,
      py::arg("num_threads") = 0,
      (prefix + R"__doc__( gridded 3D interpolation.

A )__doc__" +
       function_prefix +
       R"__doc__( 2D interpolation is performed along the X and Y axes of the 3D grid,
and linearly along the Z axis between the two values obtained by the spatial
)__doc__" +
       function_prefix + R"__doc__( 2D interpolation.

Args:
    grid: Grid containing the values to be interpolated.
    x: X-values.
    y: Y-values.
    z: Z-values.
    nx: The number of X coordinate values required to perform the interpolation.
        Defaults to ``3``.
    ny: The number of Y coordinate values required to perform the interpolation.
        Defaults to ``3``.
    fitting_model: Type of interpolation to be performed. Defaults to `)__doc__" +
       default_fitting_model + R"__doc__(`
    boundary: Type of axis boundary management. Defaults to ``undef``.
    bounds_error: If True, when interpolated values are requested outside of the
        domain of the input axes (x,y), a ValueError is raised. If False, then
        value is set to NaN.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    Values interpolated.
  )__doc__")
          .c_str());
}

template <typename DataType, typename AxisType, typename Interpolator>
void implement_bicubic_4d(py::module &m, const std::string &prefix,
                          const std::string &suffix,
                          const std::string &grid_prefix,
                          const std::string &default_fitting_model) {
  auto function_prefix = prefix;
  auto function_suffix = suffix;
  function_prefix[0] = static_cast<char>(std::tolower(function_prefix[0]));
  function_suffix[0] = static_cast<char>(std::tolower(function_suffix[0]));

  m.def(
      (function_prefix + "_" + function_suffix).c_str(),
      &pyinterp::bicubic_4d<DataType, AxisType, Interpolator>, py::arg("grid"),
      py::arg("x"), py::arg("y"), py::arg("z"), py::arg("u"), py::arg("nx") = 3,
      py::arg("ny") = 3, py::arg("fitting_model") = default_fitting_model,
      py::arg("boundary") = "undef", py::arg("bounds_error") = false,
      py::arg("num_threads") = 0,
      (prefix + R"__doc__( gridded 4D interpolation

A )__doc__" +
       function_prefix +
       R"__doc__( 2D interpolation is performed along the X and Y axes of the 4D grid,
and linearly along the Z and U axes between the four values obtained by the
spatial )__doc__" +
       function_prefix + R"__doc__( 2D interpolation.

Args:
    grid: Grid containing the values to be interpolated.
    x: X-values.
    y: Y-values.
    z: Z-values.
    u: U-values.
    nx: The number of X coordinate values required to perform the interpolation.
        Defaults to ``3``.
    ny: The number of Y coordinate values required to perform the interpolation.
        Defaults to ``3``.
    fitting_model: Type of interpolation to be performed. Defaults to `)__doc__" +
       default_fitting_model + R"__doc__(`
    boundary: Type of axis boundary management. Defaults to ``undef``.
    bounds_error: If True, when interpolated values are requested outside of the
        domain of the input axes (x,y), a ValueError is raised. If False, then
        value is set to NaN.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    Values interpolated.
  )__doc__")
          .c_str());
}

void init_bicubic(py::module &m) {
  implement_bicubic<double, pyinterp::detail::math::Bicubic>(
      m, "Bicubic", "Float64", "bicubic");
  implement_bicubic<float, pyinterp::detail::math::Bicubic>(
      m, "Bicubic", "Float32", "bicubic");

  implement_bicubic_3d<double, double, pyinterp::detail::math::Bicubic>(
      m, "Bicubic", "Float64", "", "bicubic");
  implement_bicubic_3d<double, int64_t, pyinterp::detail::math::Bicubic>(
      m, "Bicubic", "Float64", "Temporal", "bicubic");

  implement_bicubic_3d<float, double, pyinterp::detail::math::Bicubic>(
      m, "Bicubic", "Float32", "", "bicubic");
  implement_bicubic_3d<float, int64_t, pyinterp::detail::math::Bicubic>(
      m, "Bicubic", "Float32", "Temporal", "bicubic");

  implement_bicubic_4d<double, double, pyinterp::detail::math::Bicubic>(
      m, "Bicubic", "Float64", "", "bicubic");
  implement_bicubic_4d<double, int64_t, pyinterp::detail::math::Bicubic>(
      m, "Bicubic", "Float64", "Temporal", "bicubic");

  implement_bicubic_4d<float, double, pyinterp::detail::math::Bicubic>(
      m, "Bicubic", "Float32", "", "bicubic");
  implement_bicubic_4d<float, int64_t, pyinterp::detail::math::Bicubic>(
      m, "Bicubic", "Float32", "Temporal", "bicubic");

  implement_bicubic<double, pyinterp::detail::math::Spline2D>(
      m, "Spline", "Float64", "c_spline");
  implement_bicubic<float, pyinterp::detail::math::Spline2D>(
      m, "Spline", "Float32", "c_spline");

  implement_bicubic_3d<double, double, pyinterp::detail::math::Spline2D>(
      m, "Spline", "Float64", "", "c_spline");
  implement_bicubic_3d<double, int64_t, pyinterp::detail::math::Spline2D>(
      m, "Spline", "Float64", "Temporal", "c_spline");

  implement_bicubic_3d<float, double, pyinterp::detail::math::Spline2D>(
      m, "Spline", "Float32", "", "c_spline");
  implement_bicubic_3d<float, int64_t, pyinterp::detail::math::Spline2D>(
      m, "Spline", "Float32", "Temporal", "c_spline");

  implement_bicubic_4d<double, double, pyinterp::detail::math::Spline2D>(
      m, "Spline", "Float64", "", "c_spline");
  implement_bicubic_4d<double, int64_t, pyinterp::detail::math::Spline2D>(
      m, "Spline", "Float64", "Temporal", "c_spline");

  implement_bicubic_4d<float, double, pyinterp::detail::math::Spline2D>(
      m, "Spline", "Float32", "", "c_spline");
  implement_bicubic_4d<float, int64_t, pyinterp::detail::math::Spline2D>(
      m, "Spline", "Float32", "Temporal", "c_spline");
}

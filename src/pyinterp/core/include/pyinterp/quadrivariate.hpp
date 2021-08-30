// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cctype>

#include "pyinterp/bivariate.hpp"
#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/detail/math/trivariate.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/grid.hpp"

namespace pyinterp {

/// Interpolator implemented
template <template <class> class Point, typename T>
using Bivariate4D = detail::math::Bivariate<Point, T>;

/// Get the fonction used to perform the interpolation on the U-Axis
template <typename T>
constexpr auto get_u_interpolation_method(const std::string& method)
    -> pyinterp::detail::math::z_method_t<T, T> {
  if (method == "linear") {
    return &pyinterp::detail::math::linear<T, T>;
  }
  if (method == "nearest") {
    return &pyinterp::detail::math::nearest<T, T>;
  }
  throw std::invalid_argument("unknown interpolation method: " + method);
}

/// Interpolation of quadrivariate function.
///
/// @tparam Point A type of point defining a point in space.
/// @tparam Coordinate Coordinate data type
/// @tparam AxisType Axis data type
/// @tparam Type Grid data type
template <template <class> class Point, typename Coordinate, typename AxisType,
          typename Type>
auto quadrivariate(const Grid4D<Type, AxisType>& grid,
                   const pybind11::array_t<Coordinate>& x,
                   const pybind11::array_t<Coordinate>& y,
                   const pybind11::array_t<AxisType>& z,
                   const pybind11::array_t<Coordinate>& u,
                   const Bivariate4D<Point, Coordinate>* interpolator,
                   const std::optional<std::string>& z_method,
                   const std::optional<std::string>& u_method,
                   const bool bounds_error, const size_t num_threads)
    -> pybind11::array_t<Coordinate> {
  pyinterp::detail::check_array_ndim("x", 1, x, "y", 1, y, "z", 1, z, "u", 1,
                                     u);
  pyinterp::detail::check_ndarray_shape("x", x, "y", y, "z", z, "u", u);
  auto z_interpolation_method =
      pyinterp::detail::math::get_z_interpolation_method(
          interpolator, z_method.value_or("linear"));
  auto u_interpolation_method =
      get_u_interpolation_method<Coordinate>(u_method.value_or("linear"));

  auto size = x.size();
  auto result =
      pybind11::array_t<Coordinate>(pybind11::array::ShapeContainer{size});
  auto _x = x.template unchecked<1>();
  auto _y = y.template unchecked<1>();
  auto _z = z.template unchecked<1>();
  auto _u = u.template unchecked<1>();
  auto _result = result.template mutable_unchecked<1>();

  {
    pybind11::gil_scoped_release release;

    // Captures the detected exceptions in the calculation function
    // (only the last exception captured is kept)
    auto except = std::exception_ptr(nullptr);

    // Access to the shared pointer outside the loop to avoid data races
    const auto& x_axis = *grid.x();
    const auto& y_axis = *grid.y();
    const auto& z_axis = *grid.z();
    const auto& u_axis = *grid.u();

    detail::dispatch(
        [&](size_t start, size_t end) {
          try {
            for (size_t ix = start; ix < end; ++ix) {
              auto x_indexes = x_axis.find_indexes(_x(ix));
              auto y_indexes = y_axis.find_indexes(_y(ix));
              auto z_indexes = z_axis.find_indexes(_z(ix));
              auto u_indexes = u_axis.find_indexes(_u(ix));

              if (x_indexes.has_value() && y_indexes.has_value() &&
                  z_indexes.has_value() && u_indexes.has_value()) {
                int64_t ix0;
                int64_t ix1;
                int64_t iy0;
                int64_t iy1;
                int64_t iz0;
                int64_t iz1;
                int64_t iu0;
                int64_t iu1;

                std::tie(ix0, ix1) = *x_indexes;
                std::tie(iy0, iy1) = *y_indexes;
                std::tie(iz0, iz1) = *z_indexes;
                std::tie(iu0, iu1) = *u_indexes;

                auto x0 = x_axis(ix0);

                // The fourth coordinate is not used by the 3D interpolator.
                auto p = Point<Coordinate>(
                    x_axis.is_angle()
                        ? detail::math::normalize_angle(_x(ix), x0, 360.0)
                        : _x(ix),
                    _y(ix), _z(ix));
                auto p0 = Point<Coordinate>(x0, y_axis(iy0), z_axis(iz0));
                auto p1 =
                    Point<Coordinate>(x_axis(ix1), y_axis(iy1), z_axis(iz1));

                auto u0 = pyinterp::detail::math::trivariate<Point, Coordinate>(
                    p, p0, p1,
                    static_cast<Coordinate>(grid.value(ix0, iy0, iz0, iu0)),
                    static_cast<Coordinate>(grid.value(ix0, iy1, iz0, iu0)),
                    static_cast<Coordinate>(grid.value(ix1, iy0, iz0, iu0)),
                    static_cast<Coordinate>(grid.value(ix1, iy1, iz0, iu0)),
                    static_cast<Coordinate>(grid.value(ix0, iy0, iz1, iu0)),
                    static_cast<Coordinate>(grid.value(ix0, iy1, iz1, iu0)),
                    static_cast<Coordinate>(grid.value(ix1, iy0, iz1, iu0)),
                    static_cast<Coordinate>(grid.value(ix1, iy1, iz1, iu0)),
                    interpolator, z_interpolation_method);

                auto u1 = pyinterp::detail::math::trivariate<Point, Coordinate>(
                    p, p0, p1,
                    static_cast<Coordinate>(grid.value(ix0, iy0, iz0, iu1)),
                    static_cast<Coordinate>(grid.value(ix0, iy1, iz0, iu1)),
                    static_cast<Coordinate>(grid.value(ix1, iy0, iz0, iu1)),
                    static_cast<Coordinate>(grid.value(ix1, iy1, iz0, iu1)),
                    static_cast<Coordinate>(grid.value(ix0, iy0, iz1, iu1)),
                    static_cast<Coordinate>(grid.value(ix0, iy1, iz1, iu1)),
                    static_cast<Coordinate>(grid.value(ix1, iy0, iz1, iu1)),
                    static_cast<Coordinate>(grid.value(ix1, iy1, iz1, iu1)),
                    interpolator, z_interpolation_method);

                _result(ix) = u_interpolation_method(_u(ix), u_axis(iu0),
                                                     u_axis(iu1), u0, u1);

              } else {
                if (bounds_error) {
                  if (!x_indexes.has_value()) {
                    Grid4D<Type, AxisType>::index_error(x_axis, _x(ix), "x");
                  }
                  if (!y_indexes.has_value()) {
                    Grid4D<Type, AxisType>::index_error(y_axis, _y(ix), "y");
                  }
                  if (!z_indexes.has_value()) {
                    Grid4D<Type, AxisType>::index_error(z_axis, _z(ix), "z");
                  }
                  Grid4D<Type, AxisType>::index_error(u_axis, _u(ix), "u");
                }
                _result(ix) = std::numeric_limits<Coordinate>::quiet_NaN();
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

/// Implementations of quadrivariate function.
///
/// @tparam Point A type of point defining a point in space.
/// @tparam Coordinate Coordinate data type
/// @tparam AxisType Axis data type
/// @tparam Type Grid data type
template <template <class> class Point, typename Coordinate, typename AxisType,
          typename Type>
void implement_quadrivariate(pybind11::module& m, const std::string& prefix,
                             const std::string& suffix) {
  auto function_suffix = suffix;
  function_suffix[0] = std::tolower(function_suffix[0]);
  m.def(("quadrivariate_" + function_suffix).c_str(),
        &quadrivariate<Point, Coordinate, AxisType, Type>,
        pybind11::arg("grid"), pybind11::arg("x"), pybind11::arg("y"),
        pybind11::arg("z"), pybind11::arg("u"), pybind11::arg("interpolator"),
        pybind11::arg("z_method") = pybind11::none(),
        pybind11::arg("u_method") = pybind11::none(),
        pybind11::arg("bounds_error") = false, pybind11::arg("num_threads") = 0,
        (R"__doc__(
Interpolate the values provided on the defined trivariate function.

Args:
    grid (pyinterp.core.)__doc__" +
         prefix + "Grid4D" + suffix +
         R"__doc__(): Grid containing the values to be interpolated.
    x (numpy.ndarray): X-values.
    y (numpy.ndarray): Y-values.
    z (numpy.ndarray): Z-values.
    u (numpy.ndarray): U-values.
    interpolator (pyinterp.core.)__doc__" +
         prefix + R"__doc__(BivariateInterpolator3D): 3D interpolator
        used to interpolate values on the surface (x, y, z). A linear
        interpolation is used to evaluate the surface (x, y, z, u).
    z_method (str, optional): The method of interpolation to perform on
      Z-axis. Supported are ``linear`` and ``nearest``. Default to
      ``linear``.
    u_method (str, optional): The method of interpolation to perform on
      U-axis. Supported are ``linear`` and ``nearest``. Default to
      ``linear``.
    bounds_error (bool, optional): If True, when interpolated values are
      requested outside of the domain of the input axes (x, y, z, u), a
      ValueError is raised. If False, then value is set to NaN.
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.
Returns:
    numpy.ndarray: Values interpolated.
)__doc__")
            .c_str());
}

}  // namespace pyinterp

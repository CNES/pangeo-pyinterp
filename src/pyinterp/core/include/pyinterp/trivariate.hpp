// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cctype>
#include <limits>
#include <string>

#include "pyinterp/bivariate.hpp"
#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/detail/math/trivariate.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/grid.hpp"

namespace pyinterp {

/// Interpolator implemented
template <template <class> class Point, typename T>
using Bivariate3D = detail::math::Bivariate<Point, T>;

template <template <class> class Point, typename Coordinate, typename AxisType,
          typename Type>
inline auto _trivariate(const Grid3D<Type, AxisType> &grid, const Coordinate &x,
                        const Coordinate &y, const AxisType &z,
                        const Axis<double> &x_axis, const Axis<double> &y_axis,
                        const Axis<AxisType> &z_axis,
                        const Bivariate3D<Point, Coordinate> *interpolator,
                        const detail::math::z_method_t<AxisType, Coordinate>
                            &z_interpolation_method,
                        const bool bounds_error) -> Coordinate {
  auto x_indexes = x_axis.find_indexes(x);
  auto y_indexes = y_axis.find_indexes(y);
  auto z_indexes = z_axis.find_indexes(z);

  if (x_indexes.has_value() && y_indexes.has_value() && z_indexes.has_value()) {
    auto [ix0, ix1] = *x_indexes;
    auto [iy0, iy1] = *y_indexes;
    auto [iz0, iz1] = *z_indexes;

    // The fourth coordinate is not used by the 3D interpolator.
    auto x0 = x_axis(ix0);
    auto p = Point<Coordinate>(x_axis.normalize_coordinate(x, x0), y, z);
    auto p0 = Point<Coordinate>(x0, y_axis(iy0), z_axis(iz0));
    auto p1 = Point<Coordinate>(x_axis(ix1), y_axis(iy1), z_axis(iz1));

    return pyinterp::detail::math::trivariate<Point, Coordinate>(
        p, p0, p1, static_cast<Coordinate>(grid.value(ix0, iy0, iz0)),
        static_cast<Coordinate>(grid.value(ix0, iy1, iz0)),
        static_cast<Coordinate>(grid.value(ix1, iy0, iz0)),
        static_cast<Coordinate>(grid.value(ix1, iy1, iz0)),
        static_cast<Coordinate>(grid.value(ix0, iy0, iz1)),
        static_cast<Coordinate>(grid.value(ix0, iy1, iz1)),
        static_cast<Coordinate>(grid.value(ix1, iy0, iz1)),
        static_cast<Coordinate>(grid.value(ix1, iy1, iz1)), interpolator,
        z_interpolation_method);
  }

  if (bounds_error) {
    if (!x_indexes.has_value()) {
      Grid3D<Type, AxisType>::index_error(x_axis, x, "x");
    }
    if (!y_indexes.has_value()) {
      Grid3D<Type, AxisType>::index_error(y_axis, y, "y");
    }
    Grid3D<Type, AxisType>::index_error(z_axis, z, "z");
  }
  return std::numeric_limits<Coordinate>::quiet_NaN();
}

/// Interpolation of bivariate function.
///
/// @tparam Point A type of point defining a point in space.
/// @tparam Coordinate Coordinate data type
/// @tparam AxisType Axis data type
/// @tparam Type Grid data type
template <template <class> class Point, typename Coordinate, typename AxisType,
          typename Type>
auto trivariate(const Grid3D<Type, AxisType> &grid,
                const pybind11::array_t<Coordinate> &x,
                const pybind11::array_t<Coordinate> &y,
                const pybind11::array_t<AxisType> &z,
                const Bivariate3D<Point, Coordinate> *interpolator,
                const std::optional<std::string> &z_method,
                const bool bounds_error, const size_t num_threads)
    -> pybind11::array_t<Coordinate> {
  pyinterp::detail::check_array_ndim("x", 1, x, "y", 1, y, "z", 1, z);
  pyinterp::detail::check_ndarray_shape("x", x, "y", y, "z", z);
  auto z_interpolation_method =
      pyinterp::detail::math::get_z_interpolation_method(
          interpolator, z_method.value_or("linear"));
  auto size = x.size();
  auto result =
      pybind11::array_t<Coordinate>(pybind11::array::ShapeContainer{size});
  auto _x = x.template unchecked<1>();
  auto _y = y.template unchecked<1>();
  auto _z = z.template unchecked<1>();
  auto _result = result.template mutable_unchecked<1>();

  {
    pybind11::gil_scoped_release release;

    // Captures the detected exceptions in the calculation function
    // (only the last exception captured is kept)
    auto except = std::exception_ptr(nullptr);

    // Access to the shared pointer outside the loop to avoid data races
    const auto &x_axis = *grid.x();
    const auto &y_axis = *grid.y();
    const auto &z_axis = *grid.z();

    detail::dispatch(
        [&](size_t start, size_t end) {
          try {
            for (size_t ix = start; ix < end; ++ix) {
              _result(ix) = _trivariate(grid, _x(ix), _y(ix), _z(ix), x_axis,
                                        y_axis, z_axis, interpolator,
                                        z_interpolation_method, bounds_error);
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

/// Implementations trivariate interpolation
///
/// @tparam Point A type of point defining a point in space.
/// @tparam Coordinate Coordinate data type
/// @tparam AxisType Axis data type
/// @tparam Type Grid data type
template <template <class> class Point, typename Coordinate, typename AxisType,
          typename Type>
void implement_trivariate(pybind11::module &m, const std::string &prefix,
                          const std::string &suffix) {
  auto function_suffix = suffix;
  function_suffix[0] = static_cast<char>(std::tolower(function_suffix[0]));
  m.def(("trivariate_" + function_suffix).c_str(),
        &trivariate<Point, Coordinate, AxisType, Type>, pybind11::arg("grid"),
        pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("z"),
        pybind11::arg("interpolator"),
        pybind11::arg("z_method") = pybind11::none(),
        pybind11::arg("bounds_error") = false, pybind11::arg("num_threads") = 0,
        R"__doc__(
Interpolate the values provided on the defined trivariate function.

Args:
    grid: Grid containing the values to be interpolated.
    x: X-values.
    y: Y-values.
    z: Z-values.
    interpolator: 3D interpolator used to interpolate values on the surface
        (x, y, z).
    z_method: The method of interpolation to perform on Z-axis. Supported are
        ``linear`` and ``nearest``. Default to ``linear``.
    bounds_error: If True, when interpolated values are requested outside of the
        domain of the input axes (x,y,z), a ValueError is raised. If False, then
        value is set to NaN.
    num_threads: The number of threads to use for the computation. If 0 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. Defaults to ``0``.
Returns:
    Values interpolated.
)__doc__");
}

}  // namespace pyinterp

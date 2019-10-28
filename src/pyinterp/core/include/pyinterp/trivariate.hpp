// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <cctype>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "pyinterp/bivariate.hpp"
#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/detail/math/trivariate.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/grid.hpp"

namespace pyinterp {

/// Interpolator implemented
template <template <class> class Point, typename T>
using Bivariate3D = detail::math::Bivariate<Point, T>;

/// Interpolation of bivariate function.
///
/// @tparam Coordinate The type of data used by the interpolators.
/// @tparam Type The type of data used by the numerical grid.
template <template <class> class Point, typename Coordinate, typename Type>
auto trivariate(const Grid3D<Type>& grid,
                const pybind11::array_t<Coordinate>& x,
                const pybind11::array_t<Coordinate>& y,
                const pybind11::array_t<Coordinate>& z,
                const Bivariate3D<Point, Coordinate>* interpolator,
                const bool bounds_error, const size_t num_threads)
    -> pybind11::array_t<Coordinate> {
  pyinterp::detail::check_array_ndim("x", 1, x, "y", 1, y);
  pyinterp::detail::check_ndarray_shape("x", x, "y", y);

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
    const auto& x_axis = *grid.x();
    const auto& y_axis = *grid.y();
    const auto& z_axis = *grid.z();

    detail::dispatch(
        [&](size_t start, size_t end) {
          try {
            for (size_t ix = start; ix < end; ++ix) {
              auto x_indexes = x_axis.find_indexes(_x(ix));
              auto y_indexes = y_axis.find_indexes(_y(ix));
              auto z_indexes = z_axis.find_indexes(_z(ix));

              if (x_indexes.has_value() && y_indexes.has_value() &&
                  z_indexes.has_value()) {
                int64_t ix0;
                int64_t ix1;
                int64_t iy0;
                int64_t iy1;
                int64_t iz0;
                int64_t iz1;

                std::tie(ix0, ix1) = *x_indexes;
                std::tie(iy0, iy1) = *y_indexes;
                std::tie(iz0, iz1) = *z_indexes;

                auto x0 = x_axis(ix0);

                _result(ix) =
                    pyinterp::detail::math::trivariate<Point, Coordinate>(
                        Point<Coordinate>(x_axis.is_angle()
                                              ? detail::math::normalize_angle(
                                                    _x(ix), x0, 360.0)
                                              : _x(ix),
                                          _y(ix), _z(ix)),
                        Point<Coordinate>(x0, y_axis(iy0), z_axis(iz0)),
                        Point<Coordinate>(x_axis(ix1), y_axis(iy1),
                                          z_axis(iz1)),
                        static_cast<Coordinate>(grid.value(ix0, iy0, iz0)),
                        static_cast<Coordinate>(grid.value(ix0, iy1, iz0)),
                        static_cast<Coordinate>(grid.value(ix1, iy0, iz0)),
                        static_cast<Coordinate>(grid.value(ix1, iy1, iz0)),
                        static_cast<Coordinate>(grid.value(ix0, iy0, iz1)),
                        static_cast<Coordinate>(grid.value(ix0, iy1, iz1)),
                        static_cast<Coordinate>(grid.value(ix1, iy0, iz1)),
                        static_cast<Coordinate>(grid.value(ix1, iy1, iz1)),
                        interpolator);

              } else {
                if (bounds_error) {
                  if (!x_indexes.has_value()) {
                    Grid3D<Type>::index_error(x_axis, _x(ix), "x");
                  }
                  if (!y_indexes.has_value()) {
                    Grid3D<Type>::index_error(y_axis, _y(ix), "y");
                  }
                  Grid3D<Type>::index_error(z_axis, _z(ix), "z");
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

template <template <class> class Point, typename Coordinate, typename Type>
void implement_trivariate(pybind11::module& m, const std::string& suffix) {
  auto function_suffix = suffix;
  function_suffix[0] = std::tolower(function_suffix[0]);
  m.def(("trivariate_" + function_suffix).c_str(),
        &trivariate<Point, Coordinate, Type>, pybind11::arg("grid"),
        pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("z"),
        pybind11::arg("interpolator"), pybind11::arg("bounds_error") = false,
        pybind11::arg("num_threads") = 0,
        (R"__doc__(
Interpolate the values provided on the defined trivariate function.

Args:
    grid (pyinterp.core.Grid3D)__doc__" +
         suffix +
         R"__doc__(): Grid containing the values to be interpolated.
    x (numpy.ndarray): X-values
    y (numpy.ndarray): Y-values
    z (numpy.ndarray): Z-values
    interpolator (pyinterp.core.BivariateInterpolator3D): 3D interpolator
        used to interpolate values on the surface (x, y).
    bounds_error (bool, optional): If True, when interpolated values are
      requested outside of the domain of the input axes (x,y,z), a ValueError
      is raised. If False, then value is set to NaN.
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.
Return:
    numpy.ndarray: Values interpolated
)__doc__")
            .c_str());
}

}  // namespace pyinterp

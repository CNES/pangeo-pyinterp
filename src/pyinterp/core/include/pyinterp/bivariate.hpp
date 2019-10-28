// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cctype>
#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/detail/math/bivariate.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/grid.hpp"

namespace pyinterp {

/// BivariateInterpolator implemented
template <template <class> class Point, typename T>
using BivariateInterpolator = detail::math::Bivariate<Point, T>;

/// Bilinear interpolation
template <template <class> class Point, typename T>
class Bilinear : public detail::math::Bilinear<Point, T> {
 public:
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    return pybind11::make_tuple();
  }

  static auto setstate(const pybind11::tuple & /*tuple*/) -> Bilinear {
    return Bilinear();
  }
};

/// Nearest interpolation
template <template <class> class Point, typename T>
class Nearest : public detail::math::Nearest<Point, T> {
 public:
  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    return pybind11::make_tuple();
  }

  static auto setstate(const pybind11::tuple & /*tuple*/) -> Nearest {
    return Nearest();
  }
};

/// InverseDistanceWeighting
template <template <class> class Point, typename T>
class InverseDistanceWeighting
    : public detail::math::InverseDistanceWeighting<Point, T> {
 public:
  using detail::math::InverseDistanceWeighting<Point,
                                               T>::InverseDistanceWeighting;

  [[nodiscard]] auto getstate() const -> pybind11::tuple {
    return pybind11::make_tuple(this->exp());
  }

  static auto setstate(const pybind11::tuple& tuple)
      -> InverseDistanceWeighting {
    if (tuple.size() != 1) {
      throw std::runtime_error("invalid state");
    }

    return InverseDistanceWeighting(tuple[0].cast<int>());
  }
};

/// Interpolation of bivariate function.
///
/// @tparam Coordinate The type of data used by the interpolators.
/// @tparam Type The type of data used by the numerical grid.
template <template <class> class Point, typename Coordinate, typename Type>
auto bivariate(const Grid2D<Type>& grid, const pybind11::array_t<Coordinate>& x,
               const pybind11::array_t<Coordinate>& y,
               const BivariateInterpolator<Point, Coordinate>* interpolator,
               const bool bounds_error, const size_t num_threads)
    -> pybind11::array_t<Coordinate> {
  pyinterp::detail::check_array_ndim("x", 1, x, "y", 1, y);
  pyinterp::detail::check_ndarray_shape("x", x, "y", y);

  auto size = x.size();
  auto result =
      pybind11::array_t<Coordinate>(pybind11::array::ShapeContainer{size});
  auto _x = x.template unchecked<1>();
  auto _y = y.template unchecked<1>();
  auto _result = result.template mutable_unchecked<1>();

  {
    pybind11::gil_scoped_release release;

    // Captures the detected exceptions in the calculation function
    // (only the last exception captured is kept)
    auto except = std::exception_ptr(nullptr);

    // Access to the shared pointer outside the loop to avoid data races
    const auto& x_axis = *grid.x();
    const auto& y_axis = *grid.y();

    detail::dispatch(
        [&](size_t start, size_t end) {
          try {
            for (size_t ix = start; ix < end; ++ix) {
              auto x_indexes = x_axis.find_indexes(_x(ix));
              auto y_indexes = y_axis.find_indexes(_y(ix));

              if (x_indexes.has_value() && y_indexes.has_value()) {
                int64_t ix0;
                int64_t ix1;
                int64_t iy0;
                int64_t iy1;

                std::tie(ix0, ix1) = *x_indexes;
                std::tie(iy0, iy1) = *y_indexes;

                auto x0 = x_axis(ix0);

                _result(ix) = interpolator->evaluate(
                    Point<Coordinate>(
                        x_axis.is_angle()
                            ? detail::math::normalize_angle(_x(ix), x0, 360.0)
                            : _x(ix),
                        _y(ix)),
                    Point<Coordinate>(x0, y_axis(iy0)),
                    Point<Coordinate>(x_axis(ix1), y_axis(iy1)),
                    static_cast<Coordinate>(grid.value(ix0, iy0)),
                    static_cast<Coordinate>(grid.value(ix0, iy1)),
                    static_cast<Coordinate>(grid.value(ix1, iy0)),
                    static_cast<Coordinate>(grid.value(ix1, iy1)));

              } else {
                if (bounds_error) {
                  if (!x_indexes.has_value()) {
                    Grid2D<Type>::index_error(x_axis, _x(ix), "x");
                  }
                  Grid2D<Type>::index_error(y_axis, _y(ix), "y");
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

template <template <class> class Point, typename T>
void implement_bivariate_interpolator(pybind11::module& m,
                                      const std::string& suffix) {
  using CoordinateSystem = BivariateInterpolator<Point, T>;

  /// Redirects virtual calls to Python
  class PyInterpolator : public CoordinateSystem {
   public:
    using CoordinateSystem::CoordinateSystem;

    auto evaluate(const Point<T>& p, const Point<T>& p0, const Point<T>& p1,
                  const T& q00, const T& q01, const T& q10, const T& q11) const
        -> T override {
      PYBIND11_OVERLOAD_PURE(T, CoordinateSystem, "evaluate", p, p0, p1, q00,
                             q01, q10, q11);
    }
  };

  /// BivariateInterpolator implemented here
  auto interpolator = pybind11::class_<CoordinateSystem, PyInterpolator>(
      m, ("BivariateInterpolator" + suffix).c_str(),
      ("Bilinear interpolation in a " + suffix + " space").c_str());

  pybind11::class_<Bilinear<Point, T>>(
      m, ("Bilinear" + suffix).c_str(), interpolator,
      ("Bilinear interpolation in a " + suffix + " space").c_str())
      .def(pybind11::init<>())
      .def(pybind11::pickle(
          [](const Bilinear<Point, T>& self) { return self.getstate(); },
          [](const pybind11::tuple& state) {
            return Bilinear<Point, T>::setstate(state);
          }));

  pybind11::class_<Nearest<Point, T>>(
      m, ("Nearest" + suffix).c_str(), interpolator,
      ("Nearest interpolation in a " + suffix + " space").c_str())
      .def(pybind11::init<>())
      .def(pybind11::pickle(
          [](const Nearest<Point, T>& self) { return self.getstate(); },
          [](const pybind11::tuple& state) {
            return Nearest<Point, T>::setstate(state);
          }));

  pybind11::class_<InverseDistanceWeighting<Point, T>>(
      m, ("InverseDistanceWeighting" + suffix).c_str(), interpolator,
      ("Inverse distance weighting interpolation in a " + suffix + " space")
          .c_str())
      .def(pybind11::init<int>(), pybind11::arg("p") = 2)
      .def(pybind11::pickle(
          [](const InverseDistanceWeighting<Point, T>& self) {
            return self.getstate();
          },
          [](const pybind11::tuple& state) {
            return InverseDistanceWeighting<Point, T>::setstate(state);
          }));
}

template <template <class> class Point, typename Coordinate, typename Type>
void implement_bivariate(pybind11::module& m, const std::string& suffix) {
  auto function_suffix = suffix;
  function_suffix[0] = std::tolower(function_suffix[0]);
  m.def(("bivariate_" + function_suffix).c_str(),
        &bivariate<Point, Coordinate, Type>, pybind11::arg("grid"),
        pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("interpolator"),
        pybind11::arg("bounds_error") = false, pybind11::arg("num_threads") = 0,
        (R"__doc__(
Interpolate the values provided on the defined bivariate function.

Args:
    grid (pyinterp.core.Grid2D)__doc__" +
         suffix +
         R"__doc__(): Grid containing the values to be interpolated.
    x (numpy.ndarray): X-values
    y (numpy.ndarray): Y-values
    interpolator (pyinterp.core.BivariateInterpolator2D): 2D interpolator
      used to interpolate.
    bounds_error (bool, optional): If True, when interpolated values are
      requested outside of the domain of the input axes (x,y), a ValueError
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

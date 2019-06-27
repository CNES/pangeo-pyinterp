#pragma once
#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/detail/math/trivariate.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/bivariate.hpp"
#include "pyinterp/grid.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace pyinterp {

/// Interpolator implemented
template <template <class> class Point, typename T>
using Bivariate3D = detail::math::Bivariate<Point, T>;

/// Interpolation of bivariate function.
///
/// @tparam Coordinate The type of data used by the interpolators.
/// @tparam Type The type of data used by the numerical grid.
template <template <class> class Point, typename Coordinate, typename Type>
class Trivariate : public Grid3D<Type> {
 public:
  using Grid3D<Type>::Grid3D;

  /// Interpolates data using the defined interpolation function.
  pybind11::array_t<Coordinate> evaluate(
      const pybind11::array_t<Coordinate>& x,
      const pybind11::array_t<Coordinate>& y,
      const pybind11::array_t<Coordinate>& z,
      const Bivariate3D<Point, Coordinate>* interpolator,
      const size_t num_threads) {
    auto size = x.size();
    auto result = pybind11::array_t<Coordinate>(pybind11::array::ShapeContainer{size});
    auto _x = x.template unchecked<1>();
    auto _y = y.template unchecked<1>();
    auto _z = z.template unchecked<1>();
    auto _result = result.template mutable_unchecked<1>();

    {
      pybind11::gil_scoped_release release;

      // Captures the detected exceptions in the calculation function
      // (only the last exception captured is kept)
      auto except = std::exception_ptr(nullptr);

      detail::dispatch(
          [&](size_t start, size_t end) {
            try {
              for (size_t ix = start; ix < end; ++ix) {
                auto x_indexes = this->x_.find_indexes(_x(ix));
                auto y_indexes = this->y_.find_indexes(_y(ix));
                auto z_indexes = this->z_.find_indexes(_z(ix));

                if (x_indexes.has_value() && y_indexes.has_value() &&
                    z_indexes.has_value()) {
                  int64_t ix0, ix1, iy0, iy1, iz0, iz1;
                  std::tie(ix0, ix1) = *x_indexes;
                  std::tie(iy0, iy1) = *y_indexes;
                  std::tie(iz0, iz1) = *z_indexes;

                  auto x0 = this->x_(ix0);

                  _result(ix) =
                      pyinterp::detail::math::trivariate<Point, Coordinate>(
                          Point<Coordinate>(
                              this->x_.is_angle()
                                  ? detail::math::normalize_angle(_x(ix), x0)
                                  : _x(ix),
                              _y(ix), _z(ix)),
                          Point<Coordinate>(this->x_(ix0), this->y_(iy0),
                                            this->z_(iz0)),
                          Point<Coordinate>(this->x_(ix1), this->y_(iy1),
                                            this->z_(iz1)),
                          static_cast<Coordinate>(this->ptr_(ix0, iy0, iz0)),
                          static_cast<Coordinate>(this->ptr_(ix0, iy1, iz0)),
                          static_cast<Coordinate>(this->ptr_(ix1, iy0, iz0)),
                          static_cast<Coordinate>(this->ptr_(ix1, iy1, iz0)),
                          static_cast<Coordinate>(this->ptr_(ix0, iy0, iz1)),
                          static_cast<Coordinate>(this->ptr_(ix0, iy1, iz1)),
                          static_cast<Coordinate>(this->ptr_(ix1, iy0, iz1)),
                          static_cast<Coordinate>(this->ptr_(ix1, iy1, iz1)),
                          interpolator);

                } else {
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

  /// Pickle support: set state
  static Trivariate setstate(const pybind11::tuple& tuple) {
    return Trivariate(Grid3D<Type>::setstate(tuple));
  }

 private:
  /// Construct a new instance from a serialized instance
  explicit Trivariate(Grid3D<Type>&& grid) : Grid3D<Type>(grid) {}
};

template <template <class> class Point, typename Coordinate, typename Type>
void init_trivariate(pybind11::module& m, const char* const class_name) {
  pybind11::class_<Trivariate<Point, Coordinate, Type>>(m, class_name,
                                                        R"__doc__(
Interpolation of trivariate functions
)__doc__")
      .def(pybind11::init<Axis, Axis, Axis, pybind11::array_t<Type>>(),
           pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("z"),
           pybind11::arg("array"),
           R"__doc__(
Default constructor

Args:
    x (pyinterp.core.Axis): X-Axis
    y (pyinterp.core.Axis): Y-Axis
    z (pyinterp.core.Axis): Z-Axis
    array (numpy.ndarray): Trivariate function
)__doc__")
      .def_property_readonly(
          "x",
          [](const Trivariate<Point, Coordinate, Type>& self) {
            return self.x();
          },
          R"__doc__(
Gets the X-Axis handled by this instance

Returns:
    pyinterp.core.Axis: X-Axis
)__doc__")
      .def_property_readonly(
          "y",
          [](const Trivariate<Point, Coordinate, Type>& self) {
            return self.y();
          },
          R"__doc__(
Gets the Y-Axis handled by this instance

Returns:
    pyinterp.core.Axis: Y-Axis
)__doc__")
      .def_property_readonly(
          "z",
          [](const Trivariate<Point, Coordinate, Type>& self) {
            return self.z();
          },
          R"__doc__(
Gets the Z-Axis handled by this instance

Returns:
    pyinterp.core.Axis: Z-Axis
)__doc__")
      .def_property_readonly(
          "array",
          [](const Trivariate<Point, Coordinate, Type>& self) {
            return self.array();
          },
          R"__doc__(
Gets the values handled by this instance

Returns:
    numpy.ndarray: values
)__doc__")
      .def("evaluate", &Trivariate<Point, Coordinate, Type>::evaluate,
           pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("z"),
           pybind11::arg("interpolator"), pybind11::arg("num_threads") = 0,
           R"__doc__(
Interpolate the values provided on the defined trivariate function.

Args:
    x (numpy.ndarray): X-values
    y (numpy.ndarray): Y-values
    z (numpy.ndarray): Z-values
    interpolator (pyinterp.core.BivariateInterpolator3D): 3D interpolator
        used to interpolate values on the surface (x, y).
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.
Return:
    numpy.ndarray: Values interpolated
)__doc__")
      .def(pybind11::pickle(
          [](const Trivariate<Point, Coordinate, Type>& self) {
            return self.getstate();
          },
          [](const pybind11::tuple& state) {
            return Trivariate<Point, Coordinate, Type>::setstate(state);
          }));
}

}  // namespace pyinterp

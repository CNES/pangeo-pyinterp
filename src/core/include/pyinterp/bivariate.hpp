#pragma once
#include "pyinterp/detail/math/bivariate.hpp"
#include "pyinterp/detail/thread.hpp"
#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/grid.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace pyinterp {

/// Interpolator implemented
template <template <class> class Point, typename T>
using Interpolator = detail::math::Bivariate<Point, T>;

/// Bilinear interpolation
template <template <class> class Point, typename T>
class Bilinear : public detail::math::Bilinear<Point, T> {
 public:
  pybind11::tuple getstate() const { return pybind11::make_tuple(); }

  static Bilinear setstate(const pybind11::tuple& tuple) { return Bilinear(); }
};

/// Nearest interpolation
template <template <class> class Point, typename T>
class Nearest : public detail::math::Nearest<Point, T> {
 public:
  pybind11::tuple getstate() const { return pybind11::make_tuple(); }

  static Nearest setstate(const pybind11::tuple& tuple) { return Nearest(); }
};

/// InverseDistanceWeighting
template <template <class> class Point, typename T>
class InverseDistanceWeighting
    : public detail::math::InverseDistanceWeighting<Point, T> {
 public:
  using detail::math::InverseDistanceWeighting<Point,
                                               T>::InverseDistanceWeighting;

  pybind11::tuple getstate() const { return pybind11::make_tuple(this->exp()); }

  static InverseDistanceWeighting setstate(const pybind11::tuple& tuple) {
    if (tuple.size() != 1) {
      throw std::runtime_error("invalid state");
    }

    return InverseDistanceWeighting(tuple[0].cast<int>());
  }
};

/// Interpolation of bivariate function.
template <template <class> class Point, typename T>
class Bivariate : public Grid2D<> {
 public:
  using Grid2D::Grid2D;

  /// Interpolates data using the defined interpolation function.
  pybind11::array_t<T> evaluate(const pybind11::array_t<T>& x,
                                const pybind11::array_t<T>& y,
                                const Interpolator<Point, T>* interpolator,
                                const size_t num_threads) {
    auto size = x.size();
    auto result = pybind11::array_t<T>(pybind11::array::ShapeContainer{size});
    auto _x = x.template unchecked<1>();
    auto _y = y.template unchecked<1>();
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
                auto x_indexes = x_.find_indexes(_x(ix));
                auto y_indexes = y_.find_indexes(_y(ix));

                if (x_indexes.has_value() && y_indexes.has_value()) {
                  int64_t ix0, ix1, iy0, iy1;
                  std::tie(ix0, ix1) = *x_indexes;
                  std::tie(iy0, iy1) = *y_indexes;

                  auto x0 = x_(ix0);

                  _result(ix) = interpolator->evaluate(
                      Point<T>(x_.is_angle()
                                   ? detail::math::normalize_angle(_x(ix), x0)
                                   : _x(ix),
                               _y(ix)),
                      Point<T>(x_(ix0), y_(iy0)), Point<T>(x_(ix1), y_(iy1)),
                      ptr_(ix0, iy0), ptr_(ix0, iy1), ptr_(ix1, iy0),
                      ptr_(ix1, iy1));

                } else {
                  _result(ix) = std::numeric_limits<T>::quiet_NaN();
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
  static Bivariate setstate(const pybind11::tuple& tuple) {
    return Bivariate(Grid2D<>::setstate(tuple));
  }

 private:
  /// Construct a new instance from a serialized instance
  Bivariate(Grid2D<>&& grid) : Grid2D(grid) {}
};

template <template <class> class Point, typename T>
void init_bivariate(pybind11::module& m) {
  using CoordinateSystem = Interpolator<Point, T>;

  /// Redirects virtual calls to Python
  class PyInterpolator : public CoordinateSystem {
   public:
    using CoordinateSystem::CoordinateSystem;

    double evaluate(const Point<T>& p, const Point<T>& p0, const Point<T>& p1,
                    const T& q00, const T& q01, const T& q10,
                    const T& q11) const override {
      PYBIND11_OVERLOAD_PURE(T, CoordinateSystem, "evaluate", p, p0, p1, q00,
                             q01, q10, q11);
    }
  };

  /// Interpolator implemented here
  auto interpolator =
      pybind11::class_<CoordinateSystem, PyInterpolator>(m, "Interpolator");

  pybind11::class_<Bilinear<Point, T>>(m, "Bilinear", interpolator)
      .def(pybind11::init<>())
      .def(pybind11::pickle(
          [](const Bilinear<Point, T>& self) { return self.getstate(); },
          [](const pybind11::tuple& state) {
            return Bilinear<Point, T>::setstate(state);
          }));

  pybind11::class_<Nearest<Point, T>>(m, "Nearest", interpolator)
      .def(pybind11::init<>())
      .def(pybind11::pickle(
          [](const Nearest<Point, T>& self) { return self.getstate(); },
          [](const pybind11::tuple& state) {
            return Nearest<Point, T>::setstate(state);
          }));

  pybind11::class_<InverseDistanceWeighting<Point, T>>(
      m, "InverseDistanceWeighting", interpolator)
      .def(pybind11::init<int>(), pybind11::arg("p") = 2)
      .def(pybind11::pickle(
          [](const InverseDistanceWeighting<Point, T>& self) {
            return self.getstate();
          },
          [](const pybind11::tuple& state) {
            return InverseDistanceWeighting<Point, T>::setstate(state);
          }));

  pybind11::class_<Bivariate<Point, T>>(m, "Bivariate")
      .def(pybind11::init<Axis, Axis, pybind11::array_t<T>>(),
           pybind11::arg("x"), pybind11::arg("y"), pybind11::arg("z"))
      .def("evaluate", &Bivariate<Point, T>::evaluate, pybind11::arg("x"),
           pybind11::arg("y"), pybind11::arg("interpolator"),
           pybind11::arg("num_threads") = 0)
      .def(pybind11::pickle(
          [](const Bivariate<Point, T>& self) { return self.getstate(); },
          [](const pybind11::tuple& state) {
            return Bivariate<Point, T>::setstate(state);
          }));
}

}  // namespace pyinterp

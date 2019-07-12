// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/bicubic.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pyinterp {

/// Loads the interpolation frame into memory
template <typename Type>
bool Bicubic<Type>::load_frame(const double x, const double y,
                               const Axis::Boundary boundary,
                               const bool bounds_error,
                               detail::math::XArray& frame) const {
  auto y_indexes =
      this->y_->find_indexes(y, static_cast<uint32_t>(frame.ny()), boundary);
  auto x_indexes =
      this->x_->find_indexes(x, static_cast<uint32_t>(frame.nx()), boundary);

  if (x_indexes.empty() || y_indexes.empty()) {
    if (bounds_error) {
      if (x_indexes.empty()) {
        Bicubic::index_error(*this->x_, static_cast<Type>(x), "x");
      }
      Bicubic::index_error(*this->y_, static_cast<Type>(y), "y");
    }
    return false;
  }

  auto x0 = (*this->x_)(x_indexes[0]);

  for (auto jx = 0; jx < frame.y().size(); ++jx) {
    frame.y(jx) = (*this->y_)(y_indexes[jx]);
  }

  for (auto ix = 0; ix < frame.x().size(); ++ix) {
    auto index = x_indexes[ix];
    auto value = (*this->x_)(index);

    if (this->x_->is_angle()) {
      value = detail::math::normalize_angle(value, x0);
    }
    frame.x(ix) = value;

    for (auto jx = 0; jx < frame.y().size(); ++jx) {
      frame.z(ix, jx) = static_cast<double>(this->ptr_(index, y_indexes[jx]));
    }
  }
  return frame.is_valid();
}

/// Evaluate the interpolation.
template <typename Type>
py::array_t<double> Bicubic<Type>::evaluate(
    const py::array_t<double>& x, const py::array_t<double>& y, size_t nx,
    size_t ny, FittingModel fitting_model, const Axis::Boundary boundary,
    const bool bounds_error, size_t num_threads) const {
  detail::check_array_ndim("x", 1, x, "y", 1, y);
  detail::check_ndarray_shape("x", x, "y", y);

  auto size = x.size();
  auto result = py::array_t<double>(py::array::ShapeContainer{size});

  auto _x = x.template unchecked<1>();
  auto _y = y.template unchecked<1>();
  auto _result = result.template mutable_unchecked<1>();
  auto interpolator =
      detail::math::Bicubic(Bicubic::interp_type(fitting_model));
  {
    py::gil_scoped_release release;

    // Captures the detected exceptions in the calculation function
    // (only the last exception captured is kept)
    auto except = std::exception_ptr(nullptr);

    detail::dispatch(
        [&](const size_t start, const size_t end) {
          auto frame = detail::math::XArray(nx, ny);
          auto acc = detail::gsl::Accelerator();

          try {
            for (size_t ix = start; ix < end; ++ix) {
              auto xi = _x(ix);
              auto yi = _y(ix);
              _result(ix) =
                  load_frame(xi, yi, boundary, bounds_error, frame)
                      ? interpolator.interpolate(this->x_->is_angle()
                                                     ? frame.normalize_angle(xi)
                                                     : xi,
                                                 yi, frame, acc)
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

}  // namespace pyinterp

template <typename Type>
void implement_bicubic(py::module& m, const char* const class_name) {
  py::class_<pyinterp::Bicubic<Type>>(m, class_name,
                                      R"__doc__(
Extension of cubic interpolation for interpolating data points on a
two-dimensional regular grid. The interpolated surface is smoother than
corresponding surfaces obtained by bilinear interpolation or
nearest-neighbor interpolation.
)__doc__")
      .def(
          py::init<std::shared_ptr<pyinterp::Axis>,
                   std::shared_ptr<pyinterp::Axis>, const py::array_t<Type>&>(),
          py::arg("x"), py::arg("y"), py::arg("array"),
          R"__doc__(
Default constructor

Args:
    x (pyinterp.core.Axis): X-Axis
    y (pyinterp.core.Axis): Y-Axis
    array (numpy.ndarray): Bivariate function
  )__doc__")
      .def_property_readonly(
          "x", [](const pyinterp::Bicubic<Type>& self) { return self.x(); },
          R"__doc__(
Gets the X-Axis handled by this instance

Returns:
    pyinterp.core.Axis: X-Axis
)__doc__")
      .def_property_readonly(
          "y", [](const pyinterp::Bicubic<Type>& self) { return self.y(); },
          R"__doc__(
Gets the Y-Axis handled by this instance

Returns:
    pyinterp.core.Axis: Y-Axis
)__doc__")
      .def_property_readonly(
          "array",
          [](const pyinterp::Bicubic<Type>& self) { return self.array(); },
          R"__doc__(
Gets the values handled by this instance

Returns:
    numpy.ndarray: values to interpolate
)__doc__")
      .def("evaluate", &pyinterp::Bicubic<Type>::evaluate, py::arg("x"),
           py::arg("y"), py::arg("nx") = 3, py::arg("ny") = 3,
           py::arg("fitting_model") = pyinterp::FittingModel::kCSpline,
           py::arg("boundary") = pyinterp::Axis::kUndef,
           py::arg("bounds_error") = false, py::arg("num_threads") = 0,
           R"__doc__(
Evaluate the interpolation.

Args:
    x (numpy.ndarray): X-values
    y (numpy.ndarray): Y-values
    nx (int, optional): The number of X coordinate values required to perform
        the interpolation. Defaults to ``3``.
    ny (int, optional): The number of Y coordinate values required to perform
        the interpolation. Defaults to ``3``.
    fitting_model (pyinterp.core.FittingModel, optional): Type of interpolation
        to be performed. Defaults to
        :py:data:`pyinterp.core.FittingModel.CSpline`
    boundary (pyinterp.core.Axis.Boundary, optional): Type of axis boundary
        management. Defaults to
        :py:data:`pyinterp.core.Axis.Boundary.kUndef`
    bounds_error (bool, optional): If True, when interpolated values are
        requested outside of the domain of the input axes (x,y), a ValueError
        is raised. If False, then value is set to Nan.
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.
Return:
    numpy.ndarray: Values interpolated
  )__doc__")
      .def_static("_setstate", &pyinterp::Bicubic<Type>::setstate,
                  py::arg("state"), R"__doc__(
Rebuild an instance from a registered state of this object.

Args:
  state: Registred state of this object
)__doc__")
      .def(py::pickle(
          [](const pyinterp::Bicubic<Type>& self) { return self.getstate(); },
          [](const py::tuple& tuple) {
            return new pyinterp::Bicubic(
                pyinterp::Bicubic<Type>::setstate(tuple));
          }));
}

void init_bicubic(py::module& m) {
  py::enum_<pyinterp::FittingModel>(m, "FittingModel", R"__doc__(
Bicubic fitting model
)__doc__")
      .value("Linear", pyinterp::FittingModel::kLinear,
             "*Linear interpolation*.")
      .value("Polynomial", pyinterp::FittingModel::kPolynomial,
             "*Polynomial interpolation*.")
      .value("CSpline", pyinterp::FittingModel::kCSpline,
             "*Cubic spline with natural boundary conditions*.")
      .value("CSplinePeriodic", pyinterp::FittingModel::kCSplinePeriodic,
             "*Cubic spline with periodic boundary conditions*.")
      .value("Akima", pyinterp::FittingModel::kAkima,
             "*Non-rounded Akima spline with natural boundary conditions*.")
      .value("AkimaPeriodic", pyinterp::FittingModel::kAkimaPeriodic,
             "*Non-rounded Akima spline with periodic boundary conditions*.")
      .value(
          "Steffen", pyinterp::FittingModel::kSteffen,
          "*Steffen’s method guarantees the monotonicity of data points. the "
          "interpolating function between the given*.");

  implement_bicubic<double>(m, "BicubicFloat64");
  implement_bicubic<float>(m, "BicubicFloat32");
}

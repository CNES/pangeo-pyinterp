#include "pyinterp/bicubic.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pyinterp {

/// Loads the interpolation frame into memory
bool Bicubic::load_frame(const double x, const double y,
                         const Axis::Boundary boundary,
                         detail::math::XArray& frame) const {
  auto y_indexes = y_.find_indexes(y, frame.ny(), boundary);
  auto x_indexes = x_.find_indexes(x, frame.nx(), boundary);

  if (x_indexes.empty() || y_indexes.empty()) {
    return false;
  }

  auto x0 = x_(x_indexes[0]);

  for (auto jx = 0; jx < frame.y().size(); ++jx) {
    frame.y(jx) = y_(y_indexes[jx]);
  }

  for (auto ix = 0; ix < frame.x().size(); ++ix) {
    auto index = x_indexes[ix];
    auto value = x_(index);

    if (this->x_.is_angle()) {
      value = detail::math::normalize_angle(value, x0);
    }
    frame.x(ix) = value;

    for (auto jx = 0; jx < frame.y().size(); ++jx) {
      frame.z(ix, jx) = ptr_(index, y_indexes[jx]);
    }
  }
  return frame.is_valid();
}

/// Evaluate the interpolation.
py::array_t<double> Bicubic::evaluate(const py::array_t<double>& x,
                                      const py::array_t<double>& y, size_t nx,
                                      size_t ny, Type type,
                                      const Axis::Boundary boundary,
                                      size_t num_threads) const {
  detail::check_array_ndim("x", 1, x, "y", 1, y);
  detail::check_ndarray_shape("x", x, "y", y);

  auto size = x.size();
  auto result = py::array_t<double>(py::array::ShapeContainer{size});

  auto _x = x.template unchecked<1>();
  auto _y = y.template unchecked<1>();
  auto _result = result.template mutable_unchecked<1>();
  auto interpolator = detail::math::Bicubic(Bicubic::interp_type(type));
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
                  load_frame(xi, yi, boundary, frame)
                      ? interpolator.interpolate(
                            x_.is_angle() ? frame.normalize_angle(xi) : xi, yi,
                            frame, acc)
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

void init_bicubic(py::module& m) {
  auto bicubic = py::class_<pyinterp::Bicubic>(m, "Bicubic",
                                               R"__doc__(
Extension of cubic interpolation for interpolating data points on a
two-dimensional regular grid. The interpolated surface is smoother than
corresponding surfaces obtained by bilinear interpolation or
nearest-neighbor interpolation.
)__doc__");

  py::enum_<pyinterp::Bicubic::Type>(bicubic, "Type", R"__doc__(
Bicubic fitting model
)__doc__")
      .value("kLinear", pyinterp::Bicubic::kLinear, "*Linear interpolation*.")
      .value("kPolynomial", pyinterp::Bicubic::kPolynomial,
             "*Polynomial interpolation*.")
      .value("kCSpline", pyinterp::Bicubic::kCSpline,
             "*Cubic spline with natural boundary conditions*.")
      .value("kCSplinePeriodic", pyinterp::Bicubic::kCSplinePeriodic,
             "*Cubic spline with periodic boundary conditions*.")
      .value("kAkima", pyinterp::Bicubic::kAkima,
             "*Non-rounded Akima spline with natural boundary conditions*.")
      .value("kAkimaPeriodic", pyinterp::Bicubic::kAkimaPeriodic,
             "*Non-rounded Akima spline with periodic boundary conditions*.")
      .value(
          "kSteffen", pyinterp::Bicubic::kSteffen,
          "*Steffen’s method guarantees the monotonicity of data points. the "
          "interpolating function between the given*.");

  bicubic
      .def(py::init<pyinterp::Axis, pyinterp::Axis,
                    const py::array_t<double>&>(),
           py::arg("x"), py::arg("y"), py::arg("array"),
           R"__doc__(
Default constructor

Args:
    x (pyinterp.core.Axis): X-Axis
    y (pyinterp.core.Axis): Y-Axis
    array (numpy.ndarray): Bivariate function
  )__doc__")
      .def_property_readonly(
          "x", [](const pyinterp::Bicubic& self) { return self.x(); },
          R"__doc__(
Gets the X-Axis handled by this instance

Returns:
    pyinterp.core.Axis: X-Axis
)__doc__")
      .def_property_readonly(
          "y", [](const pyinterp::Bicubic& self) { return self.y(); },
          R"__doc__(
Gets the Y-Axis handled by this instance

Returns:
    pyinterp.core.Axis: Y-Axis
)__doc__")
      .def_property_readonly(
          "array", [](const pyinterp::Bicubic& self) { return self.array(); },
          R"__doc__(
Gets the values handled by this instance

Returns:
    numpy.ndarray: values
)__doc__")
      .def("evaluate", &pyinterp::Bicubic::evaluate, py::arg("x"), py::arg("y"),
           py::arg("nx") = 3, py::arg("ny") = 3,
           py::arg("type") = pyinterp::Bicubic::kCSpline,
           py::arg("boundary") = pyinterp::Axis::kUndef,
           py::arg("num_threads") = 0, R"__doc__(
Evaluate the interpolation.

Args:
    x (numpy.ndarray): X-values
    y (numpy.ndarray): Y-values
    nx (int, optional): The number of X coordinate values required to perform
        the interpolation. Defaults to ``3``.
    ny (int, optional): The number of Y coordinate values required to perform
        the interpolation. Defaults to ``3``.
    type (pyinterp.core.Bicubic.Type, optional): Type of interpolation
        to be performed. Defaults to
        :py:data:`pyinterp.core.Bicubic.Type.kCSpline`
    boundary (pyinterp.core.Axis.Boundary, optional): Type of axis boundary
        management. Defaults to
        :py:data:`pyinterp.core.Axis.Boundary.kUndef`
    num_threads (int, optional): The number of threads to use for the
        computation. If 0 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging.
        Defaults to ``0``.
Return:
    numpy.ndarray: Values interpolated
  )__doc__")
      .def(py::pickle(
          [](const pyinterp::Bicubic& self) { return self.getstate(); },
          [](const py::tuple& tuple) {
            return new pyinterp::Bicubic(pyinterp::Bicubic::setstate(tuple));
          }));
}

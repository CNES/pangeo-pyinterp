// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/interpolate1d.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "pyinterp/detail/axis.hpp"

namespace py = pybind11;

void init_interpolate1d(py::module& core) {
  core.def("interpolate1d", &pyinterp::interpolate_1d, R"__doc__(
Interpolate a 1D function

Args:
    x: Axis of the function to be interpolated
    y: Values of the function to be interpolated
    xi: Coordinate of the value to be interpolated
    half_window_size: Size of the half window. Default is 3.
    bounds_error: If true, an exception is raised if the value to be
        interpolated is out of the range of the axis.
    kind: Type of spline to be used. Default is cubic spline. Possible values
        are: ``linear``, ``c_spline``, ``c_spline_periodic``, ``akima``,
        ``akima_periodic``, ``steffen``.
Returns:
    The interpolated value
)__doc__",
           py::arg("x"), py::arg("y"), py::arg("xi"),
           py::arg("half_window_size") = 3, py::arg("bounds_error") = true,
           py::arg("kind") = "c_spline",
           py::call_guard<py::gil_scoped_release>());
}

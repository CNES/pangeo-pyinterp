// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/statistics.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
namespace statistics = pyinterp::statistics;

template <typename Type>
void implement_statistics(py::module& m, const std::string& suffix) {
  py::class_<statistics::Binned<Type>>(m, ("Binned" + suffix).c_str(),
                                       R"__doc__(
Handles the calculation of binned statistics of a geographic grid.
)__doc__")
      .def(py::init<std::shared_ptr<pyinterp::Axis>,
                    std::shared_ptr<pyinterp::Axis>>(),
           py::arg("x"), py::arg("y"), R"__doc__(
Default constructor

Args:
    x (pyinterp.core.Axis): Definition of the binned values for the X axis of
        the grid.
    y (pyinterp.core.Axis): Definition of the binned values for the Y axis of
        the grid.
)__doc__")
      .def_property_readonly(
          "x", [](const statistics::Binned<Type>& self) { return self.x(); },
          R"__doc__(
Gets the binned values for the X Axis of the grid

Return:
    pyinterp.core.Axis: X-Axis
)__doc__")
      .def_property_readonly(
          "y", [](const statistics::Binned<Type>& self) { return self.y(); },
          R"__doc__(
Gets the binned values for the Y Axis of the grid

Return:
    pyinterp.core.Axis: Y-Axis
)__doc__")
      .def("clear", &statistics::Binned<Type>::clear, "Reset the statistics")
      .def("count", &statistics::Binned<Type>::count,
           R"__doc__(
Get the number of samples pushed into the bin.

Return:
    numpy.array: number of samples binned for the defined grid.
)__doc__")
      .def("kurtosis", &statistics::Binned<Type>::kurtosis,
           R"__doc__(
Get the kurtosis of samples pushed into the bin.

Return:
    numpy.array: kurtosis of values binned for the defined grid.
)__doc__")
      .def("max", &statistics::Binned<Type>::max,
           R"__doc__(
Get the maximum of values pushed into the bin.

Return:
    numpy.array: maximum of values binned for the defined grid.
)__doc__")
      .def("mean", &statistics::Binned<Type>::mean,
           R"__doc__(
Get the mean of values pushed into the bin.

Return:
    numpy.array: mean of values binned for the defined grid.
)__doc__")
      .def("median", &statistics::Binned<Type>::median,
           R"__doc__(
Get the median of values pushed into the bin.

Return:
    numpy.array: median of values binned for the defined grid.
)__doc__")
      .def("min", &statistics::Binned<Type>::min,
           R"__doc__(
Get the minimum of values pushed into the bin.

Return:
    numpy.array: minimum of values binned for the defined grid.
)__doc__")
      .def("push", &statistics::Binned<Type>::push, R"__doc__(
Push new samples into the defined grid.

Args:
    x (numpy.ndarray): X coordinates of the values to push.
    y (numpy.ndarray): Y coordinates of the values to push.
    z (numpy.ndarray): New samples to push
)__doc__")
      .def("skewness", &statistics::Binned<Type>::skewness,
           R"__doc__(
Get the skewness of values pushed into the bin.

Return:
    numpy.array: skewness of values binned for the defined grid.
)__doc__")
      .def("variance", &statistics::Binned<Type>::variance,
           R"__doc__(
Get the variance of values pushed into the bin.

Return:
    numpy.array: variance of values binned for the defined grid.
)__doc__");
}

void init_statistics(py::module& m) {
  implement_statistics<double>(m, "Float64");
  implement_statistics<float>(m, "Float32");
}

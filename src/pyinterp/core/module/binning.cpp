// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/binning.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
namespace binning = pyinterp::binning;

template <typename Type>
void implement_binning_nearest(py::module& m, const std::string& suffix) {
  py::class_<binning::Nearest<Type>>(m, ("Nearest" + suffix).c_str(),
                                       R"__doc__(
Discretizes the data into a regular grid (computes a binned approximation)
using the nearest binning technique.
)__doc__")
      .def(py::init<std::shared_ptr<pyinterp::Axis>,
                    std::shared_ptr<pyinterp::Axis>>(),
           py::arg("x"), py::arg("y"), R"__doc__(
Default constructor

Args:
    x (pyinterp.core.Axis): Definition of the bin edges for the X axis of
        the grid.
    y (pyinterp.core.Axis): Definition of the bin edges for the Y axis of
        the grid.
)__doc__")
      .def_property_readonly(
          "x", [](const binning::Nearest<Type>& self) { return self.x(); },
          R"__doc__(
Gets the bin edges for the X Axis of the grid

Return:
    pyinterp.core.Axis: X-Axis
)__doc__")
      .def_property_readonly(
          "y", [](const binning::Nearest<Type>& self) { return self.y(); },
          R"__doc__(
Gets the bin edges for the Y Axis of the grid

Return:
    pyinterp.core.Axis: Y-Axis
)__doc__")
      .def("clear", &binning::Nearest<Type>::clear, "Reset the statistics")
      .def("count", &binning::Nearest<Type>::count,
           R"__doc__(
Get the number of samples pushed into the bin.

Return:
    numpy.ndarray: number of samples binned for the defined grid.
)__doc__")
      .def("kurtosis", &binning::Nearest<Type>::kurtosis,
           R"__doc__(
Get the kurtosis of samples pushed into the bin.

Return:
    numpy.ndarray: kurtosis of values binned for the defined grid.
)__doc__")
      .def("max", &binning::Nearest<Type>::max,
           R"__doc__(
Get the maximum of values pushed into the bin.

Return:
    numpy.ndarray: maximum of values binned for the defined grid.
)__doc__")
      .def("mean", &binning::Nearest<Type>::mean,
           R"__doc__(
Get the mean of values pushed into the bin.

Return:
    numpy.ndarray: mean of values binned for the defined grid.
)__doc__")
      .def("median", &binning::Nearest<Type>::median,
           R"__doc__(
Get the median of values pushed into the bin.

Return:
    numpy.ndarray: median of values binned for the defined grid.
)__doc__")
      .def("min", &binning::Nearest<Type>::min,
           R"__doc__(
Get the minimum of values pushed into the bin.

Return:
    numpy.ndarray: minimum of values binned for the defined grid.
)__doc__")
      .def("push", &binning::Nearest<Type>::push, R"__doc__(
Push new samples into the defined grid.

Args:
    x (numpy.ndarray): X coordinates of the values to push.
    y (numpy.ndarray): Y coordinates of the values to push.
    z (numpy.ndarray): New samples to push
)__doc__")
      .def("skewness", &binning::Nearest<Type>::skewness,
           R"__doc__(
Get the skewness of values pushed into the bin.

Return:
    numpy.ndarray: skewness of values binned for the defined grid.
)__doc__")
      .def("variance", &binning::Nearest<Type>::variance,
           R"__doc__(
Get the variance of values pushed into the bin.

Return:
    numpy.ndarray: variance of values binned for the defined grid.
)__doc__");
}

void init_binning(py::module& m) {
  implement_binning_nearest<double>(m, "Float64");
  implement_binning_nearest<float>(m, "Float32");
}

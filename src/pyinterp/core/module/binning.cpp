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
void implement_binning_nearest_bivariate(py::module& m,
                                         const std::string& suffix) {
  py::class_<binning::NearestBivariate<Type>>(
      m, ("NearestBivariate" + suffix).c_str(),
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
          "x",
          [](const binning::NearestBivariate<Type>& self) { return self.x(); },
          R"__doc__(
Gets the bin edges for the X Axis of the grid

Return:
    pyinterp.core.Axis: X-Axis
)__doc__")
      .def_property_readonly(
          "y",
          [](const binning::NearestBivariate<Type>& self) { return self.y(); },
          R"__doc__(
Gets the bin edges for the Y Axis of the grid

Return:
    pyinterp.core.Axis: Y-Axis
)__doc__")
      .def("clear", &binning::NearestBivariate<Type>::clear,
           "Reset the statistics")
      .def("count", &binning::NearestBivariate<Type>::count,
           R"__doc__(
Compute the count of points within each bin.

Return:
    numpy.ndarray: count of points within each bin.
)__doc__")
      .def("kurtosis", &binning::NearestBivariate<Type>::kurtosis,
           R"__doc__(
Compute the kurtosis of values for points within each bin.

Return:
    numpy.ndarray: kurtosis of values for points within each bin.
)__doc__")
      .def("max", &binning::NearestBivariate<Type>::max,
           R"__doc__(
Compute the maximum of values for points within each bin.

Return:
    numpy.ndarray: maximum of values for points within each bin.
)__doc__")
      .def("mean", &binning::NearestBivariate<Type>::mean,
           R"__doc__(
Compute the mean of values for points within each bin.

Return:
    numpy.ndarray: mean of values for points within each bin.
)__doc__")
      .def("median", &binning::NearestBivariate<Type>::median,
           R"__doc__(
Compute the median of values for points within each bin.

Return:
    numpy.ndarray: mdeian of values for points within each bin.
)__doc__")
      .def("min", &binning::NearestBivariate<Type>::min,
           R"__doc__(
Compute the minimum of values for points within each bin.

Return:
    numpy.ndarray: minimum of values for points within each bin.
)__doc__")
      .def("push", &binning::NearestBivariate<Type>::push, R"__doc__(
Push new samples into the defined bins.

Args:
    x (numpy.ndarray): X coordinates of the values to push.
    y (numpy.ndarray): Y coordinates of the values to push.
    z (numpy.ndarray): New samples to push
)__doc__")
      .def("sum", &binning::NearestBivariate<Type>::sum,
           R"__doc__(
Compute the sum of values for points within each bin.

Return:
    numpy.ndarray: sum of values for points within each bin.
)__doc__")
      .def("skewness", &binning::NearestBivariate<Type>::skewness,
           R"__doc__(
Compute the skewness of values for points within each bin.

Return:
    numpy.ndarray: skewness of values for points within each bin.
)__doc__")
      .def("variance", &binning::NearestBivariate<Type>::variance,
           R"__doc__(
Compute the variance of values for points within each bin.

Return:
    numpy.ndarray: variance of values for points within each bin.
)__doc__");
}

void init_binning(py::module& m) {
  implement_binning_nearest_bivariate<double>(m, "Float64");
  implement_binning_nearest_bivariate<float>(m, "Float32");
}

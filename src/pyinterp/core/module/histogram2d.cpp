// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/histogram2d.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename Type>
void implement_histogram_2d(py::module& m, const std::string& suffix) {
  PYBIND11_NUMPY_DTYPE(pyinterp::detail::math::Bin<Type>, value, weight);
  py::class_<pyinterp::Histogram2D<Type>>(m, ("Histogram2D" + suffix).c_str(),
                                          R"__doc__(
Group a number of more or less continuous values into a smaller number of
"bins" located on a grid.
)__doc__")
      .def(py::init<std::shared_ptr<pyinterp::Axis<double>>,
                    std::shared_ptr<pyinterp::Axis<double>>,
                    const std::optional<size_t>&>(),
           py::arg("x"), py::arg("y"), py::arg("bins") = py::none(),
           R"__doc__(
Default constructor

Args:
    x (pyinterp.core.Axis): Definition of the bin centers for the X axis of
        the grid.
    y (pyinterp.core.Axis): Definition of the bin centers for the Y axis of
        the grid.
    bins (int, optional): Maximum number of bins per pixel to use to calculate
        the histogram.
)__doc__")
      .def_property_readonly(
          "x", [](const pyinterp::Histogram2D<Type>& self) { return self.x(); },
          R"__doc__(
Gets the bin centers for the X Axis of the grid

Return:
    pyinterp.core.Axis: X-Axis
)__doc__")
      .def_property_readonly(
          "y", [](const pyinterp::Histogram2D<Type>& self) { return self.y(); },
          R"__doc__(
Gets the bin centers for the Y Axis of the grid

Return:
    pyinterp.core.Axis: Y-Axis
)__doc__")
      .def("clear", &pyinterp::Histogram2D<Type>::clear, "Reset the statistics")
      .def("count", &pyinterp::Histogram2D<Type>::count,
           R"__doc__(
Compute the count of points within each bin.

Return:
    numpy.ndarray: count of points within each bin.
)__doc__")
      .def("histograms", &pyinterp::Histogram2D<Type>::histograms,
           R"__doc__(
Compute the histograms for each bin.

Return:
    numpy.ndarray: histograms for each bin.
)__doc__")
      .def("kurtosis", &pyinterp::Histogram2D<Type>::kurtosis,
           R"__doc__(
Compute the kurtosis of values for points within each bin.

Return:
    numpy.ndarray: kurtosis of values for points within each bin.
)__doc__")
      .def("quantile", &pyinterp::Histogram2D<Type>::quantile,
           R"__doc__(
Compute the quantile of points within each bin.

Args:
    q (float): Quantile to compute.

Return:
    numpy.ndarray: quantile of points within each bin.
)__doc__",
           py::arg("q") = 0.5)
      .def("max", &pyinterp::Histogram2D<Type>::max,
           R"__doc__(
Compute the maximum of values for points within each bin.

Return:
    numpy.ndarray: maximum of values for points within each bin.
)__doc__")
      .def("mean", &pyinterp::Histogram2D<Type>::mean,
           R"__doc__(
Compute the mean of values for points within each bin.

Return:
    numpy.ndarray: mean of values for points within each bin.
)__doc__")
      .def("min", &pyinterp::Histogram2D<Type>::min,
           R"__doc__(
Compute the minimum of values for points within each bin.

Return:
    numpy.ndarray: minimum of values for points within each bin.
)__doc__")
      .def("push", &pyinterp::Histogram2D<Type>::push, py::arg("x"),
           py::arg("y"), py::arg("z"), R"__doc__(
Push new samples into the defined bins.

Args:
    x (numpy.ndarray): X coordinates of the values to push.
    y (numpy.ndarray): Y coordinates of the values to push.
    z (numpy.ndarray): New samples to push
)__doc__")
      .def("sum_of_weights", &pyinterp::Histogram2D<Type>::sum_of_weights,
           R"__doc__(
Compute the sum of weights for points within each bin.

Return:
    numpy.ndarray: sum of weights for points within each bin.
)__doc__")
      .def("skewness", &pyinterp::Histogram2D<Type>::skewness,
           R"__doc__(
Compute the skewness of values for points within each bin.

Return:
    numpy.ndarray: skewness of values for points within each bin.
)__doc__")
      .def("variance", &pyinterp::Histogram2D<Type>::variance,
           R"__doc__(
Compute the variance of values for points within each bin.

Return:
    numpy.ndarray: variance of values for points within each bin.
)__doc__")
      .def("__iadd__", &pyinterp::Histogram2D<Type>::operator+=,
           py::call_guard<py::gil_scoped_release>())
      .def(py::pickle(
          [](const pyinterp::Histogram2D<Type>& self) {
            return self.getstate();
          },
          [](const py::tuple& state) {
            return pyinterp::Histogram2D<Type>::setstate(state);
          }));
}

void init_histogram2d(py::module& m) {
  implement_histogram_2d<double>(m, "Float64");
  implement_histogram_2d<float>(m, "Float32");
}

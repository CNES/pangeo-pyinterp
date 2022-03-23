// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/histogram2d.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename Type>
void implement_histogram_2d(py::module &m, const std::string &suffix) {
  PYBIND11_NUMPY_DTYPE(pyinterp::detail::math::Bin<Type>, value, weight);
  py::class_<pyinterp::Histogram2D<Type>>(m, ("Histogram2D" + suffix).c_str(),
                                          ("Histogram2D" + suffix +
                                           "(self, x: pyinterp.core.Axis,"
                                           " y: pyinterp.core.Axis,"
                                           " bins: Optional[int] = None)" +
                                           R"__doc__(

Group a number of more or less continuous values into a smaller number of
"bins" located on a grid.

Args:
    x: Definition of the bin centers for the X axis of the grid.
    y: Definition of the bin centers for the Y axis of the grid.
    bins: Maximum number of bins per pixel to use to calculate the histogram.
        Default is None, which means that the number of bins is computed
        automatically.
)__doc__")
                                              .c_str())
      .def(py::init<std::shared_ptr<pyinterp::Axis<double>>,
                    std::shared_ptr<pyinterp::Axis<double>>,
                    const std::optional<size_t> &>(),
           py::arg("x"), py::arg("y"), py::arg("bins") = std::nullopt)
      .def_property_readonly(
          "x", [](const pyinterp::Histogram2D<Type> &self) { return self.x(); },
          R"__doc__(
Gets the bin centers for the X Axis of the grid.

Returns:
    X-Axis.
)__doc__")
      .def_property_readonly(
          "y", [](const pyinterp::Histogram2D<Type> &self) { return self.y(); },
          R"__doc__(
Gets the bin centers for the Y Axis of the grid.

Returns:
    pyinterp.core.Axis: Y-Axis.
)__doc__")
      .def("clear", &pyinterp::Histogram2D<Type>::clear,
           "Reset the statistics.")
      .def("count", &pyinterp::Histogram2D<Type>::count,
           R"__doc__(
Compute the count of points within each bin.

Returns:
    Count of points within each bin.
)__doc__")
      .def("histograms", &pyinterp::Histogram2D<Type>::histograms,
           R"__doc__(
Compute the histograms for each bin.

Returns:
    Histograms for each bin.
)__doc__")
      .def("kurtosis", &pyinterp::Histogram2D<Type>::kurtosis,
           R"__doc__(
Compute the kurtosis of values for points within each bin.

Returns:
    numpy.ndarray: kurtosis of values for points within each bin.
)__doc__")
      .def("quantile", &pyinterp::Histogram2D<Type>::quantile,
           R"__doc__(
Compute the quantile of points within each bin.

Args:
    q: Quantile to compute.

Returns:
    Quantile of points within each bin.
)__doc__",
           py::arg("q") = 0.5)
      .def("max", &pyinterp::Histogram2D<Type>::max,
           R"__doc__(
Compute the maximum of values for points within each bin.

Returns:
    Maximum of values for points within each bin.
)__doc__")
      .def("mean", &pyinterp::Histogram2D<Type>::mean,
           R"__doc__(
Compute the mean of values for points within each bin.

Returns:
    Mean of values for points within each bin.
)__doc__")
      .def("min", &pyinterp::Histogram2D<Type>::min,
           R"__doc__(
Compute the minimum of values for points within each bin.

Returns:
    Minimum of values for points within each bin.
)__doc__")
      .def("push", &pyinterp::Histogram2D<Type>::push, py::arg("x"),
           py::arg("y"), py::arg("z"), R"__doc__(
Push new samples into the defined bins.

Args:
    x: X coordinates of the values to push.
    y: Y coordinates of the values to push.
    z: New samples to push.
)__doc__")
      .def("sum_of_weights", &pyinterp::Histogram2D<Type>::sum_of_weights,
           R"__doc__(
Compute the sum of weights for points within each bin.

Returns:
    Sum of weights for points within each bin.
)__doc__")
      .def("skewness", &pyinterp::Histogram2D<Type>::skewness,
           R"__doc__(
Compute the skewness of values for points within each bin.

Returns:
    Skewness of values for points within each bin.
)__doc__")
      .def("variance", &pyinterp::Histogram2D<Type>::variance,
           R"__doc__(
Compute the variance of values for points within each bin.

Returns:
    Variance of values for points within each bin.
)__doc__")
      .def(
          "__copy__",
          [](const pyinterp::Histogram2D<Type> &self) {
            return pyinterp::Histogram2D<Type>(self);
          },
          "Implements the shallow copy operation.",
          py::call_guard<py::gil_scoped_release>())
      .def("__iadd__", &pyinterp::Histogram2D<Type>::operator+=,
           py::arg("other"),
           "Overrides the default behavior of the ``+=`` operator.",
           py::call_guard<py::gil_scoped_release>())
      .def(py::pickle(
          [](const pyinterp::Histogram2D<Type> &self) {
            return self.getstate();
          },
          [](const py::tuple &state) {
            return pyinterp::Histogram2D<Type>::setstate(state);
          }));
}

void init_histogram2d(py::module &m) {
  implement_histogram_2d<double>(m, "Float64");
  implement_histogram_2d<float>(m, "Float32");
}

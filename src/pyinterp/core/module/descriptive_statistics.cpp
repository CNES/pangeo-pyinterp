// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/descriptive_statistics.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

namespace py = pybind11;

template <typename Type>
void implement_descriptive_statistics(py::module& m,
                                      const std::string& suffix) {
  py::class_<pyinterp::DescriptiveStatistics<Type>>(
      m, ("DescriptiveStatistics" + suffix).c_str(),
      "Univariate descriptive statistics.")
      .def(py::init<py::array_t<Type, py::array::c_style>&,
                    std::optional<py::array_t<Type, py::array::c_style>>&,
                    std::optional<std::list<py::ssize_t>>&>(),
           py::arg("values"), py::arg("weights") = py::none(),
           py::arg("axis") = py::none(), R"__doc__(
Default constructor

Args:
    values (numpy.ndarray): Array containing numbers whose statistics are
        desired.
    weights (numpy.ndarray, optional): An array of weights associated with the
        values. If not provided, all values are assumed to have equal weight.
    axes (iterable, optional): Axis or axes along which to compute the
        statistics. If not provided, the statistics are computed over the
        flattened array.
)__doc__")
      .def("count", &pyinterp::DescriptiveStatistics<Type>::count,
           R"__doc__(
Returns the count of samples.

Return:
    numpy.ndarray: count of samples.
)__doc__")
      .def("kurtosis", &pyinterp::DescriptiveStatistics<Type>::kurtosis,
           R"__doc__(
Returns the kurtosis of samples.

Return:
    numpy.ndarray: kurtosis of samples.
)__doc__")
      .def("max", &pyinterp::DescriptiveStatistics<Type>::max,
           R"__doc__(
Returns maximum of samples.

Return:
    numpy.ndarray: maximum of samples.
)__doc__")
      .def("mean", &pyinterp::DescriptiveStatistics<Type>::mean,
           R"__doc__(
Returns the mean of samples.

Return:
    numpy.ndarray: mean of samples.
)__doc__")
      .def("min", &pyinterp::DescriptiveStatistics<Type>::min,
           R"__doc__(
Returns the minimum of samples.

Return:
    numpy.ndarray: minimum of samples.
)__doc__")
      .def("sum", &pyinterp::DescriptiveStatistics<Type>::sum,
           R"__doc__(
Returns the sum of samples.

Return:
    numpy.ndarray: sum of samples.
)__doc__")
      .def("sum_of_weights",
           &pyinterp::DescriptiveStatistics<Type>::sum_of_weights,
           R"__doc__(
Returns the sum of samples.

Return:
    numpy.ndarray: sum of samples.
)__doc__")
      .def("skewness", &pyinterp::DescriptiveStatistics<Type>::skewness,
           R"__doc__(
Returns the skewness of samples.

Return:
    numpy.ndarray: skewness of samples.
)__doc__")
      .def("variance", &pyinterp::DescriptiveStatistics<Type>::variance,
           py::arg("ddof") = 0,
           R"__doc__(
Returns the variance of samples.

Args:
    ddof (int, optional): Means Delta Degrees of Freedom. The divisor used in
        calculations is N - ddof, where N represents the number of elements.
        By default ddof is zero.

Return:
    numpy.ndarray: variance of samples.
)__doc__")
      .def("__iadd__", &pyinterp::DescriptiveStatistics<Type>::operator+=,
           py::arg("other"), py::call_guard<py::gil_scoped_release>())
      .def(py::pickle(
          [](const pyinterp::DescriptiveStatistics<Type>& self) {
            return self.getstate();
          },
          [](const py::tuple& state) {
            return pyinterp::DescriptiveStatistics<Type>::setstate(state);
          }));
}

void init_descriptive_statistics(py::module& m) {
  implement_descriptive_statistics<double>(m, "Float64");
  implement_descriptive_statistics<float>(m, "Float32");
}

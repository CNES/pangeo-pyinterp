// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/descriptive_statistics.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

namespace py = pybind11;

template <typename Type>
void implement_descriptive_statistics(py::module &m,
                                      const std::string &suffix) {
  py::class_<pyinterp::DescriptiveStatistics<Type>>(
      m, ("DescriptiveStatistics" + suffix).c_str(),
      ("DescriptiveStatistics" + suffix +
       "(self,"
       " values: numpy.ndarray[numpy.float64],"
       " weights: Optional[numpy.ndarray[numpy.float64]] = None,"
       " axis: Optional[List[int]] = None)"
       R"__doc__(

Univariate descriptive statistics.

Args:
    values: Array containing numbers whose statistics are
        desired.
    weights: An array of weights associated with the values. If not provided,
        all values are assumed to have equal weight.
    axes: Axis or axes along which to compute the statistics. If not provided,
        the statistics are computed over the flattened array.
)__doc__")
          .c_str())
      .def(py::init<py::array_t<Type, py::array::c_style> &,
                    std::optional<py::array_t<Type, py::array::c_style>> &,
                    std::optional<std::list<py::ssize_t>> &>(),
           py::arg("values"), py::arg("weights") = std::nullopt,
           py::arg("axis") = std::nullopt)
      .def(
          "__copy__",
          [](const pyinterp::DescriptiveStatistics<Type> &self) {
            return pyinterp::DescriptiveStatistics<Type>(self);
          },
          "Implements the shallow copy operation.",
          py::call_guard<py::gil_scoped_release>())
      .def("count", &pyinterp::DescriptiveStatistics<Type>::count,
           R"__doc__(
Returns the count of samples.

Returns:
    Count of samples.
)__doc__")
      .def("kurtosis", &pyinterp::DescriptiveStatistics<Type>::kurtosis,
           R"__doc__(
Returns the kurtosis of samples.

Returns:
    Kurtosis of samples.
)__doc__")
      .def("max", &pyinterp::DescriptiveStatistics<Type>::max,
           R"__doc__(
Returns maximum of samples.

Returns:
    Maximum of samples.
)__doc__")
      .def("mean", &pyinterp::DescriptiveStatistics<Type>::mean,
           R"__doc__(
Returns the mean of samples.

Returns:
    Mean of samples.
)__doc__")
      .def("min", &pyinterp::DescriptiveStatistics<Type>::min,
           R"__doc__(
Returns the minimum of samples.

Returns:
    Minimum of samples.
)__doc__")
      .def("sum", &pyinterp::DescriptiveStatistics<Type>::sum,
           R"__doc__(
Returns the sum of samples.

Returns:
    Sum of samples.
)__doc__")
      .def("sum_of_weights",
           &pyinterp::DescriptiveStatistics<Type>::sum_of_weights,
           R"__doc__(
Returns the sum of samples.

Returns:
    Sum of samples.
)__doc__")
      .def("skewness", &pyinterp::DescriptiveStatistics<Type>::skewness,
           R"__doc__(
Returns the skewness of samples.

Returns:
    Skewness of samples.
)__doc__")
      .def("variance", &pyinterp::DescriptiveStatistics<Type>::variance,
           py::arg("ddof") = 0,
           R"__doc__(
Returns the variance of samples.

Args:
    ddof: Means Delta Degrees of Freedom. The divisor used in calculations is
        N - ``ddof``, where N represents the number of elements. By default
        ``ddof`` is zero.

Returns:
    Variance of samples.
)__doc__")
      .def("__iadd__", &pyinterp::DescriptiveStatistics<Type>::operator+=,
           py::arg("other"),
           "Overrides the default behavior of the ``+=`` operator.",
           py::call_guard<py::gil_scoped_release>())
      .def(
          "__add__",
          [](const pyinterp::DescriptiveStatistics<Type> &self,
             const pyinterp::DescriptiveStatistics<Type> &other) {
            auto result = pyinterp::DescriptiveStatistics<Type>(self);
            result += other;
            return result;
          },
          py::arg("other"),
          "Overrides the default behavior of the ``+`` operator.",
          py::call_guard<py::gil_scoped_release>())
      .def(py::pickle(
          [](const pyinterp::DescriptiveStatistics<Type> &self) {
            return self.getstate();
          },
          [](const py::tuple &state) {
            return pyinterp::DescriptiveStatistics<Type>::setstate(state);
          }));
}

void init_descriptive_statistics(py::module &m) {
  implement_descriptive_statistics<double>(m, "Float64");
  implement_descriptive_statistics<float>(m, "Float32");
}

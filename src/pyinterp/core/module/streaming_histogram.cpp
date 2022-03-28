// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/streaming_histogram.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

namespace py = pybind11;

template <typename Type>
void implement_streaming_histogram(py::module &m, const std::string &suffix) {
  py::class_<pyinterp::StreamingHistogram<Type>>(
      m, ("StreamingHistogram" + suffix).c_str(),
      ("StreamingHistogram" + suffix +
       "(self,"
       " values: numpy.ndarray[numpy.float32],"
       " weights: Optional[numpy.ndarray[numpy.float32]] = None,"
       " axis: Optional[List[int]] = None,"
       " bin_count: Optional[int] = None)" +
       R"__doc__(

Args:
    values: Array containing numbers whose statistics are desired.
    weights: An array of weights associated with the values. If not provided,
      all values are assumed to have equal weight.
    axis: Axis or axes along which to compute the statistics. If not provided,
      the statistics are computed over the flattened array.
    bin_count: Maximum number of bins to use to calculate the histogram.
      Default is 100.
)__doc__")
          .c_str())

      .def(py::init<py::array_t<Type, py::array::c_style> &,
                    std::optional<py::array_t<Type, py::array::c_style>> &,
                    std::optional<std::list<py::ssize_t>> &,
                    std::optional<size_t> &>(),
           py::arg("values"), py::arg("weights") = std::nullopt,
           py::arg("axis") = std::nullopt, py::arg("bin_count") = std::nullopt)
      .def("bins", &pyinterp::StreamingHistogram<Type>::bins,
           R"__doc__(
Returns the histogram bins.

Returns:
    The histogram bins.
)__doc__")
      .def("count", &pyinterp::StreamingHistogram<Type>::count,
           R"__doc__(
Returns the count of samples.

Returns:
    Count of samples.
      )__doc__")
      .def("kurtosis", &pyinterp::StreamingHistogram<Type>::kurtosis,
           R"__doc__(
Returns the kurtosis of samples.

Returns:
    Kurtosis of samples.
      )__doc__")
      .def("max", &pyinterp::StreamingHistogram<Type>::max,
           R"__doc__(
Returns maximum of samples.

Returns:
    Maximum of samples.
      )__doc__")
      .def("mean", &pyinterp::StreamingHistogram<Type>::mean,
           R"__doc__(
Returns the mean of samples.

Returns:
    Mean of samples.
      )__doc__")
      .def("min", &pyinterp::StreamingHistogram<Type>::min,
           R"__doc__(
Returns the minimum of samples.

Returns:
    Minimum of samples.
      )__doc__")
      .def("quantile", &pyinterp::StreamingHistogram<Type>::quantile,
           py::arg("q") = static_cast<Type>(0.5),
           R"__doc__(
Returns the quantile of samples.

Args:
    q: Quantile to compute.
Returns:
    Quantile of samples.
      )__doc__")
      .def("resize", &pyinterp::StreamingHistogram<Type>::resize,
           "Resize the maximum number of bins.")
      .def("size", &pyinterp::StreamingHistogram<Type>::size,
           R"__doc__(
Returns the number of bins.

Returns:
    Sum of bins.
      )__doc__")
      .def("sum_of_weights",
           &pyinterp::StreamingHistogram<Type>::sum_of_weights,
           R"__doc__(
Returns the sum of samples.

Returns:
    Sum of samples.
      )__doc__")
      .def("skewness", &pyinterp::StreamingHistogram<Type>::skewness,
           R"__doc__(
Returns the skewness of samples.

Returns:
    Skewness of samples.
      )__doc__")
      .def("variance", &pyinterp::StreamingHistogram<Type>::variance,
           R"__doc__(
Returns the variance of samples.

Returns:
    Variance of samples.
      )__doc__")
      .def(
          "__copy__",
          [](const pyinterp::StreamingHistogram<Type> &self) {
            return pyinterp::StreamingHistogram<Type>(self);
          },
          "Implements the shallow copy operation.",
          py::call_guard<py::gil_scoped_release>())
      .def("__iadd__", &pyinterp::StreamingHistogram<Type>::operator+=,
           py::arg("other"),
           "Overrides the default behavior of the ``+=`` operator.",
           py::call_guard<py::gil_scoped_release>())
      .def(
          "__add__",
          [](const pyinterp::StreamingHistogram<Type> &self,
             const pyinterp::StreamingHistogram<Type> &other) {
            auto result = pyinterp::StreamingHistogram<Type>(self);
            result += other;
            return result;
          },
          py::arg("other"),
          "Overrides the default behavior of the ``+`` operator.",
          py::call_guard<py::gil_scoped_release>())
      .def(py::pickle(
          [](const pyinterp::StreamingHistogram<Type> &self) {
            return self.getstate();
          },
          [](const py::tuple &state) {
            return pyinterp::StreamingHistogram<Type>::setstate(state);
          }));
}

void init_streaming_histogram(py::module &m) {
  implement_streaming_histogram<double>(m, "Float64");
  implement_streaming_histogram<float>(m, "Float32");
}

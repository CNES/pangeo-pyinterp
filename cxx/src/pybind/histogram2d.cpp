// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/histogram2d.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <optional>
#include <string_view>

#include "pyinterp/pybind/axis.hpp"
#include "pyinterp/pybind/dtype_to_str.hpp"
#include "pyinterp/pybind/histogram2d.hpp"
#include "pyinterp/pybind/ndarray_serialization.hpp"

namespace nb = nanobind;
using nb::literals::operator""_a;

namespace pyinterp::pybind {

constexpr const char *kHistogram2DDoc = R"(
Create a 2D histogram for binning continuous values into a grid using TDigest.

Groups continuous values into bins located on a 2D grid. Each bin maintains
statistical distributions using the TDigest algorithm for efficient quantile
estimation and statistical analysis.

The TDigest algorithm provides accurate quantile estimates (especially at the
tails of the distribution) while using bounded memory. It's particularly useful
for:

- Computing percentiles and quantiles
- Identifying outliers
- Analyzing large datasets that don't fit in memory
- Merging statistics from multiple datasets

Args:
    x: Definition of the bin centers for the X axis of the grid.
    y: Definition of the bin centers for the Y axis of the grid.
    compression: TDigest compression parameter (default: 100). Higher values
        provide better accuracy at the cost of memory usage. Typical values
        range from 100 to 1000.
    dtype: Data type for internal storage, either 'float32' or 'float64'.
        Determines precision and memory usage. Defaults to 'float64'.

Examples:
    >>> import pyinterp
    >>> import numpy as np
    >>> x = pyinterp.Axis(np.arange(0, 360, 1), period=360)
    >>> y = pyinterp.Axis(np.arange(-90, 90, 1))
    >>> hist = pyinterp.Histogram2D(x, y)
    >>>
    >>> # Add some sample data
    >>> x_data = np.random.uniform(0, 360, 10000)
    >>> y_data = np.random.uniform(-90, 90, 10000)
    >>> z_data = np.random.normal(0, 1, 10000)
    >>> hist.push(x_data, y_data, z_data)
    >>>
    >>> # Compute statistics
    >>> mean_grid = hist.mean()
    >>> median_grid = hist.quantile(0.5)
    >>> p95_grid = hist.quantile(0.95)
    >>>
    >>> # Create with custom compression for better accuracy
    >>> hist_accurate = pyinterp.Histogram2D(x, y, compression=500)
    >>>
    >>> # Create with float32 for reduced memory usage
    >>> hist_compact = pyinterp.Histogram2D(x, y, dtype='float32')

Notes:
    Unlike traditional histograms that bin values into discrete counts,
    Histogram2D uses TDigest to maintain a compressed representation of the
    distribution in each bin. This allows for:

    - Accurate quantile queries without storing all data points
    - Efficient merging of histograms from different datasets
    - Bounded memory usage regardless of the number of samples

    TDigest provides better accuracy at the tails of the distribution
    (near 0th and 100th percentiles) where traditional histograms struggle.
)";

constexpr const char *kHistogram2DPushDoc = R"(
Push new samples into the histogram bins.

Values are assigned to bins based on their (x, y) coordinates. NaN values
are automatically filtered out. Each value is added to the TDigest in its
corresponding bin.

Args:
    x: X coordinates of the values to push (1D array).
    y: Y coordinates of the values to push (1D array).
    z: Values to accumulate in the bins (1D array).

Raises:
    ValueError: If arrays have different shapes or are not 1-dimensional.

Examples:
    >>> import numpy as np
    >>> import pyinterp
    >>> x_axis = pyinterp.Axis(np.arange(0, 10, 1))
    >>> y_axis = pyinterp.Axis(np.arange(0, 10, 1))
    >>> hist = pyinterp.Histogram2D(x_axis, y_axis)
    >>>
    >>> # Add single batch of data
    >>> x = np.array([1.5, 2.3, 3.7])
    >>> y = np.array([4.2, 5.1, 6.8])
    >>> z = np.array([10.0, 20.0, 30.0])
    >>> hist.push(x, y, z)
    >>>
    >>> # Add more data incrementally
    >>> x2 = np.array([1.8, 2.9])
    >>> y2 = np.array([4.5, 5.5])
    >>> z2 = np.array([15.0, 25.0])
    >>> hist.push(x2, y2, z2)
)";

constexpr const char *kHistogram2DQuantileDoc = R"(
Compute the specified quantile for each bin.

Uses the TDigest algorithm to estimate quantiles with high accuracy,
particularly at the tails of the distribution.

Args:
    q: Quantile to compute, in the range [0, 1]. For example:
        - 0.0 returns the minimum
        - 0.25 returns the 25th percentile (Q1)
        - 0.5 returns the median
        - 0.75 returns the 75th percentile (Q3)
        - 1.0 returns the maximum

Returns:
    2D array containing the quantile value for each bin. Bins with no
    data return NaN.

Raises:
    ValueError: If q is not in the range [0, 1].

Examples:
    >>> hist = pyinterp.Histogram2D(x_axis, y_axis)
    >>> hist.push(x_data, y_data, z_data)
    >>>
    >>> # Compute various percentiles
    >>> min_grid = hist.quantile(0.0)
    >>> q1_grid = hist.quantile(0.25)
    >>> median_grid = hist.quantile(0.5)
    >>> q3_grid = hist.quantile(0.75)
    >>> max_grid = hist.quantile(1.0)
    >>>
    >>> # Compute 95th percentile
    >>> p95 = hist.quantile(0.95)
)";

namespace detail {

/// @brief Helper to bind common statistical methods for Histogram2D
/// @tparam HistogramType Type of the histogram class
/// @tparam ClassType nanobind class type
template <typename HistogramType, typename ClassType>
auto bind_histogram_statistics_methods(ClassType &cls) -> void {
  cls.def("clear", &HistogramType::clear,
          "Reset all statistics and clear all bins.",
          nb::call_guard<nb::gil_scoped_release>())

      .def("count", &HistogramType::count,
           "Compute the count of points within each bin. Returns a 2D array "
           "of counts.",
           nb::call_guard<nb::gil_scoped_release>())

      .def("max", &HistogramType::max,
           "Compute the maximum value in each bin. Returns a 2D array of "
           "maximum values.",
           nb::call_guard<nb::gil_scoped_release>())

      .def("mean", &HistogramType::mean,
           "Compute the mean value in each bin. Returns a 2D array of mean "
           "values.",
           nb::call_guard<nb::gil_scoped_release>())

      .def("min", &HistogramType::min,
           "Compute the minimum value in each bin. Returns a 2D array of "
           "minimum values.",
           nb::call_guard<nb::gil_scoped_release>())

      .def("sum_of_weights", &HistogramType::sum_of_weights,
           "Compute the sum of weights in each bin. Returns a 2D array of "
           "weight sums.",
           nb::call_guard<nb::gil_scoped_release>())

      .def("quantile", &HistogramType::quantile, "q"_a, kHistogram2DQuantileDoc,
           nb::call_guard<nb::gil_scoped_release>());
}

/// @brief Helper to bind pickle support for Histogram2D
/// @tparam HistogramType Type of the histogram class
/// @tparam ClassType nanobind class type
template <typename HistogramType, typename ClassType>
auto bind_histogram_pickle_support(ClassType &cls) -> void {
  cls.def(
         "__copy__",
         [](const HistogramType &self) -> auto { return HistogramType(self); },
         "Implement the shallow copy operation.",
         nb::call_guard<nb::gil_scoped_release>())

      .def(
          "__getstate__",
          [](const HistogramType &self) -> auto {
            serialization::Writer state;
            {
              nb::gil_scoped_release release;
              state = self.pack();
            }
            return nb::make_tuple(writer_to_ndarray(std::move(state)));
          },
          "Get the state of the instance for pickling.")

      .def(
          "__setstate__",
          [](HistogramType &self, const nb::tuple &state) -> void {
            if (state.size() != 1) {
              throw std::invalid_argument("Invalid state");
            }
            auto array = nb::cast<NanobindArray1DUInt8>(state[0]);
            auto reader = reader_from_ndarray(array);
            {
              nb::gil_scoped_release release;
              new (&self)
                  HistogramType(std::move(HistogramType::unpack(reader)));
            }
          },
          "state"_a, "Set the state of the instance from pickling.");
}

}  // namespace detail

template <typename T>
auto init_histogram2d(nb::module_ &m, std::string_view suffix) -> void {
  auto class_name = "Histogram2D" + std::string(suffix);

  // Bind Histogram2D class
  auto histogram2d_cls =
      nb::class_<Histogram2D<T>>(m, class_name.c_str(), kHistogram2DDoc);

  histogram2d_cls
      .def(nb::init<Axis<double>, Axis<double>, std::optional<size_t>>(), "x"_a,
           "y"_a, "compression"_a = std::nullopt)

      .def_prop_ro("x", &Histogram2D<T>::x,
                   "Get the bin centers for the X axis of the grid.")

      .def_prop_ro("y", &Histogram2D<T>::y,
                   "Get the bin centers for the Y axis of the grid.")

      .def("push", &Histogram2D<T>::push, "x"_a, "y"_a, "z"_a,
           kHistogram2DPushDoc, nb::call_guard<nb::gil_scoped_release>())

      .def("__iadd__", &Histogram2D<T>::operator+=, "other"_a,
           "Merge another histogram into this one. Histograms must have "
           "identical grid definitions.",
           nb::call_guard<nb::gil_scoped_release>());

  // Bind statistical methods
  detail::bind_histogram_statistics_methods<Histogram2D<T>>(histogram2d_cls);

  // Bind pickle support
  detail::bind_histogram_pickle_support<Histogram2D<T>>(histogram2d_cls);
}

/// @brief Histogram2D factory function that accepts dtype parameter
/// @param[in] x Definition of the bin centers for the X axis of the grid.
/// @param[in] y Definition of the bin centers for the Y axis of the grid.
/// @param[in] compression TDigest compression parameter (default: 100).
/// @param[in] dtype Data type for internal storage, either 'float32' or
/// 'float64'. Determines precision and memory usage. Defaults to 'float64'.
/// @return nanobind::object containing the Histogram2D instance.
auto histogram_2d_factory(const Axis<double> &x, const Axis<double> &y,
                          const std::optional<size_t> &compression,
                          const nb::object &dtype) -> nb::object {
  auto dtype_str = dtype_to_str(dtype).value_or("float64");

  // Create appropriate Histogram2D based on dtype string
  if (dtype_str == "float32") {
    return nb::cast(Histogram2D<float>(x, y, compression), nb::rv_policy::move);
  }
  if (dtype_str == "float64") {
    return nb::cast(Histogram2D<double>(x, y, compression),
                    nb::rv_policy::move);
  }
  throw std::invalid_argument("dtype must be 'float32' or 'float64', got: " +
                              dtype_str);
}

constexpr const char *kHistogram2DFactoryDoc = R"(
Create a 2D histogram for binning continuous values into a grid using TDigest.

Groups continuous values into bins located on a 2D grid. Each bin maintains
statistical distributions using the TDigest algorithm for efficient quantile
estimation and statistical analysis.

Args:
    x: Definition of the bin centers for the X axis of the grid.
    y: Definition of the bin centers for the Y axis of the grid.
    compression: TDigest compression parameter (default: 100). Higher values
        provide better accuracy at the cost of memory usage. Typical values
        range from 100 to 1000.
    dtype: Data type for internal storage, either 'float32' or 'float64'.
        Determines precision and memory usage. Defaults to 'float64'.

Examples:
    >>> import pyinterp
    >>> import numpy as np
    >>> x = pyinterp.Axis(np.arange(0, 360, 1), period=360)
    >>> y = pyinterp.Axis(np.arange(-90, 90, 1))
    >>> hist = pyinterp.Histogram2D(x, y)
    >>>
    >>> # Create with custom compression for better accuracy
    >>> hist = pyinterp.Histogram2D(x, y, compression=500)
    >>>
    >>> # Create with float32 for reduced memory usage
    >>> hist = pyinterp.Histogram2D(x, y, dtype='float32')

Returns:
    Histogram2D instance with the specified dtype.
)";

auto init_histogram2d(nb::module_ &m) -> void {
  // Register the concrete Histogram2D classes
  init_histogram2d<float>(m, "Float32");
  init_histogram2d<double>(m, "Float64");

  // Register the factory function
  m.def("Histogram2D", &histogram_2d_factory, kHistogram2DFactoryDoc, "x"_a,
        "y"_a, "compression"_a = std::nullopt, "dtype"_a = nb::none());
}

}  // namespace pyinterp::pybind

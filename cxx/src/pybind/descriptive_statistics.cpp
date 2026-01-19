// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/descriptive_statistics.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <concepts>
#include <cstdint>
#include <format>
#include <optional>
#include <stdexcept>

#include "pyinterp/pybind/dtype_to_str.hpp"
#include "pyinterp/pybind/ndarray_serialization.hpp"
#include "pyinterp/tensor.hpp"

namespace nb = nanobind;

namespace pyinterp::pybind {

/// Documentation strings
constexpr const char* const kDescriptiveStatisticsDoc = R"doc(
Univariate descriptive statistics.

Computes statistics using numerically stable algorithms that support
parallel and online computation with arbitrary weights.

Reference: https://doi.org/10.1007/s00180-015-0637-z

Args:
    values: Input array of values.
    weights: Optional array of weights (same shape as values).
    axis: Optional axis or axes along which to compute statistics.

Example:
    >>> import numpy as np
    >>> import pyinterp
    >>> data = np.random.randn(100, 50)
    >>> stats = pyinterp.DescriptiveStatistics(data)
    >>> print(f"Mean: {stats.mean()}")

    Compute along axis

    >>> stats_axis = pyinterp.DescriptiveStatistics(data, axis=0)
    >>> print(f"Means shape: {stats_axis.mean().shape}")
)doc";

constexpr const char* const kVarianceDoc = R"doc(
Returns the variance.

Args:
    ddof: Delta degrees of freedom. The divisor used is N - ddof,
          where N is the number of elements. Default is 0.

Returns:
    The variance of the samples.
)doc";

template <std::floating_point T>
void DescriptiveStatistics<T>::validate_axis_bounds(const Shape& shape,
                                                    const VectorIndex& axis) {
  const auto ndim = static_cast<int64_t>(shape.size());
  for (const auto& ax : axis) {
    if (ax < 0 || ax >= ndim) {
      throw std::out_of_range(std::format(
          "Axis index {} is out of bounds for array of dimension {}", ax,
          ndim));
    }
  }
}

/// Python wrapper for DescriptiveStatistics
template <std::floating_point T>
class PyDescriptiveStatistics {
 public:
  using Core = DescriptiveStatistics<T>;
  using Shape = typename Core::Shape;

  // Default constructor
  PyDescriptiveStatistics(
      const nb::ndarray<T, nb::device::cpu>& values,
      const std::optional<nb::ndarray<T, nb::device::cpu>>& weights,
      const std::optional<std::vector<int64_t>>& axis);

  /// Construct from core instance
  explicit PyDescriptiveStatistics(std::unique_ptr<Core> core)
      : core_(std::move(core)) {}

  /// Copy constructor
  PyDescriptiveStatistics(const PyDescriptiveStatistics& other) {
    nb::gil_scoped_release release;
    core_ = std::make_unique<Core>(*other.core_);
  }
  // Returns count of samples
  [[nodiscard]] auto count() const -> nb::ndarray<nb::numpy, uint64_t> {
    Vector<uint64_t> result;
    {
      nb::gil_scoped_release release;
      result = core_->count();
    }
    return to_numpy_array(result);
  }

  /// Returns minimum as numpy array with proper shape
  [[nodiscard]] auto min() const -> nb::ndarray<nb::numpy, T> {
    Vector<T> result;
    {
      nb::gil_scoped_release release;
      result = core_->min();
    }
    return to_numpy_array(result);
  }

  /// Returns maximum as numpy array with proper shape
  [[nodiscard]] auto max() const -> nb::ndarray<nb::numpy, T> {
    Vector<T> result;
    {
      nb::gil_scoped_release release;
      result = core_->max();
    }
    return to_numpy_array(result);
  }

  /// Returns mean as numpy array with proper shape
  [[nodiscard]] auto mean() const -> nb::ndarray<nb::numpy, T> {
    Vector<T> result;
    {
      nb::gil_scoped_release release;
      result = core_->mean();
    }
    return to_numpy_array(result);
  }

  /// Returns variance as numpy array with proper shape
  [[nodiscard]] auto variance(int ddof = 0) const -> nb::ndarray<nb::numpy, T> {
    Vector<T> result;
    {
      nb::gil_scoped_release release;
      result = core_->variance(ddof);
    }
    return to_numpy_array(result);
  }

  /// Returns kurtosis as numpy array with proper shape
  [[nodiscard]] auto kurtosis() const -> nb::ndarray<nb::numpy, T> {
    Vector<T> result;
    {
      nb::gil_scoped_release release;
      result = core_->kurtosis();
    }
    return to_numpy_array(result);
  }

  /// Returns skewness as numpy array with proper shape
  [[nodiscard]] auto skewness() const -> nb::ndarray<nb::numpy, T> {
    Vector<T> result;
    {
      nb::gil_scoped_release release;
      result = core_->skewness();
    }
    return to_numpy_array(result);
  }

  /// Returns sum as numpy array with proper shape
  [[nodiscard]] auto sum() const -> nb::ndarray<nb::numpy, T> {
    Vector<T> result;
    {
      nb::gil_scoped_release release;
      result = core_->sum();
    }
    return to_numpy_array(result);
  }

  /// Returns sum of weights as numpy array with proper shape
  [[nodiscard]] auto sum_of_weights() const -> nb::ndarray<nb::numpy, T> {
    Vector<T> result;
    {
      nb::gil_scoped_release release;
      result = core_->sum_of_weights();
    }
    return to_numpy_array(result);
  }

  /// Aggregation operator
  auto operator+=(const PyDescriptiveStatistics& other)
      -> PyDescriptiveStatistics& {
    nb::gil_scoped_release release;
    *core_ += *other.core_;
    return *this;
  }

  /// Get the state for pickling
  [[nodiscard]] auto getstate() const -> nanobind::tuple;

  /// Set the state for unpickling
  [[nodiscard]] static auto setstate(const nanobind::tuple& state)
      -> PyDescriptiveStatistics<T>;

 private:
  std::unique_ptr<Core> core_{};

  /// Convert Eigen vector to numpy array with proper shape
  template <typename U>
  [[nodiscard]] auto to_numpy_array(const Vector<U>& vec) const
      -> nb::ndarray<nb::numpy, U>;
};

/// Bind DescriptiveStatistics for a specific type
template <typename T>
auto bind_descriptive_statistics(nb::module_& m, std::string_view suffix)
    -> void {
  auto class_name = std::format("DescriptiveStatistics{}", suffix);
  nb::class_<PyDescriptiveStatistics<T>>(m, class_name.c_str(),
                                         kDescriptiveStatisticsDoc)
      .def(nb::init<nb::ndarray<T, nb::device::cpu>,
                    std::optional<nb::ndarray<T, nb::device::cpu>>,
                    std::optional<std::vector<int64_t>>>(),
           nb::arg("values"), nb::arg("weights") = nb::none(),
           nb::arg("axis") = nb::none())

      .def("count", &PyDescriptiveStatistics<T>::count,
           "Return the count of samples.")

      .def("min", &PyDescriptiveStatistics<T>::min,
           "Return the minimum of samples.")

      .def("max", &PyDescriptiveStatistics<T>::max,
           "Return the maximum of samples.")

      .def("mean", &PyDescriptiveStatistics<T>::mean,
           "Return the mean of samples.")

      .def("variance", &PyDescriptiveStatistics<T>::variance,
           nb::arg("ddof") = 0, kVarianceDoc)

      .def("kurtosis", &PyDescriptiveStatistics<T>::kurtosis,
           "Return the kurtosis of samples.")

      .def("skewness", &PyDescriptiveStatistics<T>::skewness,
           "Return the skewness of samples.")

      .def("sum", &PyDescriptiveStatistics<T>::sum,
           "Return the sum of samples.")

      .def("sum_of_weights", &PyDescriptiveStatistics<T>::sum_of_weights,
           "Return the sum of weights.")

      .def("__iadd__", &PyDescriptiveStatistics<T>::operator+=,
           nb::arg("other"),
           "Override the default behavior of the `+=` operator.")

      .def("__getstate__", &PyDescriptiveStatistics<T>::getstate,
           "Get the state for pickling.")
      .def(
          "__setstate__",
          [](PyDescriptiveStatistics<T>& self, nb::tuple& state) -> void {
            new (&self) PyDescriptiveStatistics<T>(
                PyDescriptiveStatistics<T>::setstate(state));
          },
          nb::arg("state"), "Set the state for unpickling.")

      .def(
          "__copy__",
          [](const PyDescriptiveStatistics<T>& self) -> auto {
            return PyDescriptiveStatistics<T>(self);
          },
          "Create a copy of this object.");
}

/// @brief DescriptiveStatistics factory function that accepts dtype parameter
auto descriptive_statistics_factory(
    const nb::object& values_obj, const std::optional<nb::object>& weights_obj,
    const std::optional<std::vector<int64_t>>& axis, const nb::object& dtype)
    -> nb::object {
  auto dtype_str = dtype_to_str(dtype).value_or("float64");

  // Create appropriate DescriptiveStatistics based on dtype string
  if (dtype_str == "float32") {
    auto values_f32 = nb::cast<nb::ndarray<float, nb::device::cpu>>(values_obj);
    std::optional<nb::ndarray<float, nb::device::cpu>> weights_f32;
    if (weights_obj) {
      weights_f32 = nb::cast<nb::ndarray<float, nb::device::cpu>>(*weights_obj);
    }
    return nb::cast(
        PyDescriptiveStatistics<float>(values_f32, weights_f32, axis),
        nb::rv_policy::move);
  }
  if (dtype_str == "float64") {
    auto values_f64 =
        nb::cast<nb::ndarray<double, nb::device::cpu>>(values_obj);
    std::optional<nb::ndarray<double, nb::device::cpu>> weights_f64;
    if (weights_obj) {
      weights_f64 =
          nb::cast<nb::ndarray<double, nb::device::cpu>>(*weights_obj);
    }
    return nb::cast(
        PyDescriptiveStatistics<double>(values_f64, weights_f64, axis),
        nb::rv_policy::move);
  }
  throw std::invalid_argument("dtype must be 'float32' or 'float64', got: " +
                              dtype_str);
}

constexpr const char* const kDescriptiveStatisticsFactoryDoc = R"doc(
Univariate descriptive statistics.

Computes statistics using numerically stable algorithms that support
parallel and online computation with arbitrary weights.

Reference: https://doi.org/10.1007/s00180-015-0637-z

Parameters:
    values: Input array of values.
    weights: Optional array of weights (same shape as values).
    axis: Optional axis or axes along which to compute statistics.
    dtype: Data type for internal storage, either 'float32' or 'float64'.
        Determines precision and memory usage. Defaults to 'float64'.

Examples:
    >>> import numpy as np
    >>> import pyinterp

    Compute statistics for a 1D array with float64 (default)

    >>> data = np.random.randn(100)
    >>> stats = pyinterp.DescriptiveStatistics(data)
    >>> print(f"Mean: {stats.mean()}, Std: {np.sqrt(stats.variance())}")

    Compute statistics with float32 for reduced memory usage

    >>> data = data.astype('float32')
    >>> stats = pyinterp.DescriptiveStatistics(data, dtype='float32')

    Compute along a specific axis

    >>> data = np.random.randn(100, 50)
    >>> stats_axis = pyinterp.DescriptiveStatistics(data, axis=[0])
    >>> print(f"Means shape: {stats_axis.mean().shape}")

    Compute with weights

    >>> weights = np.random.rand(100, 50)
    >>> stats_weighted = pyinterp.DescriptiveStatistics(data, weights=weights)
)doc";

auto init_descriptive_statistics(nb::module_& m) -> void {
  // Register the concrete DescriptiveStatistics classes
  bind_descriptive_statistics<double>(m, "Float64");
  bind_descriptive_statistics<float>(m, "Float32");

  // Register the factory function
  m.def("DescriptiveStatistics", &descriptive_statistics_factory,
        kDescriptiveStatisticsFactoryDoc, nb::arg("values"),
        nb::arg("weights") = std::nullopt, nb::arg("axis") = nb::none(),
        nb::arg("dtype") = nb::none());
}

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
PyDescriptiveStatistics<T>::PyDescriptiveStatistics(
    const nb::ndarray<T, nb::device::cpu>& values,
    const std::optional<nb::ndarray<T, nb::device::cpu>>& weights,
    const std::optional<std::vector<int64_t>>& axis) {
  // Extract shape
  Shape shape;
  shape.reserve(values.ndim());
  for (size_t ix = 0; ix < values.ndim(); ++ix) {
    shape.push_back(values.shape(ix));
  }

  // Extract strides (in elements, not bytes)
  std::vector<int64_t> strides;
  strides.reserve(values.ndim());
  for (std::size_t i = 0; i < values.ndim(); ++i) {
    strides.push_back(values.stride(i));
  }

  // Validate weights shape and strides
  std::vector<int64_t> weight_strides;
  if (weights) {
    if (weights->ndim() != values.ndim()) {
      throw std::invalid_argument("values and weights must have same ndim");
    }
    for (size_t ix = 0; ix < values.ndim(); ++ix) {
      if (weights->shape(ix) != values.shape(ix)) {
        throw std::invalid_argument("values and weights must have same shape");
      }
    }
    weight_strides.reserve(weights->ndim());
    for (size_t ix = 0; ix < weights->ndim(); ++ix) {
      weight_strides.push_back(weights->stride(ix));
    }
  }

  /// Convert axis to optional vector of int64_t (negative axes are replaced)
  std::optional<std::vector<int64_t>> axis_vector =
      axis ? std::make_optional(
                 tensor::validate_and_convert_axis(*axis, shape.size()))
           : std::nullopt;

  // Get raw pointers
  const T* values_ptr = values.data();
  const T* weights_ptr = weights ? weights->data() : nullptr;

  {
    nb::gil_scoped_release release;
    core_ = std::make_unique<Core>(values_ptr, shape, strides, weights_ptr,
                                   weight_strides, axis_vector);
  }
}

// ============================================================================

template <std::floating_point T>
template <typename U>
auto PyDescriptiveStatistics<T>::to_numpy_array(const Vector<U>& vec) const
    -> nb::ndarray<nb::numpy, U> {
  const auto& shape = core_->shape();

  // For scalar result, return 0-d array
  if (shape.size() == 1 && shape[0] == 1) {
    auto* data = new U[1];
    data[0] = vec[0];
    nb::capsule owner(
        data, [](void* p) noexcept -> void { delete[] static_cast<U*>(p); });
    return nb::ndarray<nb::numpy, U>(data, {}, owner);
  }

  // Allocate and copy data
  auto* data = new U[vec.size()];
  std::copy_n(vec.data(), vec.size(), data);

  nb::capsule owner(
      data, [](void* p) noexcept -> void { delete[] static_cast<U*>(p); });
  return nb::ndarray<nb::numpy, U>(data, shape.size(), shape.data(), owner);
}

// ============================================================================

template <std::floating_point T>
auto PyDescriptiveStatistics<T>::getstate() const -> nanobind::tuple {
  serialization::Writer state;
  {
    nb::gil_scoped_release release;
    state = core_->pack();
  }
  return nanobind::make_tuple(writer_to_ndarray(std::move(state)));
}

// ============================================================================

template <std::floating_point T>
auto PyDescriptiveStatistics<T>::setstate(const nanobind::tuple& state)
    -> PyDescriptiveStatistics<T> {
  if (state.size() != 1) {
    throw std::invalid_argument("Invalid state");
  }
  auto array = nanobind::cast<NanobindArray1DUInt8>(state[0]);
  auto reader = reader_from_ndarray(array);
  {
    nb::gil_scoped_release release;
    auto core = DescriptiveStatistics<T>::unpack(reader);
    return PyDescriptiveStatistics<T>{
        std::make_unique<DescriptiveStatistics<T>>(std::move(core))};
  }
}

}  // namespace pyinterp::pybind

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/pybind/tdigest.hpp"

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
constexpr const char* const kTDigestDoc = R"doc(
T-Digest for incremental quantile estimation.

Computes quantiles using the t-digest algorithm which provides accurate
estimates, especially at the tails of the distribution. Supports parallel
and online computation with arbitrary weights.

Reference: Computing Extremely Accurate Quantiles Using t-Digests
https://github.com/tdunning/t-digest/blob/master/docs/t-digest-paper/histo.pdf

Args:
    values: Input array of values.
    weights: Optional array of weights (same shape as values).
    axis: Optional axis or axes along which to compute quantiles.
    compression: Compression parameter controlling accuracy vs memory tradeoff.
        Higher values provide better accuracy but use more memory.
        Typical values: 100-1000. Default is 100.

Example:
    >>> import numpy as np
    >>> import pyinterp
    >>> data = np.random.randn(10000)
    >>> tdigest = pyinterp.TDigest(data)
    >>> median = tdigest.quantile(0.5)
    >>> print(f"Median: {median}")

    Compute multiple quantiles

    >>> quantiles = tdigest.quantile(np.array([0.25, 0.5, 0.75]))
    >>> print(f"Q25, Q50, Q75: {quantiles}")

    Compute along axis

    >>> data_2d = np.random.randn(100, 50)
    >>> tdigest_axis = pyinterp.TDigest(data_2d, axis=[0])
    >>> medians = tdigest_axis.quantile(0.5)
    >>> print(f"Medians shape: {medians.shape}")
)doc";

constexpr const char* const kQuantileScalarDoc = R"doc(
Calculate quantile using t-digest algorithm.

Args:
    q: Quantile in range [0, 1]. For example, 0.5 for median, 0.95 for 95th
        percentile.

Returns:
    Estimated quantile value(s).
)doc";

constexpr const char* const kQuantileVectorDoc = R"doc(
Calculate multiple quantiles using t-digest algorithm.

Args:
    quantiles: Array of quantile values in range [0, 1].

Returns:
    Matrix of quantile estimates with shape [n_digests x n_quantiles].
    If axis reduction was not used (single digest), returns a 1D array.
)doc";

/// Python wrapper for TDigest
template <std::floating_point T>
class PyTDigest {
 public:
  using Core = TDigest<T>;
  using Shape = typename Core::Shape;

  // Default constructor
  PyTDigest(const nb::ndarray<T, nb::device::cpu>& values,
            const std::optional<nb::ndarray<T, nb::device::cpu>>& weights,
            const std::optional<std::vector<int64_t>>& axis,
            size_t compression);

  /// Construct from core instance
  explicit PyTDigest(std::unique_ptr<Core> core) : core_(std::move(core)) {}

  /// Copy constructor
  PyTDigest(const PyTDigest& other) {
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

  /// Returns sum of weights as numpy array with proper shape
  [[nodiscard]] auto sum_of_weights() const -> nb::ndarray<nb::numpy, T> {
    Vector<T> result;
    {
      nb::gil_scoped_release release;
      result = core_->sum_of_weights();
    }
    return to_numpy_array(result);
  }

  /// Calculate single quantile
  [[nodiscard]] auto quantile_scalar(const T q) const
      -> nb::ndarray<nb::numpy, T> {
    Vector<T> result;
    {
      nb::gil_scoped_release release;
      result = core_->quantile(q);
    }
    return to_numpy_array(result);
  }

  /// Calculate multiple quantiles
  [[nodiscard]] auto quantile_vector(
      const nb::ndarray<T, nb::device::cpu>& quantiles) const
      -> nb::ndarray<nb::numpy, T> {
    // Convert numpy array to Eigen vector
    Eigen::VectorX<T> q_vec(quantiles.size());
    for (size_t i = 0; i < quantiles.size(); ++i) {
      q_vec[i] = quantiles.data()[i];
    }

    Eigen::MatrixX<T> result;
    {
      nb::gil_scoped_release release;
      result = core_->quantile(q_vec);
    }
    return to_numpy_matrix(result);
  }

  /// Aggregation operator
  auto operator+=(const PyTDigest& other) -> PyTDigest& {
    nb::gil_scoped_release release;
    *core_ += *other.core_;
    return *this;
  }

  /// Get the state for pickling
  [[nodiscard]] auto getstate() const -> nanobind::tuple;

  /// Set the state for unpickling
  [[nodiscard]] static auto setstate(const nanobind::tuple& state)
      -> PyTDigest<T>;

 private:
  std::unique_ptr<Core> core_{};

  /// Convert Eigen vector to numpy array with proper shape
  template <typename U>
  [[nodiscard]] auto to_numpy_array(const Vector<U>& vec) const
      -> nb::ndarray<nb::numpy, U>;

  /// Convert Eigen matrix to numpy array with proper shape
  [[nodiscard]] auto to_numpy_matrix(const Eigen::MatrixX<T>& mat) const
      -> nb::ndarray<nb::numpy, T>;
};

/// Bind TDigest for a specific type
template <typename T>
auto bind_tdigest(nb::module_& m, std::string_view suffix) -> void {
  auto class_name = std::format("TDigest{}", suffix);
  nb::class_<PyTDigest<T>>(m, class_name.c_str(), kTDigestDoc)
      .def(nb::init<nb::ndarray<T, nb::device::cpu>,
                    std::optional<nb::ndarray<T, nb::device::cpu>>,
                    std::optional<std::vector<int64_t>>, size_t>(),
           nb::arg("values"), nb::arg("weights") = nb::none(),
           nb::arg("axis") = nb::none(), nb::arg("compression") = 100)

      .def("count", &PyTDigest<T>::count, "Return the count of samples.")

      .def("min", &PyTDigest<T>::min, "Return the minimum of samples.")

      .def("max", &PyTDigest<T>::max, "Return the maximum of samples.")

      .def("mean", &PyTDigest<T>::mean, "Return the mean of samples.")

      .def("sum_of_weights", &PyTDigest<T>::sum_of_weights,
           "Return the sum of weights.")

      .def("quantile", &PyTDigest<T>::quantile_scalar, nb::arg("q"),
           kQuantileScalarDoc)

      .def("quantile", &PyTDigest<T>::quantile_vector, nb::arg("quantiles"),
           kQuantileVectorDoc)

      .def("__iadd__", &PyTDigest<T>::operator+=, nb::arg("other"),
           "Override the default behavior of the `+=` operator.")

      .def("__getstate__", &PyTDigest<T>::getstate,
           "Get the state for pickling.")
      .def(
          "__setstate__",
          [](PyTDigest<T>& self, nb::tuple& state) -> void {
            new (&self) PyTDigest<T>(PyTDigest<T>::setstate(state));
          },
          nb::arg("state"), "Set the state for unpickling.")

      .def(
          "__copy__",
          [](const PyTDigest<T>& self) -> auto { return PyTDigest<T>(self); },
          "Create a copy of this object.");
}

/// @brief TDigest factory function that accepts dtype parameter
auto tdigest_factory(const nb::object& values_obj,
                     const std::optional<nb::object>& weights_obj,
                     const std::optional<std::vector<int64_t>>& axis,
                     size_t compression, const nb::object& dtype)
    -> nb::object {
  auto dtype_str = dtype_to_str(dtype).value_or("float64");

  // Create appropriate TDigest based on dtype string
  if (dtype_str == "float32") {
    auto values_f32 = nb::cast<nb::ndarray<float, nb::device::cpu>>(values_obj);
    std::optional<nb::ndarray<float, nb::device::cpu>> weights_f32;
    if (weights_obj) {
      weights_f32 = nb::cast<nb::ndarray<float, nb::device::cpu>>(*weights_obj);
    }
    return nb::cast(
        PyTDigest<float>(values_f32, weights_f32, axis, compression),
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
        PyTDigest<double>(values_f64, weights_f64, axis, compression),
        nb::rv_policy::move);
  }
  throw std::invalid_argument("dtype must be 'float32' or 'float64', got: " +
                              dtype_str);
}

constexpr const char* const kTDigestFactoryDoc = R"doc(
T-Digest for incremental quantile estimation.

Computes quantiles using the t-digest algorithm which provides accurate
estimates, especially at the tails of the distribution. Supports parallel
and online computation with arbitrary weights.

Reference: Computing Extremely Accurate Quantiles Using t-Digests
https://github.com/tdunning/t-digest/blob/master/docs/t-digest-paper/histo.pdf

Parameters:
    values: Input array of values.
    weights: Optional array of weights (same shape as values).
    axis: Optional axis or axes along which to compute quantiles.
    compression: Compression parameter controlling accuracy vs memory tradeoff.
        Higher values provide better accuracy but use more memory.
        Typical values: 100-1000. Default is 100.
    dtype: Data type for internal storage, either 'float32' or 'float64'.
        Determines precision and memory usage. Defaults to 'float64'.

Examples:
    >>> import numpy as np
    >>> import pyinterp

    # Compute t-digest for a 1D array with float64 (default)
    >>> data = np.random.randn(10000)
    >>> tdigest = pyinterp.TDigest(data)
    >>> median = tdigest.quantile(0.5)
    >>> print(f"Median: {median}")

    # Compute t-digest with float32 for reduced memory usage
    >>> data = data.astype('float32')
    >>> tdigest = pyinterp.TDigest(data, dtype='float32')

    # Compute along a specific axis
    >>> data_2d = np.random.randn(100, 50)
    >>> tdigest_axis = pyinterp.TDigest(data_2d, axis=[0])
    >>> medians = tdigest_axis.quantile(0.5)
    >>> print(f"Medians shape: {medians.shape}")

    # Compute with weights and higher compression for better accuracy
    >>> weights = np.random.rand(100, 50)
    >>> tdigest_weighted = pyinterp.TDigest(data_2d,
        weights=weights,
        compression=500,
    )
    >>> weighted_median = tdigest_weighted.quantile(0.5)
)doc";

auto init_tdigest(nb::module_& m) -> void {
  // Register the concrete TDigest classes
  bind_tdigest<double>(m, "Float64");
  bind_tdigest<float>(m, "Float32");

  // Register the factory function
  m.def("TDigest", &tdigest_factory, kTDigestFactoryDoc, nb::arg("values"),
        nb::arg("weights") = std::nullopt, nb::arg("axis") = nb::none(),
        nb::arg("compression") = 100, nb::arg("dtype") = nb::none());
}

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
PyTDigest<T>::PyTDigest(
    const nb::ndarray<T, nb::device::cpu>& values,
    const std::optional<nb::ndarray<T, nb::device::cpu>>& weights,
    const std::optional<std::vector<int64_t>>& axis, size_t compression) {
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
                                   weight_strides, axis_vector, compression);
  }
}

// ============================================================================

template <std::floating_point T>
template <typename U>
auto PyTDigest<T>::to_numpy_array(const Vector<U>& vec) const
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
auto PyTDigest<T>::to_numpy_matrix(const Eigen::MatrixX<T>& mat) const
    -> nb::ndarray<nb::numpy, T> {
  const auto rows = mat.rows();
  const auto cols = mat.cols();

  // Allocate and copy data (column-major to row-major conversion)
  auto* data = new T[rows * cols];
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      data[i * cols + j] = mat(i, j);
    }
  }

  nb::capsule owner(
      data, [](void* p) noexcept -> void { delete[] static_cast<T*>(p); });

  // If single digest (rows=1), return 1D array
  if (rows == 1) {
    std::array<size_t, 1> shape_1d = {static_cast<size_t>(cols)};
    return nb::ndarray<nb::numpy, T>(data, 1, shape_1d.data(), owner);
  }

  // Otherwise return 2D array
  std::array<size_t, 2> shape_2d = {static_cast<size_t>(rows),
                                    static_cast<size_t>(cols)};
  return nb::ndarray<nb::numpy, T>(data, 2, shape_2d.data(), owner);
}

// ============================================================================

template <std::floating_point T>
auto PyTDigest<T>::getstate() const -> nanobind::tuple {
  serialization::Writer state;
  {
    nb::gil_scoped_release release;
    state = core_->pack();
  }
  return nanobind::make_tuple(writer_to_ndarray(std::move(state)));
}

// ============================================================================

template <std::floating_point T>
auto PyTDigest<T>::setstate(const nanobind::tuple& state) -> PyTDigest<T> {
  if (state.size() != 1) {
    throw std::invalid_argument("Invalid state");
  }
  auto array = nanobind::cast<NanobindArray1DUInt8>(state[0]);
  auto reader = reader_from_ndarray(array);
  {
    nb::gil_scoped_release release;
    auto core = TDigest<T>::unpack(reader);
    return PyTDigest<T>{std::make_unique<TDigest<T>>(std::move(core))};
  }
}

}  // namespace pyinterp::pybind

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <nanobind/nanobind.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <vector>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/descriptive_statistics.hpp"
#include "pyinterp/serialization_buffer.hpp"
#include "pyinterp/tensor.hpp"

namespace pyinterp::pybind {

/// Univariate descriptive statistics with parallel and online computation
/// support.
///
/// Reference: Numerically stable, scalable formulas for parallel and online
/// computation of higher-order multivariate central moments with arbitrary
/// weights.
/// https://doi.org/10.1007/s00180-015-0637-z
template <std::floating_point T>
class DescriptiveStatistics {
 public:
  /// @brief Alias for the underlying accumulator type
  using Accumulators = math::DescriptiveStatistics<T>;

  /// @brief Alias of vector of dimensions
  using Shape = tensor::Shape;

  /// @brief Alias of vector of indexes
  using VectorIndex = tensor::VectorIndex;

  /// @brief Construct from strided data with optional weights and axis
  /// reduction
  ///
  /// This constructor supports non-contiguous numpy arrays (e.g., x[::2, ::5])
  /// by using strides to compute correct memory offsets.
  ///
  /// @param[in] values Pointer to values data
  /// @param[in] shape Shape of the array
  /// @param[in] strides Element strides for each dimension
  /// @param[in] weights Optional pointer to weights data (nullptr if no
  /// weights)
  /// @param[in] weight_strides Element strides for weights (empty if no
  /// weights)
  /// @param[in] axis Optional axis indices to reduce over
  DescriptiveStatistics(const T* values, const Shape& shape,
                        const VectorIndex& strides, const T* weights,
                        const VectorIndex& weight_strides,
                        const std::optional<VectorIndex>& axis);

  /// @brief Copy constructor
  DescriptiveStatistics(const DescriptiveStatistics&) = default;

  /// @brief Move constructor
  DescriptiveStatistics(DescriptiveStatistics&&) noexcept = default;

  /// @brief Copy assignment
  auto operator=(const DescriptiveStatistics&)
      -> DescriptiveStatistics& = default;

  /// @brief Move assignment
  auto operator=(DescriptiveStatistics&&) noexcept
      -> DescriptiveStatistics& = default;

  /// @brief Destructor
  ~DescriptiveStatistics() = default;

  /// @brief Returns the count of samples
  [[nodiscard]] auto count() const -> Vector<uint64_t> {
    return calculate_statistics<&Accumulators::count, uint64_t>();
  }

  /// @brief Returns the minimum of samples
  [[nodiscard]] auto min() const -> Vector<T> {
    return calculate_statistics<&Accumulators::min>();
  }

  /// @brief Returns the maximum of samples
  [[nodiscard]] auto max() const -> Vector<T> {
    return calculate_statistics<&Accumulators::max>();
  }

  /// @brief Returns the mean of samples
  [[nodiscard]] auto mean() const -> Vector<T> {
    return calculate_statistics<&Accumulators::mean>();
  }

  /// @brief Returns the variance of samples
  [[nodiscard]] auto variance(int ddof = 0) const -> Vector<T> {
    return calculate_statistics<&Accumulators::variance>(ddof);
  }

  /// @brief Returns the kurtosis of samples
  [[nodiscard]] auto kurtosis() const -> Vector<T> {
    return calculate_statistics<&Accumulators::kurtosis>();
  }

  /// @brief Returns the skewness of samples
  [[nodiscard]] auto skewness() const -> Vector<T> {
    return calculate_statistics<&Accumulators::skewness>();
  }

  /// @brief Returns the sum of samples
  [[nodiscard]] auto sum() const -> Vector<T> {
    return calculate_statistics<&Accumulators::sum>();
  }

  /// @brief Returns the sum of weights
  [[nodiscard]] auto sum_of_weights() const -> Vector<T> {
    return calculate_statistics<&Accumulators::sum_of_weights>();
  }

  /// @brief Returns the output shape
  [[nodiscard]] constexpr auto shape() const noexcept -> const Shape& {
    return shape_;
  }

  /// @brief Aggregation of statistics
  /// @param[in] other Another DescriptiveStatistics instance
  /// @return Reference to this object after aggregation
  auto operator+=(const DescriptiveStatistics& other) -> DescriptiveStatistics&;

  /// @brief Serialize the object for storage or transmission.
  /// @return Serialized state as a Writer object
  [[nodiscard]] auto pack() const -> serialization::Writer;

  /// @brief Deserialize an object from serialized state.
  /// @param[in] state Reference to serialization Reader containing encoded data
  /// @return New DescriptiveStatistics instance with restored state
  /// @throw std::invalid_argument If the state is invalid or empty
  [[nodiscard]] static auto unpack(serialization::Reader& state)
      -> DescriptiveStatistics<T>;

 private:
  /// @brief Magic number for validation
  static constexpr uint32_t kMagicNumber = 0x44535441;

  /// @brief Shape of the resulting statistics array
  Shape shape_;

  /// @brief Accumulators for each resulting element
  std::vector<Accumulators> accumulators_;

  /// @brief Default constructor
  DescriptiveStatistics() = default;

  /// @brief Get the size of the resulting statistics array
  /// @returns Total number of elements
  [[nodiscard]] constexpr auto size() const -> size_t {
    return std::ranges::fold_left(shape_, size_t{1}, std::multiplies{});
  }

  /// @brief Validate axis indices are within bounds
  /// @param shape Shape of the array
  /// @param axis Axis indices to validate
  static void validate_axis_bounds(const Shape& shape, const VectorIndex& axis);

  /// @brief Push all values from strided array (no axis reduction)
  /// @param[in] values Pointer to values data
  /// @param[in] shape Shape of the array
  /// @param[in] strides Element strides for each dimension
  /// @returns Vector of accumulators with computed statistics
  [[nodiscard]] auto push_all_strided(const T* values, const Shape& shape,
                                      const VectorIndex& strides)
      -> std::vector<Accumulators>;

  /// @brief Push all values with weights from strided arrays (no axis
  /// reduction)
  /// @param[in] values Pointer to values data
  /// @param[in] shape Shape of the array
  /// @param[in] strides Element strides for each dimension
  /// @param[in] weights Pointer to weights data
  /// @param[in] weight_strides Element strides for weights
  /// @returns Vector of accumulators with computed statistics
  [[nodiscard]] auto push_all_strided(const T* values, const Shape& shape,
                                      const VectorIndex& strides,
                                      const T* weights,
                                      const VectorIndex& weight_strides)
      -> std::vector<Accumulators>;

  /// @brief Push values with axis reduction from strided arrays
  /// @param[in] values Pointer to values data
  /// @param[in] shape Shape of the array
  /// @param[in] strides Element strides for each dimension
  /// @param[in] weights Pointer to weights data
  /// @param[in] weight_strides Element strides for weights
  /// @param[in] adjusted_strides Adjusted strides for reduced dimensions
  /// @returns Vector of accumulators with computed statistics
  [[nodiscard]] auto push_reduced_strided(const T* values, const Shape& shape,
                                          const VectorIndex& strides,
                                          const T* weights,
                                          const VectorIndex& weight_strides,
                                          const VectorIndex& adjusted_strides)
      -> std::vector<Accumulators>;

  /// @brief Calculate statistics using member function pointer
  /// @tparam MemberFunc Pointer to member function of Accumulators
  /// @tparam ResultType Result type of the member function (default: T)
  /// @tparam Args Additional arguments for the member function
  /// @param args Additional arguments for the member function
  /// @returns Vector of computed statistics
  template <auto MemberFunc, typename ResultType = T, typename... Args>
  [[nodiscard]] auto calculate_statistics(Args... args) const
      -> Vector<ResultType>;
};

/// @brief Initialize DescriptiveStatistics bindings
/// @param[in,out] m Nanobind module
auto init_descriptive_statistics(nanobind::module_& m) -> void;

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
auto DescriptiveStatistics<T>::push_all_strided(const T* values,
                                                const Shape& shape,
                                                const VectorIndex& strides)
    -> std::vector<Accumulators> {
  auto result = std::vector<Accumulators>(1);
  auto& acc = result[0];

  const auto ndim = static_cast<int64_t>(shape.size());
  const auto total =
      std::ranges::fold_left(shape, size_t{1}, std::multiplies{});

  VectorIndex indices = VectorIndex(ndim, 0);

  for (size_t ix = 0; ix < total; ++ix) {
    const auto offset = tensor::compute_offset(indices, strides);
    const auto value = values[offset];
    if (!std::isnan(value)) {
      acc(value);
    }
    tensor::increment_indices(indices, shape);
  }
  return result;
}

// ============================================================================

template <std::floating_point T>
auto DescriptiveStatistics<T>::push_all_strided(
    const T* values, const Shape& shape, const VectorIndex& strides,
    const T* weights, const VectorIndex& weight_strides)
    -> std::vector<Accumulators> {
  auto result = std::vector<Accumulators>(1);
  auto& acc = result[0];

  const auto ndim = static_cast<int64_t>(shape.size());
  const auto total =
      std::ranges::fold_left(shape, size_t{1}, std::multiplies{});

  VectorIndex indices = VectorIndex(ndim, 0);

  for (size_t ix = 0; ix < total; ++ix) {
    const auto value_offset = tensor::compute_offset(indices, strides);
    const auto value = values[value_offset];
    if (!std::isnan(value)) {
      const auto weight_offset =
          tensor::compute_offset(indices, weight_strides);
      acc(value, weights[weight_offset]);
    }
    tensor::increment_indices(indices, shape);
  }
  return result;
}

// ============================================================================

template <std::floating_point T>
auto DescriptiveStatistics<T>::push_reduced_strided(
    const T* values, const Shape& shape, const VectorIndex& strides,
    const T* weights, const VectorIndex& weight_strides,
    const VectorIndex& adjusted_strides) -> std::vector<Accumulators> {
  auto result = std::vector<Accumulators>(size());

  const auto ndim = static_cast<int64_t>(shape.size());
  const auto total =
      std::ranges::fold_left(shape, size_t{1}, std::multiplies{});

  VectorIndex indices = VectorIndex(ndim, 0);
  const T default_weight = T{1};

  for (size_t ix = 0; ix < total; ++ix) {
    const auto value_offset = tensor::compute_offset(indices, strides);
    const auto value = values[value_offset];

    if (!std::isnan(value)) {
      // Compute output index using adjusted strides
      const auto output_idx =
          std::ranges::fold_left(std::views::zip(indices, adjusted_strides),
                                 int64_t{0}, [](int64_t acc, auto pair) {
                                   auto [idx, stride] = pair;
                                   return acc + idx * stride;
                                 });

      T weight = default_weight;
      if (weights != nullptr) {
        const auto weight_offset =
            tensor::compute_offset(indices, weight_strides);
        weight = weights[weight_offset];
      }
      result[output_idx](value, weight);
    }
    tensor::increment_indices(indices, shape);
  }
  return result;
}

// ============================================================================

template <std::floating_point T>
DescriptiveStatistics<T>::DescriptiveStatistics(
    const T* values, const Shape& shape, const VectorIndex& strides,
    const T* weights, const VectorIndex& weight_strides,
    const std::optional<VectorIndex>& axis) {
  if (axis) {
    validate_axis_bounds(shape, *axis);
  }

  if (!axis) {
    // Compute statistics for the whole array
    shape_ = {1};
    accumulators_ = weights ? push_all_strided(values, shape, strides, weights,
                                               weight_strides)
                            : push_all_strided(values, shape, strides);
  } else {
    // Compute statistics on reduced dimensions
    auto properties = tensor::compute_reduced_properties(shape, strides, *axis);
    shape_ = std::move(properties.shape);
    accumulators_ =
        push_reduced_strided(values, shape, strides, weights, weight_strides,
                             properties.adjusted_strides);
  }
}

// ============================================================================

template <std::floating_point T>
template <auto MemberFunc, typename ResultType, typename... Args>
auto DescriptiveStatistics<T>::calculate_statistics(Args... args) const
    -> Vector<ResultType> {
  const auto n = size();
  Eigen::VectorX<ResultType> result(n);

  for (size_t ix = 0; ix < n; ++ix) {
    result[static_cast<int64_t>(ix)] = (accumulators_[ix].*MemberFunc)(args...);
  }
  return result;
}

// ============================================================================

template <std::floating_point T>
auto DescriptiveStatistics<T>::operator+=(const DescriptiveStatistics<T>& other)
    -> DescriptiveStatistics<T>& {
  if (shape_ != other.shape_) {
    throw std::invalid_argument(
        "Cannot aggregate DescriptiveStatistics with different shapes");
  }

  for (auto [lhs, rhs] : std::views::zip(accumulators_, other.accumulators_)) {
    lhs += rhs;
  }
  return *this;
}

// ============================================================================

template <std::floating_point T>
auto DescriptiveStatistics<T>::pack() const -> serialization::Writer {
  serialization::Writer buffer;
  buffer.write(kMagicNumber);
  buffer.write(shape_);
  buffer.write(accumulators_);
  return buffer;
}

// ============================================================================

template <std::floating_point T>
auto DescriptiveStatistics<T>::unpack(serialization::Reader& state)
    -> DescriptiveStatistics<T> {
  if (state.size() == 0) {
    throw std::invalid_argument(
        "Cannot unpack DescriptiveStatistics from "
        "empty state");
  }
  const auto magic_number = state.read<uint32_t>();
  if (magic_number != kMagicNumber) {
    throw std::invalid_argument(
        "Invalid DescriptiveStatistics state (bad magic "
        "number)");
  }
  DescriptiveStatistics<T> result;
  result.shape_ = state.read_vector<size_t>();
  result.accumulators_ = state.read_vector<Accumulators>();
  return result;
}

}  // namespace pyinterp::pybind

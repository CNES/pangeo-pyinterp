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
#include <utility>
#include <vector>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/tdigest.hpp"
#include "pyinterp/serialization_buffer.hpp"
#include "pyinterp/tensor.hpp"

namespace pyinterp::pybind {

/// T-Digest for incremental quantile estimation with parallel and online
/// computation support.
///
/// Reference: Computing Extremely Accurate Quantiles Using t-Digests
/// https://github.com/tdunning/t-digest/blob/master/docs/t-digest-paper/histo.pdf
template <std::floating_point T>
class TDigest {
 public:
  /// @brief Alias for the underlying t-digest type
  using Core = math::TDigest<T>;

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
  /// @param[in] compression Compression parameter (higher = better accuracy)
  TDigest(const T* values, const Shape& shape, const VectorIndex& strides,
          const T* weights, const VectorIndex& weight_strides,
          const std::optional<VectorIndex>& axis, size_t compression = 100);

  /// @brief Copy constructor
  TDigest(const TDigest&) = default;

  /// @brief Move constructor
  TDigest(TDigest&&) noexcept = default;

  /// @brief Copy assignment
  auto operator=(const TDigest&) -> TDigest& = default;

  /// @brief Move assignment
  auto operator=(TDigest&&) noexcept -> TDigest& = default;

  /// @brief Destructor
  ~TDigest() = default;

  /// @brief Returns the count of samples
  [[nodiscard]] auto count() const -> Vector<uint64_t> {
    return calculate_statistics<&Core::count, uint64_t>();
  }

  /// @brief Returns the minimum of samples
  [[nodiscard]] auto min() const -> Vector<T> {
    return calculate_statistics<&Core::min>();
  }

  /// @brief Returns the maximum of samples
  [[nodiscard]] auto max() const -> Vector<T> {
    return calculate_statistics<&Core::max>();
  }

  /// @brief Returns the mean of samples
  [[nodiscard]] auto mean() const -> Vector<T> {
    return calculate_statistics<&Core::mean>();
  }

  /// @brief Returns the sum of weights
  [[nodiscard]] auto sum_of_weights() const -> Vector<T> {
    return calculate_statistics<&Core::sum_of_weights>();
  }

  /// @brief Calculate quantile for each digest
  /// @param[in] q Quantile value(s) in range [0, 1]
  /// @return Quantile estimates
  [[nodiscard]] auto quantile(const T q) const -> Vector<T> {
    return calculate_statistics<&Core::quantile>(q);
  }

  /// @brief Calculate multiple quantiles for each digest
  /// @param[in] quantiles Vector of quantile values in range [0, 1]
  /// @return Matrix of quantile estimates [n_digests x n_quantiles]
  [[nodiscard]] auto quantile(
      const Eigen::Ref<const Eigen::VectorX<T>>& quantiles) const
      -> Eigen::MatrixX<T>;

  /// @brief Returns the output shape
  [[nodiscard]] constexpr auto shape() const noexcept -> const Shape& {
    return shape_;
  }

  /// @brief Aggregation of t-digests
  /// @param[in] other Another TDigest instance
  /// @return Reference to this object after aggregation
  auto operator+=(const TDigest& other) -> TDigest&;

  /// @brief Serialize the object for storage or transmission.
  /// @return Serialized state as a Writer object
  [[nodiscard]] auto pack() const -> serialization::Writer;

  /// @brief Deserialize an object from serialized state.
  /// @param[in] state Reference to serialization Reader containing encoded data
  /// @return New TDigest instance with restored state
  /// @throw std::invalid_argument If the state is invalid or empty
  [[nodiscard]] static auto unpack(serialization::Reader& state) -> TDigest<T>;

 private:
  /// @brief Magic number for validation
  static constexpr uint32_t kMagicNumber = 0x54444947;

  /// @brief Shape of the resulting t-digest array
  Shape shape_;

  /// @brief T-Digest instances for each resulting element
  std::vector<Core> digests_;

  /// @brief Default constructor
  TDigest() = default;

  /// @brief Get the size of the resulting t-digest array
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
  /// @param[in] compression Compression parameter
  /// @returns Vector of t-digests with computed statistics
  [[nodiscard]] auto push_all_strided(const T* values, const Shape& shape,
                                      const VectorIndex& strides,
                                      size_t compression) -> std::vector<Core>;

  /// @brief Push all values with weights from strided arrays (no axis
  /// reduction)
  /// @param[in] values Pointer to values data
  /// @param[in] shape Shape of the array
  /// @param[in] strides Element strides for each dimension
  /// @param[in] weights Pointer to weights data
  /// @param[in] weight_strides Element strides for weights
  /// @param[in] compression Compression parameter
  /// @returns Vector of t-digests with computed statistics
  [[nodiscard]] auto push_all_strided(const T* values, const Shape& shape,
                                      const VectorIndex& strides,
                                      const T* weights,
                                      const VectorIndex& weight_strides,
                                      size_t compression) -> std::vector<Core>;

  /// @brief Push values with axis reduction from strided arrays
  /// @param[in] values Pointer to values data
  /// @param[in] shape Shape of the array
  /// @param[in] strides Element strides for each dimension
  /// @param[in] weights Pointer to weights data
  /// @param[in] weight_strides Element strides for weights
  /// @param[in] adjusted_strides Adjusted strides for reduced dimensions
  /// @param[in] compression Compression parameter
  /// @returns Vector of t-digests with computed statistics
  [[nodiscard]] auto push_reduced_strided(const T* values, const Shape& shape,
                                          const VectorIndex& strides,
                                          const T* weights,
                                          const VectorIndex& weight_strides,
                                          const VectorIndex& adjusted_strides,
                                          size_t compression)
      -> std::vector<Core>;

  /// @brief Calculate statistics using member function pointer
  /// @tparam MemberFunc Pointer to member function of Core
  /// @tparam ResultType Result type of the member function (default: T)
  /// @tparam Args Additional arguments for the member function
  /// @param args Additional arguments for the member function
  /// @returns Vector of computed statistics
  template <auto MemberFunc, typename ResultType = T, typename... Args>
  [[nodiscard]] auto calculate_statistics(Args... args) const
      -> Vector<ResultType>;
};

/// @brief Initialize TDigest bindings
/// @param[in,out] m Nanobind module
auto init_tdigest(nanobind::module_& m) -> void;

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
auto TDigest<T>::push_all_strided(const T* values, const Shape& shape,
                                  const VectorIndex& strides,
                                  size_t compression) -> std::vector<Core> {
  auto result = std::vector<Core>{Core{compression}};
  auto& digest = result[0];

  const auto ndim = static_cast<int64_t>(shape.size());
  const auto total =
      std::ranges::fold_left(shape, size_t{1}, std::multiplies{});

  VectorIndex indices = VectorIndex(ndim, 0);

  for (size_t ix = 0; ix < total; ++ix) {
    const auto offset = tensor::compute_offset(indices, strides);
    const auto value = values[offset];
    if (!std::isnan(value)) {
      digest.add(value);
    }
    tensor::increment_indices(indices, shape);
  }
  return result;
}

// ============================================================================

template <std::floating_point T>
auto TDigest<T>::push_all_strided(const T* values, const Shape& shape,
                                  const VectorIndex& strides, const T* weights,
                                  const VectorIndex& weight_strides,
                                  size_t compression) -> std::vector<Core> {
  auto result = std::vector<Core>{Core{compression}};
  auto& digest = result[0];

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
      digest.add(value, weights[weight_offset]);
    }
    tensor::increment_indices(indices, shape);
  }
  return result;
}

// ============================================================================

template <std::floating_point T>
auto TDigest<T>::push_reduced_strided(const T* values, const Shape& shape,
                                      const VectorIndex& strides,
                                      const T* weights,
                                      const VectorIndex& weight_strides,
                                      const VectorIndex& adjusted_strides,
                                      size_t compression) -> std::vector<Core> {
  auto result = std::vector<Core>(size(), Core{compression});

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
      result[output_idx].add(value, weight);
    }
    tensor::increment_indices(indices, shape);
  }
  return result;
}

// ============================================================================

template <std::floating_point T>
TDigest<T>::TDigest(const T* values, const Shape& shape,
                    const VectorIndex& strides, const T* weights,
                    const VectorIndex& weight_strides,
                    const std::optional<VectorIndex>& axis,
                    size_t compression) {
  if (axis) {
    validate_axis_bounds(shape, *axis);
  }

  if (!axis) {
    // Compute t-digest for the whole array
    shape_ = {1};
    digests_ = weights ? push_all_strided(values, shape, strides, weights,
                                          weight_strides, compression)
                       : push_all_strided(values, shape, strides, compression);
  } else {
    // Compute t-digests on reduced dimensions
    auto properties = tensor::compute_reduced_properties(shape, strides, *axis);
    shape_ = std::move(properties.shape);
    digests_ =
        push_reduced_strided(values, shape, strides, weights, weight_strides,
                             properties.adjusted_strides, compression);
  }
}

// ============================================================================

template <std::floating_point T>
template <auto MemberFunc, typename ResultType, typename... Args>
auto TDigest<T>::calculate_statistics(Args... args) const
    -> Vector<ResultType> {
  const auto n = size();
  Eigen::VectorX<ResultType> result(n);

  for (size_t ix = 0; ix < n; ++ix) {
    result[static_cast<int64_t>(ix)] = (digests_[ix].*MemberFunc)(args...);
  }
  return result;
}

// ============================================================================

template <std::floating_point T>
auto TDigest<T>::quantile(const Eigen::Ref<const Eigen::VectorX<T>>& quantiles)
    const -> Eigen::MatrixX<T> {
  const auto n_digests = size();
  const auto n_quantiles = quantiles.size();
  Eigen::MatrixX<T> result(n_digests, n_quantiles);

  for (int64_t i = 0; std::cmp_less(i, n_digests); ++i) {
    for (int64_t j = 0; std::cmp_less(j, n_quantiles); ++j) {
      result(i, j) = digests_[static_cast<size_t>(i)].quantile(quantiles[j]);
    }
  }
  return result;
}

// ============================================================================

template <std::floating_point T>
auto TDigest<T>::operator+=(const TDigest<T>& other) -> TDigest<T>& {
  if (shape_ != other.shape_) {
    throw std::invalid_argument(
        "Cannot aggregate TDigest with different shapes");
  }

  for (auto [lhs, rhs] : std::views::zip(digests_, other.digests_)) {
    lhs += rhs;
  }
  return *this;
}

// ============================================================================

template <std::floating_point T>
auto TDigest<T>::pack() const -> serialization::Writer {
  serialization::Writer buffer;
  buffer.write(kMagicNumber);
  buffer.write(shape_);
  // Write number of digests
  buffer.write(static_cast<size_t>(digests_.size()));
  // Write each digest individually
  for (const auto& digest : digests_) {
    auto digest_writer = digest.pack();
    // Convert Writer to vector of bytes and write it
    std::vector<std::byte> digest_bytes(
        digest_writer.data(), digest_writer.data() + digest_writer.size());
    buffer.write(digest_bytes);
  }
  return buffer;
}

// ============================================================================

template <std::floating_point T>
auto TDigest<T>::unpack(serialization::Reader& state) -> TDigest<T> {
  if (state.size() == 0) {
    throw std::invalid_argument("Cannot unpack TDigest from empty state");
  }
  const auto magic_number = state.read<uint32_t>();
  if (magic_number != kMagicNumber) {
    throw std::invalid_argument("Invalid TDigest state (bad magic number)");
  }
  TDigest<T> result;
  result.shape_ = state.read_vector<size_t>();
  // Read number of digests
  const auto n_digests = state.read<size_t>();
  result.digests_.reserve(n_digests);
  // Read each digest individually
  for (size_t i = 0; i < n_digests; ++i) {
    auto digest_bytes = state.read_vector<std::byte>();
    auto digest_reader = serialization::Reader(std::move(digest_bytes));
    result.digests_.push_back(Core::unpack(digest_reader));
  }
  return result;
}

// ============================================================================

template <std::floating_point T>
void TDigest<T>::validate_axis_bounds(const Shape& shape,
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

}  // namespace pyinterp::pybind

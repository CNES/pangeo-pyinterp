// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::math {

/// @brief Centroid in the t-digest (stores mean and weight)
/// @tparam T The data type
template <std::floating_point T>
struct Centroid {
  T mean;    ///< Mean value of the cluster
  T weight;  ///< Total weight (count) of values in this cluster
};

/// @brief T-Digest implementation for efficient quantile estimation
/// Uses adaptive clustering to maintain accuracy at the tails of the
/// distribution
/// @tparam T The data type
template <std::floating_point T>
class TDigest {
 public:
  /// @brief Default constructor
  TDigest() { centroids_.reserve(compression_); }

  /// @brief Constructor with compression parameter
  /// @param[in] compression Controls accuracy vs memory tradeoff (typical:
  /// 100-1000). Higher compression = more centroids = better accuracy = more
  /// memory. Must be greater than 0.
  /// @throw std::invalid_argument If compression is 0
  explicit TDigest(const size_t compression) : compression_{compression} {
    if (compression == 0) [[unlikely]] {
      throw std::invalid_argument(
          "Compression parameter must be greater than 0");
    }
    centroids_.reserve(compression_);
  }

  /// @brief Serialize the state of the t-digest for storage or transmission
  /// @return The serialized state
  [[nodiscard]] auto pack() const -> serialization::Writer;

  /// @brief Deserialize a t-digest from serialized state
  /// @param[in] state Reference to serialization Reader containing encoded
  /// t-digest data
  /// @return New TDigest instance with restored properties
  /// @throw std::invalid_argument If the state is invalid or empty
  [[nodiscard]] static auto unpack(serialization::Reader& state) -> TDigest<T>;

  /// @brief Set the compression parameter
  /// @param[in] compression The new compression value (must be greater than 0)
  /// @throw std::invalid_argument If compression is 0
  auto set_compression(const size_t compression) -> void {
    if (compression == 0) [[unlikely]] {
      throw std::invalid_argument(
          "Compression parameter must be greater than 0");
    }
    compression_ = compression;
    compress();
  }

  /// @brief Get the centroids
  /// @return The list of centroids
  [[nodiscard]] constexpr auto centroids() const noexcept
      -> const std::vector<Centroid<T>>& {
    return centroids_;
  }

  /// @brief Clear the t-digest
  auto clear() noexcept -> void {
    centroids_.clear();
    unmerged_.clear();
    count_ = 0;
    min_ = std::numeric_limits<T>::max();
    max_ = std::numeric_limits<T>::lowest();
  }

  /// @brief Add a single value with weight
  /// @param[in] value The value to add
  /// @param[in] weight The weight of the value (default 1)
  auto add(const T value, const T weight = T{1}) -> void {
    if (weight <= T{0}) [[unlikely]] {
      return;
    }

    count_ += 1;
    min_ = std::min(min_, value);
    max_ = std::max(max_, value);

    // Add to unmerged buffer
    unmerged_.emplace_back(value, weight);

    // Compress if buffer is full - use larger buffer to reduce merge frequency
    // Merging is expensive, so we batch more items before merging
    if (unmerged_.size() >= compression_) {
      merge_unmerged();
    }
  }

  /// @brief Add a value (operator() for compatibility)
  /// @param[in] value The value to add
  /// @param[in] weight The weight of the value (default 1)
  auto operator()(const T value, const T weight = T{1}) -> void {
    add(value, weight);
  }

  /// @brief Merge another t-digest into this one
  /// @param[in] other The other t-digest to merge
  auto operator+=(const TDigest<T>& other) -> void {
    if (other.count_ == 0) [[unlikely]] {
      return;
    }

    count_ += other.count_;
    min_ = std::min(min_, other.min_);
    max_ = std::max(max_, other.max_);

    // Add all centroids from other to unmerged
    for (const auto& centroid : other.centroids_) {
      unmerged_.push_back(centroid);
    }

    merge_unmerged();
  }

  /// @brief Merge two t-digests
  /// @param[in] other The other t-digest to merge
  /// @return The merged t-digest
  [[nodiscard]] auto operator+(const TDigest<T>& other) const -> TDigest<T> {
    auto result = *this;
    result += other;
    return result;
  }

  /// @brief Get total count of values
  /// @return The total count
  [[nodiscard]] constexpr auto count() const noexcept -> uint64_t {
    return count_;
  }

  /// @brief Get total weight (same as count for uniform weights)
  /// @return The total weight
  [[nodiscard]] auto sum_of_weights() const noexcept -> T {
    if (!unmerged_.empty()) {
      merge_unmerged();
    }

    T total{0};
    for (const auto& c : centroids_) {
      total += c.weight;
    }
    return total;
  }

  /// @brief Get number of centroids
  /// @return The number of centroids
  [[nodiscard]] auto size() const noexcept -> size_t {
    if (!unmerged_.empty()) {
      merge_unmerged();
    }
    return centroids_.size();
  }

  /// @brief Calculate quantile using t-digest algorithm
  /// @param[in] q Quantile in range [0, 1]
  /// @return Estimated quantile value
  [[nodiscard]] auto quantile(const T q) const -> T;

  /// @brief Calculate mean
  /// @return The mean value
  [[nodiscard]] auto mean() const -> T;

  /// @brief Get minimum value
  /// @return The minimum value
  [[nodiscard]] constexpr auto min() const noexcept -> T {
    return count_ != 0 ? min_ : std::numeric_limits<T>::quiet_NaN();
  }

  /// @brief Get maximum value
  /// @return The maximum value
  [[nodiscard]] constexpr auto max() const noexcept -> T {
    return count_ != 0 ? max_ : std::numeric_limits<T>::quiet_NaN();
  }

 private:
  /// Magic number for t-digest serialization
  static constexpr uint32_t kMagicNumber = 0x54444947;
  /// Compression parameter
  size_t compression_{100};
  /// Total count of values added
  uint64_t count_{0};
  /// Minimum value added
  T min_{std::numeric_limits<T>::max()};
  /// Maximum value added
  T max_{std::numeric_limits<T>::lowest()};
  /// List of centroids (mutable to allow lazy merging in const methods)
  mutable std::vector<Centroid<T>> centroids_{};
  /// Buffer of unmerged centroids (mutable to allow lazy merging in const
  /// methods)
  mutable std::vector<Centroid<T>> unmerged_{};

  /// Scale function k(q, δ) - controls cluster size distribution
  /// Uses asin scaling for better accuracy at tails
  [[nodiscard]] auto k_scale(const T q) const noexcept -> T {
    const T delta =
        static_cast<T>(compression_) / (T{2} * std::numbers::pi_v<T>);
    return delta * std::asin(T{2} * q - T{1});
  }

  /// Inverse scale function
  [[nodiscard]] auto k_scale_inv(const T k) const noexcept -> T {
    const T delta =
        static_cast<T>(compression_) / (T{2} * std::numbers::pi_v<T>);
    return (std::sin(k / delta) + T{1}) / T{2};
  }

  /// Compute maximum cluster weight at quantile q
  [[nodiscard]] auto q_weight(const T q) const noexcept -> T {
    const T min_q = std::max(q - k_scale(q) / static_cast<T>(count_), T{0});
    const T max_q = std::min(q + k_scale(q) / static_cast<T>(count_), T{1});
    return static_cast<T>(count_) * (max_q - min_q);
  }

  /// Merge unmerged centroids into main list
  auto merge_unmerged() const -> void;

  /// Force compression to target size
  auto compress() -> void;
};

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
auto TDigest<T>::quantile(const T q) const -> T {
  if (centroids_.empty() && unmerged_.empty()) [[unlikely]] {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (q < T{0} || q > T{1}) [[unlikely]] {
    throw std::invalid_argument("Quantile must be in the range [0, 1]");
  }

  // Ensure we're working with merged data
  if (!unmerged_.empty()) {
    merge_unmerged();
  }

  if (centroids_.empty()) [[unlikely]] {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // Handle boundary cases
  if (q == T{0} || centroids_.size() == 1) {
    return min_;
  }
  if (q == T{1}) {
    return max_;
  }

  const T index = q * static_cast<T>(count_);
  T weight_sum{0};

  // Find the centroid containing the quantile
  for (size_t i = 0; i < centroids_.size(); ++i) {
    const T half_weight = centroids_[i].weight / T{2};

    if (weight_sum + half_weight >= index) {
      // Interpolate within/before this centroid
      if (i == 0) {
        return std::fma((index - weight_sum) / half_weight,
                        centroids_[i].mean - min_, min_);
      }
      const auto left_mean = centroids_[i - 1].mean;
      const auto left_weight = weight_sum - centroids_[i - 1].weight / T{2};
      const auto right_weight = weight_sum + half_weight;

      return std::fma((index - left_weight) / (right_weight - left_weight),
                      centroids_[i].mean - left_mean, left_mean);
    }

    weight_sum += centroids_[i].weight;

    if (i == centroids_.size() - 1) {
      // Past last centroid - interpolate to max
      return std::fma((index - weight_sum) / half_weight,
                      max_ - centroids_[i].mean, centroids_[i].mean);
    }
  }

  return centroids_.back().mean;
}

// ////////////////////////////////////////////////////////////////////////////

template <std::floating_point T>
auto TDigest<T>::mean() const -> T {
  if (centroids_.empty() && unmerged_.empty()) [[unlikely]] {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (!unmerged_.empty()) {
    merge_unmerged();
  }

  T sum{0};
  T total_weight{0};

  for (const auto& c : centroids_) {
    sum = std::fma(c.weight, c.mean, sum);
    total_weight += c.weight;
  }

  return sum / total_weight;
}

// ////////////////////////////////////////////////////////////////////////////

template <std::floating_point T>
auto TDigest<T>::merge_unmerged() const -> void {
  if (unmerged_.empty()) {
    return;
  }

  // In-place merge to avoid temporary allocations
  const auto old_size = centroids_.size();
  centroids_.reserve(old_size + unmerged_.size());
  centroids_.insert(centroids_.end(), unmerged_.begin(), unmerged_.end());

  // Sort only new data, then merge with existing sorted data
  auto middle = centroids_.begin() + old_size;
  std::ranges::sort(middle, centroids_.end(),
                    [](const Centroid<T>& a, const Centroid<T>& b) {
                      return a.mean < b.mean;
                    });
  std::inplace_merge(centroids_.begin(), middle, centroids_.end(),
                     [](const Centroid<T>& a, const Centroid<T>& b) {
                       return a.mean < b.mean;
                     });

  // In-place compaction using two-pointer approach
  if (centroids_.empty()) {
    unmerged_.clear();
    return;
  }

  const T inv_count = T{1} / static_cast<T>(count_);
  auto write_it = centroids_.begin();
  T weight_so_far{0};

  for (auto read_it = centroids_.begin(); read_it != centroids_.end();
       ++read_it) {
    const T q = (weight_so_far + read_it->weight / T{2}) * inv_count;
    const T max_weight = q_weight(q);

    if (write_it != read_it && read_it != centroids_.begin() &&
        write_it->weight + read_it->weight <= max_weight) {
      // Merge read_it into write_it
      const T old_weight = write_it->weight;
      const T new_weight = old_weight + read_it->weight;
      write_it->mean = std::fma(read_it->weight, read_it->mean,
                                old_weight * write_it->mean) /
                       new_weight;
      write_it->weight = new_weight;
    } else {
      // Start new centroid
      if (read_it != centroids_.begin()) {
        weight_so_far += write_it->weight;
        ++write_it;
      }
      if (write_it != read_it) {
        *write_it = *read_it;
      }
    }
  }

  // Finalize by keeping only compacted centroids
  centroids_.erase(std::next(write_it), centroids_.end());
  unmerged_.clear();
}

// ////////////////////////////////////////////////////////////////////////////

template <std::floating_point T>
auto TDigest<T>::compress() -> void {
  merge_unmerged();

  while (centroids_.size() > compression_) {
    // Simple compression: merge adjacent centroids
    T min_diff = std::numeric_limits<T>::max();
    size_t min_idx = 0;

    for (size_t i = 0; i < centroids_.size() - 1; ++i) {
      const T diff = centroids_[i + 1].mean - centroids_[i].mean;
      if (diff < min_diff) {
        min_diff = diff;
        min_idx = i;
      }
    }

    // Merge centroids at min_idx and min_idx + 1
    auto& c1 = centroids_[min_idx];
    const auto& c2 = centroids_[min_idx + 1];
    const T new_weight = c1.weight + c2.weight;
    c1.mean = std::fma(c2.weight, c2.mean, c1.weight * c1.mean) / new_weight;
    c1.weight = new_weight;
    centroids_.erase(centroids_.begin() + min_idx + 1);
  }
}

// ////////////////////////////////////////////////////////////////////////////

template <std::floating_point T>
auto TDigest<T>::pack() const -> serialization::Writer {
  // Ensure all data is merged before serialization
  if (!unmerged_.empty()) {
    merge_unmerged();
  }

  serialization::Writer writer;
  // Write version for future compatibility
  writer.write(kMagicNumber);
  writer.write(count_);
  writer.write(min_);
  writer.write(max_);
  writer.write(compression_);
  writer.write(centroids_);
  return writer;
}

// ///////////////////////////////////////////////////////////////////////////

template <std::floating_point T>
auto TDigest<T>::unpack(serialization::Reader& state) -> TDigest<T> {
  if (state.size() == 0) {
    throw std::invalid_argument("Cannot unpack TDigest from empty state");
  }

  TDigest<T> digest;
  // Read and validate magic number
  const auto magic = state.read<uint32_t>();
  if (magic != kMagicNumber) {
    throw std::invalid_argument("Invalid TDigest serialization magic number");
  }

  digest.count_ = state.read<uint64_t>();
  digest.min_ = state.read<T>();
  digest.max_ = state.read<T>();
  digest.compression_ = state.read<size_t>();
  digest.centroids_ = state.read_vector<Centroid<T>>();
  return digest;
}

}  // namespace pyinterp::math

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace pyinterp::math {

/// @brief Statistical accumulators for incremental computation
/// @tparam T Numeric type for values
template <std::floating_point T>
struct Accumulators {
  /// @brief Type used for internal computations to reduce numerical errors
  using WorkT = std::conditional_t<std::is_same_v<T, float>, double, T>;
  uint64_t count{0};                                ///< Number of samples
  WorkT sum_of_weights{0};                          ///< Sum of weights
  WorkT mean{0};                                    ///< Mean value
  WorkT min{std::numeric_limits<WorkT>::max()};     ///< Minimum value
  WorkT max{std::numeric_limits<WorkT>::lowest()};  ///< Maximum value
  WorkT sum{0};                                     ///< Sum of values
  WorkT mom2{0};                                    ///< Second central moment
  WorkT mom3{0};                                    ///< Third central moment
  WorkT mom4{0};                                    ///< Fourth central moment
};

/// @brief Univariate descriptive statistics.
///
/// Implements numerically stable incremental computation for univariate
/// descriptive statistics. Reference: "Numerically stable, scalable formulas
/// for parallel and online computation of higher-order multivariate central
/// moments with arbitrary weights" (https://doi.org/10.1007/s00180-015-0637-z)
/// @tparam T Numeric type for values
template <std::floating_point T>
class DescriptiveStatistics {
 public:
  /// @brief Default constructor
  constexpr DescriptiveStatistics() = default;

  /// @brief Create a new object from statistical incremental values
  /// @param[in] acc Statistical incremental values
  explicit constexpr DescriptiveStatistics(Accumulators<T> acc) noexcept
      : acc_{std::move(acc)} {}

  /// @brief Return the raw statistical incremental values
  [[nodiscard]] constexpr auto accumulators() const noexcept
      -> const Accumulators<T>& {
    return acc_;
  }

  /// @brief Reset the accumulator
  constexpr auto clear() noexcept -> void { acc_ = Accumulators<T>{}; }

  /// @brief Push a new value into the accumulator (weight = 1)
  ///
  /// @param[in] value The value to add
  constexpr auto operator()(const T value) noexcept -> void;

  /// @brief Push a new value into the accumulator with an associated weight
  ///
  /// @param[in] value The value to add
  /// @param[in] weight The weight associated with the value
  constexpr auto operator()(const T value, const T weight) noexcept -> void {
    using WorkT = typename Accumulators<T>::WorkT;
    if (acc_.sum_of_weights == WorkT{0}) [[unlikely]] {
      *this = DescriptiveStatistics{value, weight};
    } else {
      *this += DescriptiveStatistics{value, weight};
    }
  }

  /// @brief Return the number of samples pushed into the accumulator.
  [[nodiscard]] constexpr auto count() const noexcept -> uint64_t {
    return acc_.count;
  }

  /// @brief Return the sum of weights pushed into the accumulator.
  [[nodiscard]] constexpr auto sum_of_weights() const noexcept -> T {
    return static_cast<T>(acc_.sum_of_weights);
  }

  /// @brief Return the sum of the values pushed into the accumulator.
  [[nodiscard]] constexpr auto sum() const noexcept -> T {
    return static_cast<T>(acc_.sum);
  }

  /// @brief Return the mean of the samples
  [[nodiscard]] constexpr auto mean() const noexcept -> T {
    return acc_.count == 0 ? std::numeric_limits<T>::quiet_NaN()
                           : static_cast<T>(acc_.mean);
  }

  /// @brief Return the minimum of the samples
  [[nodiscard]] constexpr auto min() const noexcept -> T {
    return acc_.count == 0 ? std::numeric_limits<T>::quiet_NaN()
                           : static_cast<T>(acc_.min);
  }

  /// @brief Return the maximum of the samples
  [[nodiscard]] constexpr auto max() const noexcept -> T {
    return acc_.count == 0 ? std::numeric_limits<T>::quiet_NaN()
                           : static_cast<T>(acc_.max);
  }

  /// @brief Return the variance of the samples
  /// @param[in] ddof Delta degrees of freedom (default 0 for population
  /// variance, 1 for sample)
  /// @returns Variance of the samples
  [[nodiscard]] constexpr auto variance(const int ddof = 0) const noexcept
      -> T {
    using WorkT = typename Accumulators<T>::WorkT;
    const auto cardinal = acc_.sum_of_weights - static_cast<WorkT>(ddof);
    return cardinal <= WorkT{0} ? std::numeric_limits<T>::quiet_NaN()
                                : static_cast<T>(acc_.mom2 / cardinal);
  }

  /// @brief Return the standard deviation of the samples
  /// @param[in] ddof Delta degrees of freedom (default 0 for population, 1 for
  /// sample)
  /// @returns Standard deviation of the samples
  [[nodiscard]] constexpr auto std(const int ddof = 0) const noexcept -> T {
    return std::sqrt(variance(ddof));
  }

  /// @brief Return the skewness of the samples (measure of asymmetry)
  [[nodiscard]] constexpr auto skewness() const noexcept -> T {
    using WorkT = typename Accumulators<T>::WorkT;
    if (acc_.mom2 == WorkT{0}) [[unlikely]] {
      return std::numeric_limits<T>::quiet_NaN();
    }
    return static_cast<T>(std::sqrt(acc_.sum_of_weights) * acc_.mom3 /
                          std::pow(acc_.mom2, WorkT{1.5}));
  }

  /// @brief Return the excess kurtosis of the samples (measure of tail
  /// heaviness)
  [[nodiscard]] constexpr auto kurtosis() const noexcept -> T {
    using WorkT = typename Accumulators<T>::WorkT;
    if (acc_.mom2 == WorkT{0}) [[unlikely]] {
      return std::numeric_limits<T>::quiet_NaN();
    }
    return static_cast<T>(
        acc_.sum_of_weights * acc_.mom4 / (acc_.mom2 * acc_.mom2) - kThree);
  }

  /// @brief Combine two accumulators.
  ///
  /// @param[in] rhs The right-hand side accumulator
  /// @return The updated accumulator
  constexpr auto operator+=(const DescriptiveStatistics& rhs) noexcept
      -> DescriptiveStatistics&;

  /// @brief Combine two accumulators.
  ///
  /// @param[in] rhs The right-hand side accumulator
  /// @return The combined accumulator
  [[nodiscard]] constexpr auto operator+(const DescriptiveStatistics& rhs)
      const noexcept -> DescriptiveStatistics {
    auto result = *this;
    result += rhs;
    return result;
  }

 private:
  static constexpr auto kSix = typename Accumulators<T>::WorkT{6};
  static constexpr auto kFour = typename Accumulators<T>::WorkT{4};
  static constexpr auto kThree = typename Accumulators<T>::WorkT{3};
  static constexpr auto kTwo = typename Accumulators<T>::WorkT{2};
  static constexpr auto kOne = typename Accumulators<T>::WorkT{1};

  /// @brief Statistical accumulators
  Accumulators<T> acc_{};

  /// @brief Private constructor for single weighted value
  ///
  /// @param[in] value The value to add
  /// @param[in] weight The weight associated with the value
  constexpr DescriptiveStatistics(const T value, const T weight) noexcept {
    using WorkT = typename Accumulators<T>::WorkT;
    const auto working_value = static_cast<WorkT>(value);
    const auto working_weight = static_cast<WorkT>(weight);
    const auto weighted_value = working_value * working_weight;
    acc_ = Accumulators<T>{.count = 1,
                           .sum_of_weights = working_weight,
                           .mean = working_value,
                           .min = working_value,
                           .max = working_value,
                           .sum = weighted_value,
                           .mom2 = WorkT{0},
                           .mom3 = WorkT{0},
                           .mom4 = WorkT{0}};
  }
};

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
constexpr auto DescriptiveStatistics<T>::operator()(const T value) noexcept
    -> void {
  using WorkT = typename Accumulators<T>::WorkT;

  if (acc_.sum_of_weights == 0) [[unlikely]] {
    *this = DescriptiveStatistics{value, T{1}};
    return;
  }
  auto converted_value = static_cast<WorkT>(value);
  const auto r = acc_.sum_of_weights;

  acc_.sum_of_weights += 1;
  acc_.count += 1;
  acc_.sum += converted_value;

  const auto inv_n = kOne / acc_.sum_of_weights;
  const auto delta = converted_value - acc_.mean;
  const auto A = delta * inv_n;

  acc_.mean += A;
  acc_.mom4 +=
      A * (A * A * delta * r *
               (acc_.sum_of_weights * (acc_.sum_of_weights - kThree) + kThree) +
           kSix * A * acc_.mom2 - kFour * acc_.mom3);

  const auto B = value - acc_.mean;

  acc_.mom3 +=
      A * (B * delta * (acc_.sum_of_weights - kTwo) - kThree * acc_.mom2);
  acc_.mom2 = std::fma(delta, B, acc_.mom2);

  if (value < acc_.min) {
    acc_.min = value;
  } else if (value > acc_.max) {
    acc_.max = value;
  }
}

// ============================================================================

template <std::floating_point T>
constexpr auto DescriptiveStatistics<T>::operator+=(
    const DescriptiveStatistics& rhs) noexcept -> DescriptiveStatistics& {
  auto w = acc_.sum_of_weights + rhs.acc_.sum_of_weights;

  // Update min/max
  acc_.min = std::min(acc_.min, rhs.acc_.min);
  acc_.max = std::max(acc_.max, rhs.acc_.max);

  const auto delta = rhs.acc_.mean - acc_.mean;
  const auto delta_w = delta / w;
  const auto delta2_w2 = delta_w * delta_w;

  const auto w2 = acc_.sum_of_weights * acc_.sum_of_weights;
  const auto ww = acc_.sum_of_weights * rhs.acc_.sum_of_weights;
  const auto rhs_w2 = rhs.acc_.sum_of_weights * rhs.acc_.sum_of_weights;

  // Update fourth moment
  acc_.mom4 += rhs.acc_.mom4 +
               ww * (w2 - ww + rhs_w2) * delta * delta_w * delta2_w2 +
               kSix * (w2 * rhs.acc_.mom2 + rhs_w2 * acc_.mom2) * delta2_w2 +
               kFour *
                   (acc_.sum_of_weights * rhs.acc_.mom3 -
                    rhs.acc_.sum_of_weights * acc_.mom3) *
                   delta_w;

  // Update third moment
  acc_.mom3 +=
      rhs.acc_.mom3 +
      ww * (acc_.sum_of_weights - rhs.acc_.sum_of_weights) * delta * delta2_w2 +
      kThree *
          (acc_.sum_of_weights * rhs.acc_.mom2 -
           rhs.acc_.sum_of_weights * acc_.mom2) *
          delta_w;

  // Update second moment
  acc_.mom2 = std::fma(ww * delta, delta_w, acc_.mom2 + rhs.acc_.mom2);

  // Update mean
  acc_.mean = std::fma(rhs.acc_.sum_of_weights, delta_w, acc_.mean);

  // Update sum
  acc_.sum_of_weights = w;

  // Update count
  acc_.count += rhs.acc_.count;

  // Update sum of values
  acc_.sum += rhs.acc_.sum;

  return *this;
}

}  // namespace pyinterp::math

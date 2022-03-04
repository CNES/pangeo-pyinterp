// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

namespace pyinterp::detail::math {

/// Handled accumulators
template <typename T>
struct Accumulators {
  uint64_t count;
  T sum_of_weights;
  T mean;
  T min;
  T max;
  T sum;
  T mom2;
  T mom3;
  T mom4;
};

/// Univariate descriptive statistics
/// Reference: Numerically stable, scalable formulas for parallel and online
/// computation of higher-order multivariate central moments with arbitrary
/// weights
/// https://doi.org/10.1007/s00180-015-0637-z
template <typename T>
class DescriptiveStatistics {
 public:
  /// Default constructor
  DescriptiveStatistics() { clear(); }

  /// Create of a new object from statistical incremental values
  explicit DescriptiveStatistics(Accumulators<T> acc) : acc_(std::move(acc)) {}

  /// Returns the raw statistical incremental values
  explicit operator const Accumulators<T> &() const { return acc_; }

  /// Reset the accumulator
  constexpr auto clear() noexcept -> void {
    std::memset(&acc_, 0, sizeof(Accumulators<T>));
  }

  /// Push a new value into the accumulator
  constexpr auto operator()(const T &value) noexcept -> void {
    const auto r = acc_.sum_of_weights;

    if (r == 0) {
      *this = std::move(DescriptiveStatistics(value, 1));
    } else {
      acc_.sum_of_weights += 1;
      acc_.count += 1;
      acc_.sum += value;

      const auto inv_n = 1 / acc_.sum_of_weights;
      const auto delta = value - acc_.mean;
      const auto A = delta * inv_n;

      acc_.mean += A;
      acc_.mom4 +=
          A * (A * A * delta * r *
                   (acc_.sum_of_weights * (acc_.sum_of_weights - 3.) + 3.) +
               6. * A * acc_.mom2 - 4. * acc_.mom3);

      const auto B = value - acc_.mean;

      acc_.mom3 +=
          A * (B * delta * (acc_.sum_of_weights - 2.) - 3. * acc_.mom2);
      acc_.mom2 += delta * B;

      if (value < acc_.min) {
        acc_.min = value;
      } else if (value > acc_.max) {
        acc_.max = value;
      }
    }
  }

  /// push a new value into the accumulator associated with a weight
  constexpr auto operator()(const T &value, const T &weight) noexcept -> void {
    if (acc_.sum_of_weights == 0) {
      *this = std::move(DescriptiveStatistics(value, weight));
    } else {
      this->operator+=(DescriptiveStatistics(value, weight));
    }
  }

  /// Returns the number of samples pushed into the accumulator.
  [[nodiscard]] constexpr auto count() const noexcept -> uint64_t {
    return acc_.count;
  }

  /// Returns the sum of weights pushed into the accumulator.
  [[nodiscard]] constexpr auto sum_of_weights() const noexcept -> const T & {
    return acc_.sum_of_weights;
  }

  /// Returns the sum of the values pushed into the accumulator.
  [[nodiscard]] constexpr auto sum() const noexcept -> const T & {
    return acc_.sum;
  }

  /// Returns the mean of the samples
  [[nodiscard]] constexpr auto mean() const noexcept -> T {
    return acc_.count == 0 ? std::numeric_limits<T>::quiet_NaN() : acc_.mean;
  }

  /// Returns the min of the samples
  [[nodiscard]] constexpr auto min() const noexcept -> T {
    return acc_.count == 0 ? std::numeric_limits<T>::quiet_NaN() : acc_.min;
  }

  /// Returns the max of the samples
  [[nodiscard]] constexpr auto max() const noexcept -> T {
    return acc_.count == 0 ? std::numeric_limits<T>::quiet_NaN() : acc_.max;
  }

  /// Returns the variance of the samples
  [[nodiscard]] constexpr auto variance(const int ddof = 0) const noexcept
      -> T {
    const auto cardinal = acc_.sum_of_weights - ddof;
    return cardinal <= 0 ? std::numeric_limits<T>::quiet_NaN()
                         : acc_.mom2 / cardinal;
  }

  /// Returns the standard deviation of the samples
  [[nodiscard]] inline auto std(const int ddof = 0) const noexcept -> T {
    return std::sqrt(variance(ddof));
  }

  /// Returns the skewness of the samples
  [[nodiscard]] inline auto skewness() const noexcept -> T {
    return acc_.mom2 == 0 ? std::numeric_limits<T>::quiet_NaN()
                          : std::sqrt(acc_.sum_of_weights) * acc_.mom3 /
                                std::pow(acc_.mom2, T(1.5));
  }

  /// Returns the kurtosis of the samples
  [[nodiscard]] constexpr auto kurtosis() const noexcept -> T {
    return acc_.mom2 == 0
               ? std::numeric_limits<T>::quiet_NaN()
               : acc_.sum_of_weights * acc_.mom4 / (acc_.mom2 * acc_.mom2) -
                     T(3);
  }

  /// Combines two accumulators.
  constexpr auto operator+=(const DescriptiveStatistics &rhs) noexcept
      -> DescriptiveStatistics & {
    auto w = acc_.sum_of_weights + rhs.acc_.sum_of_weights;

    if (rhs.acc_.min < acc_.min) {
      acc_.min = rhs.acc_.min;
    }

    if (rhs.acc_.max > acc_.max) {
      acc_.max = rhs.acc_.max;
    }

    const auto delta = rhs.acc_.mean - acc_.mean;
    const auto delta_w = delta / w;
    const auto delta2_w2 = delta_w * delta_w;

    const auto w2 = acc_.sum_of_weights * acc_.sum_of_weights;
    const auto ww = acc_.sum_of_weights * rhs.acc_.sum_of_weights;
    const auto rhs_w2 = rhs.acc_.sum_of_weights * rhs.acc_.sum_of_weights;

    acc_.mom4 += rhs.acc_.mom4 +
                 ww * (w2 - ww + rhs_w2) * delta * delta_w * delta2_w2 +
                 6. * (w2 * rhs.acc_.mom2 + rhs_w2 * acc_.mom2) * delta2_w2 +
                 4. *
                     (acc_.sum_of_weights * rhs.acc_.mom3 -
                      rhs.acc_.sum_of_weights * acc_.mom3) *
                     delta_w;

    acc_.mom3 += rhs.acc_.mom3 +
                 ww * (acc_.sum_of_weights - rhs.acc_.sum_of_weights) * delta *
                     delta2_w2 +
                 3. *
                     (acc_.sum_of_weights * rhs.acc_.mom2 -
                      rhs.acc_.sum_of_weights * acc_.mom2) *
                     delta_w;

    acc_.mom2 += rhs.acc_.mom2 + ww * delta * delta_w;

    acc_.mean += rhs.acc_.sum_of_weights * delta_w;

    acc_.sum_of_weights = w;

    acc_.count += rhs.acc_.count;

    acc_.sum += rhs.acc_.sum;

    return *this;
  }

 private:
  Accumulators<T> acc_{};

  DescriptiveStatistics(const T &value, const T &weight) {
    auto weighted_value = weight * value;
    acc_ = std::move(Accumulators<T>{1, weight, value, weighted_value,
                                     weighted_value, weighted_value, 0, 0, 0});
  }
};

}  // namespace pyinterp::detail::math

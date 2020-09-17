// Copyright (c) 2020 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <cmath>
#include <cstring>
#include <memory>

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
  DescriptiveStatistics() { std::memset(&acc_, 0, sizeof(Accumulators<T>)); };

  /// Create of a new object from statistical incremental values
  explicit DescriptiveStatistics(Accumulators<T> acc) : acc_(std::move(acc)) {}

  /// Returns the raw statistical incremental values
  explicit operator const Accumulators<T> &() const { return acc_; }

  /// Push a new value into the accumulator
  void operator()(const T& value) {
    auto r = acc_.sum_of_weights;

    if (r == 0) {
      *this = std::move(DescriptiveStatistics(value, 1));
    } else {
      acc_.sum_of_weights += 1;
      acc_.count += 1;
      acc_.sum += value;

      auto inv_n = 1 / acc_.sum_of_weights;

      auto delta = value - acc_.mean;

      auto A = delta * inv_n;
      acc_.mean += A;
      acc_.mom4 +=
          A * (A * A * delta * r *
                   (acc_.sum_of_weights * (acc_.sum_of_weights - 3.) + 3.) +
               6. * A * acc_.mom2 - 4. * acc_.mom3);

      auto B = value - acc_.mean;
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
  void operator()(const T& value, const T& weight) {
    if (acc_.sum_of_weights == 0) {
      *this = std::move(DescriptiveStatistics(value, weight));
    } else {
      this->operator+=(DescriptiveStatistics(value, weight));
    }
  }

  /// Returns the number of samples pushed into the accumulator.
  [[nodiscard]] inline auto count() const -> uint64_t { return acc_.count; }

  /// Returns the sum of weights pushed into the accumulator.
  [[nodiscard]] inline auto sum_of_weights() const -> const T& {
    return acc_.sum_of_weights;
  }

  /// Returns the sum of the values pushed into the accumulator.
  [[nodiscard]] inline auto sum() const -> const T& { return acc_.sum; }

  /// Returns the mean of the samples
  [[nodiscard]] inline auto mean() const -> T {
    return acc_.count == 0 ? std::numeric_limits<T>::quiet_NaN() : acc_.mean;
  }

  /// Returns the min of the samples
  [[nodiscard]] inline auto min() const -> T {
    return acc_.count == 0 ? std::numeric_limits<T>::quiet_NaN() : acc_.min;
  }

  /// Returns the max of the samples
  [[nodiscard]] inline auto max() const -> T {
    return acc_.count == 0 ? std::numeric_limits<T>::quiet_NaN() : acc_.max;
  }

  /// Returns the variance of the samples
  [[nodiscard]] inline auto unbiased_variance() const -> T {
    auto cardinal = acc_.sum_of_weights - 1;
    return cardinal > 0 ? std::numeric_limits<T>::quiet_NaN()
                        : acc_.mom2 / cardinal;
  }

  /// Returns the variance of the samples
  [[nodiscard]] inline auto variance() const -> T {
    return acc_.sum_of_weights == 0 ? std::numeric_limits<T>::quiet_NaN()
                                    : acc_.mom2 / acc_.sum_of_weights;
  }

  /// Returns the standard deviation of the samples
  [[nodiscard]] inline auto std() const -> T { return std::sqrt(variance()); }

  /// Returns the skewness of the samples
  [[nodiscard]] inline auto skewness() const -> T {
    return acc_.mom2 == 0 ? std::numeric_limits<T>::quiet_NaN()
                          : std::sqrt(acc_.sum_of_weights) * acc_.mom3 /
                                std::pow(acc_.mom2, 1.5);
  }

  /// Returns the kurtosis of the samples
  [[nodiscard]] inline auto kurtosis() const -> T {
    return acc_.mom2 == 0
               ? std::numeric_limits<T>::quiet_NaN()
               : acc_.sum_of_weights * acc_.mom4 / (acc_.mom2 * acc_.mom2) -
                     3.0;
  }

  /// Combines two accumulators.
  auto operator+=(const DescriptiveStatistics& rhs) -> DescriptiveStatistics& {
    auto w = acc_.sum_of_weights + rhs.acc_.sum_of_weights;

    if (rhs.acc_.min < acc_.min) {
      acc_.min = rhs.acc_.min;
    }

    if (rhs.acc_.max > acc_.max) {
      acc_.max = rhs.acc_.max;
    }

    auto delta = rhs.acc_.mean - acc_.mean;
    auto delta_w = delta / w;
    auto delta2_w2 = delta_w * delta_w;

    auto w2 = acc_.sum_of_weights * acc_.sum_of_weights;
    auto ww = acc_.sum_of_weights * rhs.acc_.sum_of_weights;
    auto rhs_w2 = rhs.acc_.sum_of_weights * rhs.acc_.sum_of_weights;

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

  DescriptiveStatistics(const T& value, const T& weight)
      : acc_{1, weight, value, value, value, weight * value, 0, 0, 0} {}
};

}  // namespace pyinterp::detail::math

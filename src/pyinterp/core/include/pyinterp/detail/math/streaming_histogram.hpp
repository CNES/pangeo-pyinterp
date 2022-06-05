// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pyinterp/detail/isviewstream.hpp"

namespace pyinterp::detail::math {

/// Reference:
/// Yael Ben-Haim and Elad Tom-Tov,
/// A Streaming Parallel Decision Tree Algorithm,
/// Journal of Machine Learning Research, 11, 28, 849-872
/// http://jmlr.org/papers/v11/ben-haim10a.html

/// Handle a bin (pair between value/weight)
template <typename T>
struct Bin {
  T value;
  T weight;
};

/// Handle the calculation of differences between bins
template <typename T>
class BinDifferences {
 public:
  /// Default constructor
  BinDifferences(std::vector<Bin<T>> &bins, const bool weighted_diff)
      : calculate_(weighted_diff ? &BinDifferences::weighted
                                 : &BinDifferences::simple) {
    auto first = bins.begin() + 1;
    auto last = bins.end();
    while (first != last) {
      auto diff = calculate(*first, *(first - 1));
      if (diff < diff_) {
        diff_ = diff;
        index_ = std::distance(bins.begin(), first) - 1;
      }
      ++first;
    }
  }

  /// Get the minimum difference between bins.
  constexpr auto diff() const noexcept -> T { return diff_; }

  /// Get the index of the bin with the minimum difference.
  [[nodiscard]] constexpr auto index() const noexcept -> size_t {
    return index_;
  }

 private:
  std::function<T(const Bin<T> &, const Bin<T> &)> calculate_;
  size_t index_{std::numeric_limits<size_t>::max()};
  T diff_{std::numeric_limits<T>::max()};

  /// Simple difference calculation.
  constexpr static auto simple(const Bin<T> &lhs, const Bin<T> &rhs) -> T {
    return lhs.value - rhs.value;
  }

  /// Weighted difference calculation.
  constexpr static auto weighted(const Bin<T> &lhs, const Bin<T> &rhs) -> T {
    return BinDifferences::simple(lhs, rhs) *
           std::log(1e-5 + std::min(lhs.weight, rhs.weight));
  }

  /// Get the difference between the provided bins.
  constexpr auto calculate(const Bin<T> &lhs, const Bin<T> &rhs) const -> T {
    return calculate_(lhs, rhs);
  }
};

/// Streaming Histogram implementation
template <typename T>
class StreamingHistogram {
 public:
  /// Default constructor
  StreamingHistogram() { bins_.reserve(bin_count_); }

  /// Sets the properties of the histogram.
  StreamingHistogram(const size_t bin_count, const bool weighted_diff)
      : weighted_diff_(weighted_diff), bin_count_(bin_count) {
    bins_.reserve(bin_count_);
  }

  /// Create of a new object from serialized data.
  explicit StreamingHistogram(const std::string_view &state) {
    auto ss = isviewstream(state);
    ss.exceptions(std::stringstream::failbit);
    auto size = static_cast<size_t>(0);

    try {
      ss.read(reinterpret_cast<char *>(&weighted_diff_), sizeof(bool));
      ss.read(reinterpret_cast<char *>(&bin_count_), sizeof(size_t));
      ss.read(reinterpret_cast<char *>(&count_), sizeof(uint64_t));
      ss.read(reinterpret_cast<char *>(&min_), sizeof(T));
      ss.read(reinterpret_cast<char *>(&max_), sizeof(T));
      ss.read(reinterpret_cast<char *>(&size), sizeof(size_t));
      bins_.resize(size);
      ss.read(reinterpret_cast<char *>(bins_.data()), size * sizeof(Bin<T>));
    } catch (const std::ios_base::failure &e) {
      throw std::invalid_argument("invalid state");
    }
  }

  /// Serialize the state of the histogram.
  explicit operator std::string() const {
    auto ss = std::stringstream();
    ss.exceptions(std::stringstream::failbit);
    ss.write(reinterpret_cast<const char *>(&weighted_diff_), sizeof(bool));
    ss.write(reinterpret_cast<const char *>(&bin_count_), sizeof(size_t));
    ss.write(reinterpret_cast<const char *>(&count_), sizeof(uint64_t));
    ss.write(reinterpret_cast<const char *>(&min_), sizeof(T));
    ss.write(reinterpret_cast<const char *>(&max_), sizeof(T));
    auto size = bins_.size();
    ss.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
    ss.write(reinterpret_cast<const char *>(bins_.data()),
             size * sizeof(Bin<T>));
    return ss.str();
  }

  /// Set the maximum number of bins in the histogram.
  inline auto resize(const size_t bin_count) -> void {
    bin_count_ = bin_count;
    trim();
  }

  /// Get the bins that compose the histogram.
  inline auto bins() const noexcept -> const std::vector<Bin<T>> & {
    return bins_;
  }

  /// Clears the histogram.
  inline auto clear() noexcept -> void {
    *this = std::move(StreamingHistogram(bin_count_, weighted_diff_));
  }

  /// Push a new value into the histogram.
  inline auto operator()(const T &value, const T &weight = T(1)) -> void {
    ++count_;
    auto weighted_value = weight * value;
    if (weighted_value < min_) {
      min_ = weighted_value;
    }
    if (weighted_value > max_) {
      max_ = weighted_value;
    }
    update_bins(value, weight);
    trim();
  }

  /// Merges the provided histogram into the current one.
  inline auto operator+=(const StreamingHistogram<T> &other) -> void {
    count_ += other.count_;
    if (other.min_ < min_) {
      min_ = other.min_;
    }
    if (other.max_ > max_) {
      max_ = other.max_;
    }
    for (const auto &item : other.bins_) {
      update_bins(item.value, item.weight);
    }
    trim();
  }

  /// Returns the number of samples pushed into the histogram.
  [[nodiscard]] constexpr auto count() const -> uint64_t { return count_; }

  /// Returns the sum of weights pushed into the histogram.
  [[nodiscard]] constexpr auto sum_of_weights() const -> T {
    return std::accumulate(
        bins_.begin(), bins_.end(), T(0),
        [](T a, const Bin<T> &b) -> T { return a + b.weight; });
  }

  /// Returns the number of bins in the histogram.
  [[nodiscard]] constexpr auto size() const noexcept -> size_t {
    return bins_.size();
  }

  /// Calculate the quantile of the distribution
  [[nodiscard]] auto quantile(const T &quantile) const -> T {
    if (bins_.empty()) {
      return std::numeric_limits<T>::quiet_NaN();
    }

    if (quantile < 0.0 || quantile > 1.0) {
      throw std::invalid_argument("Quantile must be in the range [0, 1]");
    }

    auto weights = sum_of_weights();
    auto qw = weights * quantile;

    if (qw <= (bins_.front().weight * 0.5)) {  // left values
      auto ratio = qw / (bins_.front().weight * 0.5);
      return min_ + (ratio * (bins_.front().value - min_));
    }

    if (qw >= (weights - (bins_.back().weight * 0.5))) {  // right values
      auto base = qw - (weights - (bins_.back().weight * 0.5));
      auto ratio = base / (bins_.back().weight * 0.5);
      return bins_.back().value + (ratio * (max_ - bins_.back().value));
    }

    auto ix = static_cast<size_t>(0);
    auto mb = qw - bins_.front().weight * 0.5;
    while (mb - (bins_[ix].weight + bins_[ix + 1].weight) * 0.5 > 0 &&
           ix < bins_.size() - 1) {
      mb -= (bins_[ix].weight + bins_[ix + 1].weight) * 0.5;
      ix += 1;
    }
    auto ratio = mb / ((bins_[ix].weight + bins_[ix + 1].weight) * 0.5);
    return bins_[ix].value + (ratio * (bins_[ix + 1].value - bins_[ix].value));
  }

  /// Calculates the mean of the distribution.
  [[nodiscard]] auto mean() const -> T { return moment(0, 1); }

  /// Calculates the variance of the distribution.
  [[nodiscard]] auto variance() const -> T { return moment(mean(), 2); }

  /// Calculates the skewness of the distribution.
  [[nodiscard]] auto skewness() const -> T {
    const auto average = mean();
    return moment(average, 3) / std::pow(moment(average, 2), 1.5);
  }

  /// Calculates the kurtosis of the distribution.
  [[nodiscard]] auto kurtosis() const -> T {
    const auto average = mean();
    return moment(average, 4) / std::pow(moment(average, 2), 2) - 3.0;
  }

  /// Returns the minimum of the distribution.
  [[nodiscard]] constexpr auto min() const noexcept -> T {
    return count_ != 0U ? min_ : std::numeric_limits<T>::quiet_NaN();
  }

  /// Returns the maximum of the distribution.
  [[nodiscard]] constexpr auto max() const noexcept -> T {
    return count_ != 0U ? max_ : std::numeric_limits<T>::quiet_NaN();
  }

 private:
  bool weighted_diff_{false};
  size_t bin_count_{100};
  uint64_t count_{0};
  T min_{std::numeric_limits<T>::max()};
  T max_{std::numeric_limits<T>::min()};
  std::vector<Bin<T>> bins_{};

  /// Update the histogram with the provided value.
  auto update_bins(const T &value, const T &weight) -> void {
    if (bins_.empty()) {
      bins_.emplace_back(Bin<T>{value, weight});
      return;
    }

    if (value <= bins_.front().value) {
      bins_.insert(bins_.begin(), {value, weight});
      return;
    }

    if (value >= bins_.back().value) {
      // If the new value is greater than the last value inserted in the
      // histogram, the bins is extended.
      bins_.emplace_back(Bin<T>{value, weight});
      return;
    }

    // Binary search for the insertion index.
    auto it = std::lower_bound(
        bins_.begin(), bins_.end(), value,
        [](const Bin<T> &lhs, const T &rhs) { return lhs.value <= rhs; });
    auto index = std::distance(bins_.begin(), it);

    // If the value is already in the histogram, the weight is updated.
    auto &bin = bins_[index];
    if (bin.value == value) {
      bin.weight += weight;
      return;
    }
    bins_.insert(bins_.begin() + index, {value, weight});
  }

  /// Compress the histogram if necessary.
  auto trim() -> void {
    while (bins_.size() > bin_count_) {
      // Find a point qi that minimizes qi+1 - qi
      auto ix = BinDifferences<T>(bins_, weighted_diff_).index();

      // Replace the bins (qi, ki), (qi+1, ki+1)
      auto &item0 = bins_[ix];
      const auto &item1 = bins_[++ix];

      // by the bin ((qi * ki + qi+1 * ki+1) / (ki + ki+1), ki + ki+1)
      item0 = {(item0.value * item0.weight + item1.value * item1.weight) /
                   (item0.weight + item1.weight),
               item0.weight + item1.weight};
      bins_.erase(bins_.begin() + ix);
    }
  }

  /// Calculates a moment of order n
  constexpr auto moment(const T &c, const T &n) const -> T {
    auto sum_m = T(0);
    auto sum_weights = T(0);
    for (const auto &item : bins_) {
      sum_m += item.weight * std::pow((item.value - c), n);
      sum_weights += item.weight;
    }
    return sum_m / sum_weights;
  }
};

}  // namespace pyinterp::detail::math

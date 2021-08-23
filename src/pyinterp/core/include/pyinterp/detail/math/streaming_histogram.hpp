// Copyright (c) 2021 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <cmath>
#include <functional>
#include <numeric>
#include <optional>
#include <sstream>
#include <vector>

#include "pyinterp/detail/isviewstream.hpp"

namespace pyinterp::detail::math {

/// Reference:
/// Yael Ben-Haim and Elad Tom-Tov,
/// A Streaming Parallel Decision Tree Algorithm,
/// Journal of Machine Learning Research, 11, 28, 849-872
/// http://jmlr.org/papers/v11/ben-haim10a.html

/// Handle a bin (pair between value/count)
template <typename T>
struct Bin {
  T value;
  T count;
};

/// Handle the calculation of differences between bins
template <typename T>
class BinDifferences {
 public:
  /// Default constructor
  BinDifferences(std::vector<Bin<T>>& bins, const bool weighted_diff)
      : calculate_(weighted_diff ? &BinDifferences::weighted
                                 : &BinDifferences::simple) {
    for (size_t index = 1; index < bins.size(); ++index) {
      auto diff = calculate(bins[index], bins[index - 1]);
      if (diff < diff_) {
        diff_ = diff;
        index_ = index - 1;
      }
    }
  }

  /// Get the minimum difference between bins.
  inline auto diff() const noexcept -> T { return diff_; }

  /// Get the index of the bin with the minimum difference.
  [[nodiscard]] inline auto index() const noexcept -> size_t { return index_; }

 private:
  std::function<T(const Bin<T>&, const Bin<T>&)> calculate_;
  size_t index_{std::numeric_limits<size_t>::max()};
  T diff_{std::numeric_limits<T>::max()};

  /// Simple difference calculation.
  static inline auto simple(const Bin<T>& lhs, const Bin<T>& rhs) -> T {
    return lhs.value - rhs.value;
  }

  /// Weighted difference calculation.
  static inline auto weighted(const Bin<T>& lhs, const Bin<T>& rhs) -> T {
    return BinDifferences::simple(lhs, rhs) *
           std::log(1e-5 + std::min(lhs.count, rhs.count));
  }

  /// Get the difference between the provided bins.
  inline auto calculate(const Bin<T>& lhs, const Bin<T>& rhs) const -> T {
    return calculate_(lhs, rhs);
  }
};

/// Streaming Histogram implementation
template <typename T>
class StreamingHistogram {
 public:
  /// Default constructor
  StreamingHistogram() = default;

  /// Sets the properties of the histogram.
  StreamingHistogram(const size_t bin_count, const bool weighted_diff)
      : weighted_diff_(weighted_diff), bin_count_(bin_count) {}

  /// Create of a new object from serialized data.
  explicit StreamingHistogram(const std::string_view& state) {
    auto ss = isviewstream(state);
    ss.exceptions(std::stringstream::failbit);
    auto size = size_t(0);

    try {
      ss.read(reinterpret_cast<char*>(&weighted_diff_), sizeof(bool));
      ss.read(reinterpret_cast<char*>(&bin_count_), sizeof(size_t));
      ss.read(reinterpret_cast<char*>(&min_), sizeof(T));
      ss.read(reinterpret_cast<char*>(&max_), sizeof(T));
      ss.read(reinterpret_cast<char*>(&size), sizeof(size_t));
      bins_.resize(size);
      ss.read(reinterpret_cast<char*>(bins_.data()), size * sizeof(Bin<T>));
    } catch (const std::ios_base::failure& e) {
      throw std::invalid_argument("invalid state");
    }
  }

  /// Serialize the state of the histogram.
  explicit operator std::string() const {
    auto ss = std::stringstream();
    ss.exceptions(std::stringstream::failbit);
    ss.write(reinterpret_cast<const char*>(&weighted_diff_), sizeof(bool));
    ss.write(reinterpret_cast<const char*>(&bin_count_), sizeof(size_t));
    ss.write(reinterpret_cast<const char*>(&min_), sizeof(T));
    ss.write(reinterpret_cast<const char*>(&max_), sizeof(T));
    auto size = bins_.size();
    ss.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    ss.write(reinterpret_cast<const char*>(bins_.data()),
             size * sizeof(Bin<T>));
    return ss.str();
  }

  /// Get the bins
  inline auto bins() const noexcept -> const std::vector<Bin<T>>& {
    return bins_;
  }

  /// Clears the histogram.
  inline auto clear() noexcept -> void {
    *this = std::move(StreamingHistogram(bin_count_, weighted_diff_));
  }

  /// Push a new value into the histogram (update procedure in the paper).
  inline auto push(const T& value, const T& count = T(1)) -> void {
    update(value, count);
    trim();
  }

  /// Merges the provided histogram into the current one.
  inline auto merge(const StreamingHistogram<T>& other) -> void {
    for (const auto& item : other.bins_) {
      update(item.value, item.count);
    }
    trim();
  }

  /// Counts samples stored into the histogram
  inline auto count() const -> T {
    return std::accumulate(
        bins_.begin(), bins_.end(), T(0),
        [](T a, const Bin<T>& b) -> T { return a + b.count; });
  }

  /// Returns the number of bins in the histogram.
  inline auto size() const noexcept -> size_t { return bins_.size(); }

  /// Calculate the quantile of the distribution
  auto quantile(const T& quantile) const -> T {
    if (bins_.empty()) {
      throw std::invalid_argument("quantile from empty histogram");
    }

    if (quantile < 0.0 || quantile > 1.0) {
      throw std::invalid_argument("Quantile must be in the range [0, 1]");
    }

    auto total_count = count();
    auto quantile_count = total_count * quantile;

    if (quantile_count <= (bins_.front().count * 0.5)) {  // left values
      auto ratio = quantile_count / (bins_.front().count * 0.5);
      return min_ + (ratio * (bins_.front().value - min_));
    }

    if (quantile_count >=
        (total_count - (bins_.back().count * 0.5))) {  // right values
      auto base = quantile_count - (total_count - (bins_.back().count * 0.5));
      auto ratio = base / (bins_.back().count * 0.5);
      return bins_.back().value + (ratio * (max_ - bins_.back().value));
    }

    auto ix = size_t(0);
    auto mb = quantile_count - bins_.front().count * 0.5;
    while (mb - (bins_[ix].count + bins_[ix + 1].count) * 0.5 > 0 &&
           ix < bins_.size() - 1) {
      mb -= (bins_[ix].count + bins_[ix + 1].count) * 0.5;
      ix += 1;
    }
    auto ratio = mb / ((bins_[ix].count + bins_[ix + 1].count) * 0.5);
    return bins_[ix].value + (ratio * (bins_[ix + 1].value - bins_[ix].value));
  }

  /// Calculates the mean
  auto mean() const -> T { return moment(0, 1); }

  /// Calculates the variance
  auto variance() const -> T { return moment(mean(), 2); }

  /// Return the minimum
  auto min() const noexcept -> T { return min_; }

  /// Return the maximum
  auto max() const noexcept -> T { return max_; }

 private:
  bool weighted_diff_{false};
  size_t bin_count_{100};
  T min_{std::numeric_limits<T>::max()};
  T max_{std::numeric_limits<T>::min()};
  std::vector<Bin<T>> bins_{};

  /// Update the histogram with the provided value.
  auto update(const T& value, const T& count) -> void {
    auto append = false;
    auto index = size_t(0);

    // Update min/max values.
    if (min_ > value) {
      min_ = value;
    }
    if (max_ < value) {
      max_ = value;
    }

    // If the histogram is not empty, the insertion index is calculated.
    if (!bins_.empty()) {
      if (value <= bins_.front().value) {
        index = 0;
      } else if (value >= bins_.back().value) {
        index = bins_.size() - 1;
        append = true;
      } else {
        // Binary search for the insertion index.
        auto it = std::lower_bound(
            bins_.begin(), bins_.end(), value,
            [](const Bin<T>& lhs, const T& rhs) { return lhs.value <= rhs; });
        index = std::distance(bins_.begin(), it);
      }

      // If the value is already in the histogram, the count is updated.
      auto& bin = bins_[index];
      if (bin.value == value) {
        bin.count += count;
        return;
      }
    }

    if (append) {
      // If the new value is greater than the last value inserted in the
      // histogram, the bins is extended.
      bins_.emplace_back(Bin<T>{value, count});
    } else {
      // Otherwise, the new value is inserted.
      bins_.insert(bins_.begin() + index, {value, count});
    }
  }

  /// Compress the histogram if necessary.
  auto trim() -> void {
    while (bins_.size() > bin_count_) {
      // Find a point qi that minimizes qi+1 - qi
      auto ix = BinDifferences<T>(bins_, weighted_diff_).index();

      // Replace the bins (qi, ki), (qi+1, ki+1)
      auto& item0 = bins_[ix];
      const auto& item1 = bins_[++ix];

      // by the bin ((qi * ki + qi+1 * ki+1) / (ki + ki+1), ki + ki+1)
      item0 = {(item0.value * item0.count + item1.value * item1.count) /
                   (item0.count + item1.count),
               item0.count + item1.count};
      bins_.erase(bins_.begin() + ix);
    }
  }

  /// Calculates a moment of order n
  constexpr auto moment(const T& c, const T& n) const -> T {
    auto sum_m = T(0);
    auto sum_counts = T(0);
    for (const auto& item : bins_) {
      sum_m += item.count * std::pow((item.value - c), n);
      sum_counts += item.count;
    }
    return sum_m / sum_counts;
  }
};

}  // namespace pyinterp::detail::math

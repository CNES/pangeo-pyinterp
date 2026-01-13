// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <utility>

#include "pyinterp/broadcast.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/math/fill.hpp"
#include "pyinterp/math/tdigest.hpp"
#include "pyinterp/pybind/axis.hpp"
#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::pybind {

/// @brief 2D histogram for binning continuous values into a grid using
/// TDigest.
///
/// Groups continuous values into bins located on a 2D grid. Each bin maintains
/// statistical distributions using the TDigest algorithm for efficient quantile
/// estimation and statistical analysis.
///
/// @tparam T Floating-point type for values
template <std::floating_point T>
class Histogram2D {
 public:
  /// Type alias for the statistical digest used in each bin
  using Digest = math::TDigest<T>;

  /// @brief Construct a 2D histogram with specified axes.
  ///
  /// @param[in] x Definition of bin centers for the X axis
  /// @param[in] y Definition of bin centers for the Y axis
  /// @param[in] compression Optional TDigest compression parameter (default:
  /// 100). Higher values provide better accuracy at the cost of memory.
  Histogram2D(Axis<double> x, Axis<double> y,
              const std::optional<size_t> compression = std::nullopt)
      : x_axis_{std::move(x)},
        y_axis_{std::move(y)},
        compression_{compression.value_or(100)},
        bins_{x_axis_.size(), y_axis_.size()} {
    initialize_bins();
  }

  /// Default destructor
  ~Histogram2D() = default;

  /// Copy constructor
  Histogram2D(const Histogram2D&) = default;

  /// Move constructor
  Histogram2D(Histogram2D&&) noexcept = default;

  /// Copy assignment operator deleted (immutable design)
  auto operator=(const Histogram2D&) -> Histogram2D& = delete;

  /// Move assignment operator deleted (immutable design)
  auto operator=(Histogram2D&&) noexcept -> Histogram2D& = delete;

  /// @brief Insert new values into the histogram bins.
  ///
  /// Values are binned according to their (x, y) coordinates. NaN values are
  /// automatically filtered out.
  /// @param[in] x X coordinates (1D array)
  /// @param[in] y Y coordinates (1D array)
  /// @param[in] z Values to accumulate (1D array)
  /// @throw std::invalid_argument If arrays have mismatched shapes
  auto push(const Eigen::Ref<const Vector<T>>& x,
            const Eigen::Ref<const Vector<T>>& y,
            const Eigen::Ref<const Vector<T>>& z) -> void {
    broadcast::check_eigen_shape("x", x, "y", y, "z", z);

    const auto& x_axis = static_cast<const math::Axis<double>&>(x_axis_);
    const auto& y_axis = static_cast<const math::Axis<double>&>(y_axis_);

    for (auto [xi, yi, zi] : std::ranges::views::zip(x, y, z)) {
      if (math::Fill<T>::is_fill_value(zi)) [[unlikely]] {
        continue;
      }

      const auto ix = x_axis.find_index(xi, true);
      const auto iy = y_axis.find_index(yi, true);

      if (ix != -1 && iy != -1) [[likely]] {
        bins_(ix, iy).add(zi);
      }
    }
  }

  /// @brief Reset all statistics in the histogram.
  auto clear() noexcept -> void {
    for (auto& bin : bins_.reshaped()) {
      bin.clear();
    }
  }

  /// @brief Compute the count of points within each bin.
  /// @return 2D array of counts
  [[nodiscard]] auto count() const -> Matrix<uint64_t> {
    return compute_statistic<uint64_t>(
        [](const Digest& digest) { return digest.count(); });
  }

  /// @brief Compute the minimum value in each bin.
  /// @return 2D array of minimum values
  [[nodiscard]] auto min() const -> Matrix<T> {
    return compute_statistic<T>(
        [](const Digest& digest) { return digest.min(); });
  }

  /// @brief Compute the maximum value in each bin.
  /// @return 2D array of maximum values
  [[nodiscard]] auto max() const -> Matrix<T> {
    return compute_statistic<T>(
        [](const Digest& digest) { return digest.max(); });
  }

  /// @brief Compute the mean value in each bin.
  /// @return 2D array of mean values
  [[nodiscard]] auto mean() const -> Matrix<T> {
    return compute_statistic<T>(
        [](const Digest& digest) { return digest.mean(); });
  }

  /// @brief Compute the specified quantile for each bin.
  /// @param[in] q Quantile in range [0, 1]
  /// @return 2D array of quantile values
  [[nodiscard]] auto quantile(const T q) const -> Matrix<T> {
    return compute_statistic<T>(
        [q](const Digest& digest) { return digest.quantile(q); });
  }

  /// @brief Compute the sum of weights in each bin.
  /// @return 2D array of weight sums
  [[nodiscard]] auto sum_of_weights() const -> Matrix<T> {
    return compute_statistic<T>(
        [](const Digest& digest) { return digest.sum_of_weights(); });
  }

  /// @brief Get the X-axis.
  /// @return Shared pointer to the X-axis
  [[nodiscard]] constexpr auto x() const noexcept -> const Axis<double>& {
    return x_axis_;
  }

  /// @brief Get the Y-axis.
  /// @return Shared pointer to the Y-axis
  [[nodiscard]] constexpr auto y() const noexcept -> const Axis<double>& {
    return y_axis_;
  }

  /// @brief Serialize the histogram state for storage or transmission.
  /// @return Serialized state as a Writer object
  [[nodiscard]] auto pack() const -> serialization::Writer {
    serialization::Writer writer;

    // Write magic number for validation
    writer.write(kMagicNumber);

    // Append axes
    writer.append(x_axis_.pack());
    writer.append(y_axis_.pack());

    // Write compression parameter
    writer.write(compression_);

    // Write dimensions
    const auto rows = bins_.rows();
    const auto cols = bins_.cols();
    writer.write(static_cast<size_t>(rows));
    writer.write(static_cast<size_t>(cols));

    // Append each TDigest directly
    for (const auto& bin : bins_.reshaped()) {
      writer.append(bin.pack());
    }

    return writer;
  }

  /// @brief Deserialize a histogram from serialized state.
  /// @param[in,out] state Reference to serialization Reader
  /// @return New Histogram2D instance with restored properties
  /// @throw std::invalid_argument If the state is invalid
  [[nodiscard]] static auto unpack(serialization::Reader& state)
      -> Histogram2D<T> {
    if (state.size() == 0) {
      throw std::invalid_argument("Cannot unpack Histogram2D from empty state");
    }

    // Read and validate magic number
    const auto magic = state.read<uint32_t>();
    if (magic != kMagicNumber) {
      throw std::invalid_argument(
          "Invalid Histogram2D serialization magic number");
    }

    // Read axes
    Axis<double> x_axis(Axis<double>::unpack(state));
    Axis<double> y_axis(Axis<double>::unpack(state));

    // Read compression parameter
    const auto compression = state.read<size_t>();

    // Read dimensions
    const auto rows = state.read<size_t>();
    const auto cols = state.read<size_t>();

    // Validate dimensions
    if (rows != static_cast<size_t>(x_axis.size()) ||
        cols != static_cast<size_t>(y_axis.size())) {
      throw std::invalid_argument(
          "Histogram2D dimensions don't match axis sizes");
    }

    // Create histogram
    auto result = Histogram2D<T>(x_axis, y_axis, compression);

    // Read each TDigest
    for (auto& bin : result.bins_.reshaped()) {
      bin = math::TDigest<T>::unpack(state);
    }
    return result;
  }

  /// @brief Merge another histogram into this one.
  /// @param[in] other The histogram to merge
  /// @return Reference to this histogram
  /// @throw std::invalid_argument If histograms have incompatible grids
  auto operator+=(const Histogram2D& other) -> Histogram2D& {
    if (x_axis_ != other.x_axis_ || y_axis_ != other.y_axis_) {
      throw std::invalid_argument(
          "Cannot combine histograms with different grids");
    }

    for (auto [lhs, rhs] :
         std::ranges::views::zip(bins_.reshaped(), other.bins_.reshaped())) {
      lhs += rhs;
    }

    return *this;
  }

 private:
  /// Magic number for histogram serialization
  static constexpr uint32_t kMagicNumber = 0x48495354;  // "HIST"

  /// X-axis definition
  Axis<double> x_axis_;

  /// Y-axis definition
  Axis<double> y_axis_;

  /// TDigest compression parameter
  size_t compression_;

  /// 2D grid of TDigest objects for statistical accumulation
  Matrix<Digest> bins_;

  /// @brief Initialize all bins with the compression parameter.
  auto initialize_bins() -> void {
    for (auto& bin : bins_.reshaped()) {
      bin.set_compression(compression_);
    }
  }

  /// @brief Generic statistic computation with a callable.
  ///
  /// Applies the provided function to each bin and returns the results as a
  /// 2D array. Uses C++20 ranges and concepts for type safety.
  /// @tparam Result The return type of the statistic
  /// @tparam Func Callable type (auto-deduced)
  /// @param[in] func Function to apply to each Digest
  /// @return 2D array of computed statistics
  template <typename Result, std::invocable<const Digest&> Func>
  [[nodiscard]] auto compute_statistic(Func&& func) const -> Matrix<Result> {
    const auto rows = x_axis_.size();
    const auto cols = y_axis_.size();
    // Allocate output array
    auto result = Matrix<Result>(rows, cols);

    for (auto [lhs, rhs] :
         std::ranges::views::zip(result.reshaped(), bins_.reshaped())) {
      lhs = std::invoke(func, rhs);
    }

    return result;
  }
};

/// @brief Initialize Histogram2D Python bindings
/// @param m The nanobind module to register bindings in
auto init_histogram2d(nanobind::module_& m) -> void;

}  // namespace pyinterp::pybind

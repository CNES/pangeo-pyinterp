// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "pyinterp/math/fill.hpp"
#include "pyinterp/math/interpolate/bivariate.hpp"
#include "pyinterp/math/interpolate/univariate.hpp"

namespace pyinterp::math::interpolate::bivariate {

/// @brief Bivariate spline interpolation using separable 1D interpolation
///
/// This class implements 2D interpolation by applying 1D interpolation
/// along each axis sequentially. The order of interpolation is chosen
/// based on the matrix storage order for optimal cache performance.
///
/// @tparam T type of the data (must be floating point)
///
/// @note Thread safety: Not thread-safe due to internal state (interp_values_).
///       Use separate Spline instances for concurrent interpolation.
template <std::floating_point T>
class Spline : public BivariateBase<T> {
 public:
  /// @brief Constructor
  ///
  /// @param interpolator Unique pointer to the univariate interpolator
  explicit Spline(std::unique_ptr<interpolate::Univariate<T>> interpolator)
      : interpolator_(std::move(interpolator)), interp_values_() {
    // Pre-allocate reasonable initial size to reduce early allocations
    interp_values_.resize(initial_capacity_);
  }

  /// @brief The minimum size of the arrays to be interpolated.
  /// @return Minimum size of the arrays
  [[nodiscard]] constexpr auto min_size() const -> int64_t final {
    return interpolator_->min_size();
  }

  /// @brief Return the interpolated value at point (x, y)
  ///
  /// Performs separable 2D interpolation by interpolating along the Y-axis
  /// first, then X-axis (optimal for column-major storage, which is the
  /// internal storage format after prepare()).
  ///
  /// @param[in] x X-coordinate for interpolation
  /// @param[in] y Y-coordinate for interpolation
  /// @return Interpolated value at (x, y)
  ///
  /// @throws std::runtime_error if grid is empty
  [[nodiscard]] auto operator()(const T& x, const T& y) -> T final {
    // Check if interpolator was properly initialized BEFORE accessing data
    if (!this->is_valid()) [[unlikely]] {
      return math::Fill<T>::value();
    }

    const auto& xa = this->xa();
    const auto& ya = this->ya();
    const auto& za = this->za();

    return interpolate_y_then_x(xa, ya, za, x, y);
  }

  // Batch interpolation is inherited from BivariateBase<T>
  using BivariateBase<T>::operator();

 private:
  /// @brief Interpolate along Y-axis first, then X-axis
  ///
  /// Optimal when the first dimension (rows) is contiguous in memory
  /// (column-major storage). This minimizes cache misses during the first
  /// interpolation pass.
  ///
  /// Algorithm:
  /// 1. For each row i, interpolate f(x_i, y) along Y at position x
  ///    → produces n_rows intermediate values
  /// 2. Interpolate those intermediate values along X at position y
  ///    → produces final result
  ///
  /// @param[in] xa X-axis coordinates
  /// @param[in] ya Y-axis coordinates
  /// @param[in] za Grid data
  /// @param[in] x X-coordinate
  /// @param[in] y Y-coordinate
  /// @return Interpolated value
  auto interpolate_y_then_x(const Eigen::Ref<const Vector<T>>& xa,
                            const Eigen::Ref<const Vector<T>>& ya,
                            const Eigen::Ref<const Matrix<T>>& za, const T x,
                            const T y) -> T {
    const auto n_rows = za.rows();

    // Ensure temporary storage has sufficient capacity
    ensure_capacity(n_rows);

    // Step 1: For each row, interpolate along Y-axis at the given X coordinate.
    // This produces a 1D array of values, one per row.
    for (int64_t ix = 0; ix < n_rows; ++ix) {
      // Extract row (all Y values at this X coordinate)
      const auto row = za.row(ix);
      // Interpolate along Y at position x
      interp_values_(ix) = (*interpolator_)(ya, row, y);
    }

    // Step 2: Interpolate the intermediate values along X-axis
    // interp_values_ may have extra capacity beyond n_rows; only use the first
    // n_rows elements
    return (*interpolator_)(xa, interp_values_.head(n_rows), x);
  }

  /// @brief Ensure temporary storage has sufficient capacity
  ///
  /// Resizes the temporary vector only when necessary, with a growth factor
  /// to amortize allocation costs. Uses a 1.5x growth factor for reasonable
  /// space/time tradeoff.
  ///
  /// @param[in] required_size Minimum required size
  inline void ensure_capacity(const int64_t required_size) {
    if (interp_values_.size() < required_size) {
      // Grow with 1.5x factor to amortize future allocations
      // This gives O(log n) allocations for n calls with increasing sizes
      const auto new_size = std::max(
          required_size, static_cast<int64_t>(interp_values_.size() * 1.5));
      interp_values_.resize(new_size);
    }
  }

  /// Univariate interpolator instance
  /// Reused for both X and Y directions to avoid repeated allocations
  std::unique_ptr<interpolate::Univariate<T>> interpolator_;

  /// Temporary storage for intermediate interpolation results
  /// Stores one value per row, used during interpolation.
  /// Size grows dynamically but never shrinks to amortize allocation costs
  Vector<T> interp_values_;

  /// Initial capacity for temporary storage
  static constexpr int64_t initial_capacity_{16};
};

}  // namespace pyinterp::math::interpolate::bivariate

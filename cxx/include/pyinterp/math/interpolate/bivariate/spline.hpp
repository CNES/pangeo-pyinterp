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

#include "Eigen/Core"
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

  /// @brief Return the interpolated value at point (x, y)
  ///
  /// Performs separable 2D interpolation by interpolating along one axis
  /// first, then along the other. The order is chosen based on the grid's
  /// storage order for optimal cache performance.
  ///
  /// @param[in] xa X-axis coordinates
  /// @param[in] ya Y-axis coordinates
  /// @param[in] za Grid data
  /// @param[in] x X-coordinate for interpolation
  /// @param[in] y Y-coordinate for interpolation
  /// @return Interpolated value at (x, y)
  ///
  /// @throws std::runtime_error if grid is empty
  [[nodiscard]] auto operator()(const Eigen::Ref<const Vector<T>>& xa,
                                const Eigen::Ref<const Vector<T>>& ya,
                                const Eigen::Ref<const Matrix<T>>& za,
                                const T& x, const T& y) -> T final {
    // Validate grid dimensions
    if (za.size() == 0) {
      throw std::runtime_error("Cannot interpolate on empty grid");
    }

    // Choose interpolation order based on memory layout for better cache
    // performance. When the last dimension is contiguous (row-major), we
    // interpolate along X first. When the first dimension is contiguous
    // (column-major), we interpolate along Y first.
    static_assert(Matrix<T>::RowsAtCompileTime == Eigen::Dynamic &&
                      Matrix<T>::ColsAtCompileTime == Eigen::Dynamic,
                  "This method assumes dynamic-size Eigen matrices");

    return za.IsRowMajor ? interpolate_x_then_y(xa, ya, za, x, y)
                         : interpolate_y_then_x(xa, ya, za, x, y);
  }

  // Batch interpolation is inherited from BivariateBase<T>
  using BivariateBase<T>::operator();

 private:
  /// @brief Interpolate along X-axis first, then Y-axis
  ///
  /// Optimal when the last dimension (columns) is contiguous in memory
  /// (row-major storage). This minimizes cache misses during the first
  /// interpolation pass.
  ///
  /// Algorithm:
  /// 1. For each column j, interpolate f(x, y_j) along X at position y
  ///    → produces n_cols intermediate values
  /// 2. Interpolate those intermediate values along Y at position x
  ///    → produces final result
  ///
  /// @param[in] xa X-axis coordinates
  /// @param[in] ya Y-axis coordinates
  /// @param[in] za Grid data
  /// @param[in] x X-coordinate
  /// @param[in] y Y-coordinate
  /// @return Interpolated value
  auto interpolate_x_then_y(const Eigen::Ref<const Vector<T>>& xa,
                            const Eigen::Ref<const Vector<T>>& ya,
                            const Eigen::Ref<const Matrix<T>>& za, const T x,
                            const T y) -> T {
    const auto n_cols = za.cols();

    // Ensure temporary storage has sufficient capacity
    ensure_capacity(n_cols);

    // Step 1: For each column, interpolate along X-axis at the given Y
    // coordinate. This produces a 1D array of values, one per column.
    for (int64_t ix = 0; ix < n_cols; ++ix) {
      // Extract column (all X values at this Y coordinate)
      const auto column = za.col(ix);
      // Interpolate along X at position y
      interp_values_(ix) = (*interpolator_)(xa, column, x);
    }

    // Step 2: Interpolate the intermediate values along Y-axis
    // interp_values_ may have extra capacity beyond n_cols; only use the first
    // n_cols elements
    return (*interpolator_)(ya, interp_values_.head(n_cols), y);
  }

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
  /// Stores one value per row or column, depending on interpolation order
  /// Size grows dynamically but never shrinks to amortize allocation costs
  Vector<T> interp_values_;

  /// Initial capacity for temporary storage
  static constexpr int64_t initial_capacity_{16};
};

}  // namespace pyinterp::math::interpolate::bivariate

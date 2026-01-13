// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <ranges>
#include <span>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/interpolate/univariate/cspline_base.hpp"

namespace pyinterp::math::interpolate::univariate {

/// Cubic spline interpolation using natural boundary conditions
/// @tparam T Type of the data to interpolate (must be floating point)
template <std::floating_point T>
class CSpline : public CSplineBase<T> {
 public:
  using CSplineBase<T>::CSplineBase;
  using CSplineBase<T>::operator();
  using CSplineBase<T>::derivative;

  /// @brief The minimum number of points required for the interpolation.
  /// @return Minimum number of points
  [[nodiscard]] constexpr auto min_size() const -> int64_t final { return 4; }

 private:
  /// @brief Compute the coefficients of the interpolation
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @return True if coefficients computed successfully
  [[nodiscard]] constexpr auto compute_coefficients(const Vector<T>& xa,
                                                    const Vector<T>& ya)
      -> bool final;

  /// @brief Solve a symmetric tridiagonal system using Thomas algorithm
  /// @param[in,out] x Output span for the solution
  constexpr auto solve_symmetric_tridiagonal(std::span<T> x) -> void;

  /// Temporary arrays for tridiagonal solver
  Vector<T> c_;
  Vector<T> d_;
};

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
constexpr auto CSpline<T>::solve_symmetric_tridiagonal(std::span<T> x) -> void {
  const auto size = this->A_.rows();
  const auto size_m1 = size - 1;
  const auto size_m2 = size - 2;

  // Forward elimination
  T item = T{1} / this->A_(0, 0);
  c_(0) = this->A_(0, 1) * item;
  d_(0) = this->b_(0) * item;

  for (auto i : std::views::iota(int64_t{1}, size_m1)) {
    item = this->A_(i, i - 1);
    const T inv_m = T{1} / (this->A_(i, i) - item * c_(i - 1));
    c_(i) = this->A_(i, i + 1) * inv_m;
    d_(i) = std::fma(-item, d_(i - 1), this->b_(i)) * inv_m;
  }

  // Last row
  item = this->A_(size_m1, size_m2);
  d_(size_m1) = (this->b_(size_m1) - item * d_(size_m2)) /
                (this->A_(size_m1, size_m1) - item * c_(size_m2));
  x[size_m1] = d_(size_m1);

  // Back substitution
  for (int64_t i = size_m2; i >= 0; --i) {
    x[i] = std::fma(-c_(i), x[i + 1], d_(i));
  }
}

template <std::floating_point T>
constexpr auto CSpline<T>::compute_coefficients(const Vector<T>& xa,
                                                const Vector<T>& ya) -> bool {
  if (!CSplineBase<T>::compute_coefficients(xa, ya)) [[unlikely]] {
    return false;
  }

  const auto size = xa.size();
  const auto size_m2 = size - 2;

  // Resize working arrays if needed
  if (this->x_.size() != size) {
    this->A_.resize(size_m2, size_m2);
    this->b_.resize(size_m2);
    this->x_.resize(size);
    c_.resize(size_m2);
    d_.resize(size_m2);

    this->A_.setZero();
    this->x_.setZero();
  }

  // Build tridiagonal system for natural cubic spline
  for (auto i : std::views::iota(int64_t{0}, size_m2)) {
    const T x1 = xa(i + 1);
    const T y1 = ya(i + 1);

    // Interval widths
    const T h_i0 = x1 - xa(i);
    const T h_i1 = xa(i + 2) - x1;

    // Slope differences
    const T y_i0 = y1 - ya(i);
    const T y_i1 = ya(i + 2) - y1;

    // Reciprocals (with zero check)
    const T g_i0 = (h_i0 != T{0}) ? T{1} / h_i0 : T{0};
    const T g_i1 = (h_i1 != T{0}) ? T{1} / h_i1 : T{0};

    // Fill tridiagonal matrix
    if (i > 0) {
      this->A_(i, i - 1) = h_i0;
    }
    this->A_(i, i) = T{2} * (h_i0 + h_i1);
    if (i < size_m2 - 1) {
      this->A_(i, i + 1) = h_i1;
    }

    // Right-hand side
    this->b_(i) = T{3} * std::fma(y_i1, g_i1, -y_i0 * g_i0);
  }

  // Solve for interior derivatives
  // (x_[0] and x_[size-1] remain zero - natural BC)
  solve_symmetric_tridiagonal(std::span<T>(this->x_.data() + 1, size_m2));

  return true;
}

}  // namespace pyinterp::math::interpolate::univariate

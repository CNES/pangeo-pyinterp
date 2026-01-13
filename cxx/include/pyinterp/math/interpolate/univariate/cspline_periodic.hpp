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

/// @brief Periodic cubic spline interpolation with cyclic boundary conditions
/// @tparam T Type of data handled by this interpolator (must be floating point)
template <std::floating_point T>
class CSplinePeriodic : public CSplineBase<T> {
 public:
  using CSplineBase<T>::CSplineBase;
  using CSplineBase<T>::operator();
  using CSplineBase<T>::derivative;

  /// @brief Returns the minimum number of points required for the
  /// interpolation.
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

  /// @brief Solve a symmetric cyclic tridiagonal system using Sherman-Morrison
  /// formula
  /// @param[in,out] x Output span for the solution
  constexpr auto solve_symmetric_cyclic_tridiagonal(std::span<T> x) -> void;

  /// Working arrays for cyclic tridiagonal solver
  Vector<T> alpha_{};
  Vector<T> gamma_{};
  Vector<T> delta_{};
  Vector<T> c_{};
  Vector<T> z_{};
};

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
constexpr auto CSplinePeriodic<T>::solve_symmetric_cyclic_tridiagonal(
    std::span<T> x) -> void {
  const auto size = this->A_.rows();

  // Handle trivial case
  if (size == 1) [[unlikely]] {
    x[0] = this->b_(0) / this->A_(0, 0);
    return;
  }

  const auto size_m1 = size - 1;
  const auto size_m2 = size - 2;
  const auto size_m3 = size - 3;

  // Forward elimination with Sherman-Morrison modification
  alpha_(0) = this->A_(0, 0);
  gamma_(0) = this->A_(0, 1) / alpha_(0);
  delta_(0) = this->A_(size_m1, 0) / alpha_(0);

  for (auto i : std::views::iota(int64_t{1}, size_m2)) {
    alpha_(i) = this->A_(i, i) - this->A_(i, i - 1) * gamma_(i - 1);
    gamma_(i) = this->A_(i, i + 1) / alpha_(i);
    delta_(i) = -delta_(i - 1) * this->A_(i, i - 1) / alpha_(i);
  }

  // Compute sum of weighted deltas
  const T sum_delta = alpha_.segment(0, size_m2)
                          .cwiseProduct(delta_.segment(0, size_m2))
                          .cwiseProduct(delta_.segment(0, size_m2))
                          .sum();

  alpha_(size_m2) =
      this->A_(size_m2, size_m2) - this->A_(size_m2, size_m3) * gamma_(size_m3);
  gamma_(size_m2) = (this->A_(size_m2, size_m1) -
                     this->A_(size_m2, size_m3) * delta_(size_m3)) /
                    alpha_(size_m2);
  alpha_(size_m1) = this->A_(size_m1, size_m1) - sum_delta -
                    alpha_(size_m2) * gamma_(size_m2) * gamma_(size_m2);

  // Forward substitution for right-hand side
  z_(0) = this->b_(0);
  for (auto i : std::views::iota(int64_t{1}, size_m1)) {
    z_(i) = this->b_(i) - z_(i - 1) * gamma_(i - 1);
  }

  const T sum_z =
      delta_.segment(0, size_m2).cwiseProduct(z_.segment(0, size_m2)).sum();

  z_(size_m1) = this->b_(size_m1) - sum_z - gamma_(size_m2) * z_(size_m2);

  // Solve intermediate system
  c_.segment(0, size) =
      z_.segment(0, size).array() / alpha_.segment(0, size).array();

  // Back substitution
  x[size_m1] = c_(size_m1);
  x[size_m2] = c_(size_m2) - gamma_(size_m2) * x[size_m1];

  if (size >= 3) {
    for (int64_t i = size_m3; i >= 0; --i) {
      x[i] = c_(i) - gamma_(i) * x[i + 1] - delta_(i) * x[size_m1];
    }
  }
}

template <std::floating_point T>
constexpr auto CSplinePeriodic<T>::compute_coefficients(const Vector<T>& xa,
                                                        const Vector<T>& ya)
    -> bool {
  if (!CSplineBase<T>::compute_coefficients(xa, ya)) [[unlikely]] {
    return false;
  }

  const auto size = xa.size();

  // Resize working arrays if needed
  if (this->x_.size() < size) {
    this->A_.resize(size - 1, size - 1);
    this->b_.resize(size - 1);
    this->x_.resize(size);
    alpha_.resize(size - 1);
    gamma_.resize(size - 1);
    delta_.resize(size - 1);
    c_.resize(size);
    z_.resize(size);

    this->A_.setZero();
    this->x_.setZero();
  }

  // Build cyclic tridiagonal system for periodic boundary conditions
  for (auto i : std::views::iota(int64_t{0}, size - 2)) {
    // Interval widths
    const T h_i0 = xa(i + 1) - xa(i);
    const T h_i1 = xa(i + 2) - xa(i + 1);

    // Slope differences
    const T y_i0 = ya(i + 1) - ya(i);
    const T y_i1 = ya(i + 2) - ya(i + 1);

    // Reciprocals (with zero check)
    const T g_i0 = (h_i0 != T{0}) ? T{1} / h_i0 : T{0};
    const T g_i1 = (h_i1 != T{0}) ? T{1} / h_i1 : T{0};

    // Fill tridiagonal entries
    this->A_(i + 1, i) = h_i1;
    this->A_(i, i) = T{2} * (h_i0 + h_i1);
    this->A_(i, i + 1) = h_i1;

    // Right-hand side
    this->b_(i) = T{3} * std::fma(y_i1, g_i1, -y_i0 * g_i0);
  }

  // Handle wraparound for periodic boundary
  const auto i = size - 2;
  const T h_i0 = xa(i + 1) - xa(i);
  const T h_i1 = xa(1) - xa(0);
  const T y_i0 = ya(i + 1) - ya(i);
  const T y_i1 = ya(1) - ya(0);
  const T g_i0 = (h_i0 != T{0}) ? T{1} / h_i0 : T{0};
  const T g_i1 = (h_i1 != T{0}) ? T{1} / h_i1 : T{0};

  this->A_(i, 0) = h_i1;
  this->A_(i, i) = T{2} * (h_i0 + h_i1);
  this->A_(0, i) = h_i1;
  this->b_(i) = T{3} * std::fma(y_i1, g_i1, -y_i0 * g_i0);

  // Solve cyclic system (skip first element, solve for interior/last)
  solve_symmetric_cyclic_tridiagonal(
      std::span<T>(this->x_.data() + 1, size - 1));

  // Enforce periodicity: first derivative equals last
  this->x_(0) = this->x_(size - 1);

  return true;
}

}  // namespace pyinterp::math::interpolate::univariate

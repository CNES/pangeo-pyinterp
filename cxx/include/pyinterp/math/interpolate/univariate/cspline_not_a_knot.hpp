// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <ranges>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/interpolate/univariate/cspline_base.hpp"

namespace pyinterp::math::interpolate::univariate {

/// @brief Cubic spline interpolation with not-a-knot end conditions.
/// Not-a-knot conditions force the third derivative to be continuous at the
/// second and second-to-last interior points, resulting in a more natural
/// curve.
///
/// This implementation uses a reduced tridiagonal system approach:
/// - The not-a-knot conditions are used to express M[0] and M[n-1] in terms
///   of interior values
/// - This reduces the n×n system to an (n-2)×(n-2) tridiagonal system
/// - Solved efficiently using Thomas algorithm in O(n) time
///
/// @tparam T Data type
template <std::floating_point T>
class CSplineNotAKnot : public CSplineBase<T> {
 public:
  using CSplineBase<T>::operator();
  using CSplineBase<T>::derivative;

  /// @brief Default constructor.
  CSplineNotAKnot() : CSplineBase<T>() {}

  /// @brief Minimum number of data points required is 4.
  /// @return Minimum number of data points required
  [[nodiscard]] constexpr auto min_size() const -> int64_t final { return 4; }

 protected:
  /// @brief Compute the spline coefficients using an optimized O(n) algorithm.
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @return True if coefficients computed successfully
  [[nodiscard]] constexpr auto compute_coefficients(const Vector<T>& xa,
                                                    const Vector<T>& ya)
      -> bool final;

 private:
  /// @brief Solve the reduced tridiagonal system using Thomas algorithm
  /// @param[in] h Step sizes array
  /// @param[in] size_m2 Size of the reduced system (n-2)
  constexpr auto solve_reduced_tridiagonal(const Vector<T>& h, int64_t size_m2)
      -> void;

  /// Temporary arrays for tridiagonal solver (Thomas algorithm)
  Vector<T> c_;  // Modified superdiagonal
  Vector<T> d_;  // Modified RHS
};

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
constexpr auto CSplineNotAKnot<T>::solve_reduced_tridiagonal(const Vector<T>& h,
                                                             int64_t size_m2)
    -> void {
  // Thomas algorithm for the modified tridiagonal system
  // The system has the form: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
  // Where:
  //   a[i] = h[i]      (subdiagonal)
  //   b[i] = 2*(h[i] + h[i+1])  (diagonal) - with modifications at boundaries
  //   c[i] = h[i+1]    (superdiagonal)

  // Forward elimination
  // First row is modified due to not-a-knot condition
  T diag_0 = this->A_(0, 0);
  T inv_diag = T{1} / diag_0;
  c_(0) = this->A_(0, 1) * inv_diag;
  d_(0) = this->b_(0) * inv_diag;

  // Interior rows
  for (auto i : std::views::iota(int64_t{1}, size_m2 - 1)) {
    const T sub = h(i);  // Subdiagonal element
    const T diag = this->A_(i, i);
    const T super = h(i + 1);  // Superdiagonal element

    inv_diag = T{1} / std::fma(-sub, c_(i - 1), diag);
    c_(i) = super * inv_diag;
    d_(i) = std::fma(-sub, d_(i - 1), this->b_(i)) * inv_diag;
  }

  // Last row (modified due to not-a-knot condition)
  const auto last = size_m2 - 1;
  const T sub_last = this->A_(last, last - 1);
  inv_diag = T{1} / std::fma(-sub_last, c_(last - 1), this->A_(last, last));
  d_(last) = std::fma(-sub_last, d_(last - 1), this->b_(last)) * inv_diag;

  // Back substitution - store in interior positions of x_
  this->x_(size_m2) = d_(last);  // M[n-2] stored at index size_m2 = n-2
  for (int64_t i = last - 1; i >= 0; --i) {
    this->x_(i + 1) = std::fma(-c_(i), this->x_(i + 2), d_(i));
  }
}

template <std::floating_point T>
constexpr auto CSplineNotAKnot<T>::compute_coefficients(const Vector<T>& xa,
                                                        const Vector<T>& ya)
    -> bool {
  if (!CSplineBase<T>::compute_coefficients(xa, ya)) [[unlikely]] {
    return false;
  }

  const auto size = xa.size();
  const auto size_m1 = size - 1;
  const auto size_m2 = size - 2;
  const auto size_m3 = size - 3;

  // Use tolerance for numerical stability
  constexpr T epsilon = std::numeric_limits<T>::epsilon() * T{10};

  // Compute step sizes: h_i = x[i+1] - x[i]
  const Vector<T> h = xa.tail(size_m1) - xa.head(size_m1);

  // Check for duplicate x values
  if ((h.array().abs() < epsilon).any()) [[unlikely]] {
    return false;
  }

  // Resize working arrays if needed
  if (this->x_.size() != size) {
    this->A_.resize(size_m2, size_m2);
    this->b_.resize(size_m2);
    this->x_.resize(size);
    c_.resize(size_m2);
    d_.resize(size_m2);

    this->A_.setZero();
    this->x_.setZero();
  } else {
    this->A_.setZero();
    this->x_.setZero();
  }

  // Compute divided differences: delta_i = (y[i+1] - y[i]) / h[i]
  const Vector<T> delta =
      (ya.tail(size_m1) - ya.head(size_m1)).array() / h.array();

  // =========================================================================
  // Build the reduced (n-2)×(n-2) tridiagonal system for M[1] ... M[n-2]
  //
  // The not-a-knot conditions allow us to express M[0] and M[n-1] as:
  //   M[0] = α₀ * M[1] + β₀ * M[2]
  //   M[n-1] = αₙ * M[n-3] + βₙ * M[n-2]
  //
  // Where the coefficients come from the continuity of third derivative.
  // =========================================================================

  // Not-a-knot coefficients for left boundary (M[0] in terms of M[1], M[2])
  // From: -h[1]*M[0] + (h[0]+h[1])*M[1] - h[0]*M[2] = 0
  // =>    M[0] = ((h[0]+h[1])/h[1])*M[1] - (h[0]/h[1])*M[2]
  const T alpha_0 = (h(0) + h(1)) / h(1);
  const T beta_0 = -h(0) / h(1);

  // Not-a-knot coefficients for right boundary (M[n-1] in terms of M[n-3],
  // M[n-2]) From: h[n-2]*M[n-3] - (h[n-3]+h[n-2])*M[n-2] + h[n-3]*M[n-1] = 0
  // =>    M[n-1] = -(h[n-2]/h[n-3])*M[n-3] + ((h[n-3]+h[n-2])/h[n-3])*M[n-2]
  const T alpha_n = -h(size_m2) / h(size_m3);
  const T beta_n = (h(size_m3) + h(size_m2)) / h(size_m3);

  // Build the tridiagonal system
  // Standard equation for row i (1 <= i <= n-2):
  //   h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1] = 3*(delta[i] -
  //   delta[i-1])
  //
  // For i=1: substitute M[0] = α₀*M[1] + β₀*M[2]
  // For i=n-2: substitute M[n-1] = αₙ*M[n-3] + βₙ*M[n-2]

  // Row 0 in reduced system (corresponds to equation at i=1)
  // Original: h[0]*M[0] + 2*(h[0]+h[1])*M[1] + h[1]*M[2] = 3*(delta[1] -
  // delta[0]) Substitute M[0]: h[0]*(α₀*M[1] + β₀*M[2]) + 2*(h[0]+h[1])*M[1] +
  // h[1]*M[2]
  //   = (h[0]*α₀ + 2*(h[0]+h[1]))*M[1] + (h[0]*β₀ + h[1])*M[2]
  this->A_(0, 0) = std::fma(h(0), alpha_0, T{2} * (h(0) + h(1)));
  this->A_(0, 1) = std::fma(h(0), beta_0, h(1));
  this->b_(0) = T{3} * (delta(1) - delta(0));

  // Interior rows (i = 2 to n-3, which are rows 1 to n-4 in reduced system)
  for (auto i : std::views::iota(int64_t{2}, size_m2)) {
    const auto row = i - 1;  // Row index in reduced system
    this->A_(row, row - 1) = h(i - 1);
    this->A_(row, row) = T{2} * (h(i - 1) + h(i));
    if (row + 1 < size_m2) {
      this->A_(row, row + 1) = h(i);
    }
    this->b_(row) = T{3} * (delta(i) - delta(i - 1));
  }

  // Last row in reduced system (corresponds to equation at i=n-2)
  // Original: h[n-3]*M[n-3] + 2*(h[n-3]+h[n-2])*M[n-2] + h[n-2]*M[n-1] =
  // 3*(delta[n-2] - delta[n-3]) Substitute M[n-1]: h[n-3]*M[n-3] +
  // 2*(h[n-3]+h[n-2])*M[n-2] + h[n-2]*(αₙ*M[n-3] + βₙ*M[n-2])
  //   = (h[n-3] + h[n-2]*αₙ)*M[n-3] + (2*(h[n-3]+h[n-2]) + h[n-2]*βₙ)*M[n-2]
  const auto last_row = size_m2 - 1;
  if (last_row > 0) {
    this->A_(last_row, last_row - 1) =
        std::fma(h(size_m2), alpha_n, h(size_m3));
  }
  this->A_(last_row, last_row) =
      std::fma(h(size_m2), beta_n, T{2} * (h(size_m3) + h(size_m2)));
  this->b_(last_row) = T{3} * (delta(size_m2) - delta(size_m3));

  // Solve the reduced tridiagonal system for M[1] ... M[n-2]
  solve_reduced_tridiagonal(h, size_m2);

  // Recover boundary values using not-a-knot relations
  // M[0] = α₀ * M[1] + β₀ * M[2]
  this->x_(0) = std::fma(alpha_0, this->x_(1), beta_0 * this->x_(2));

  // M[n-1] = αₙ * M[n-3] + βₙ * M[n-2]
  this->x_(size_m1) =
      std::fma(alpha_n, this->x_(size_m3), beta_n * this->x_(size_m2));

  return true;
}

}  // namespace pyinterp::math::interpolate::univariate

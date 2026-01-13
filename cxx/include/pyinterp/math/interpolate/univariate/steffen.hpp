// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <ranges>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/fill.hpp"
#include "pyinterp/math/interpolate/univariate.hpp"

namespace pyinterp::math::interpolate::univariate {

/// @brief Steffen interpolation
/// A monotonicity-preserving cubic interpolation method that avoids
/// overshooting and produces visually pleasing curves. Based on Steffen (1990):
/// "A Simple Method for Monotonic Interpolation in One Dimension"
/// @tparam T Data type
template <std::floating_point T>
class Steffen : public Univariate<T> {
 public:
  using Univariate<T>::Univariate;
  using Univariate<T>::operator();
  using Univariate<T>::derivative;

  /// @brief Returns the minimum number of points required for the
  /// interpolation.
  /// @return Minimum number of points required
  [[nodiscard]] constexpr auto min_size() const -> int64_t final { return 3; }

 private:
  /// The slopes at each data point (first derivatives)
  Vector<T> y_prime_;

  /// @brief Compute the coefficients of the interpolation
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @return True if coefficients computed successfully
  [[nodiscard]] constexpr auto compute_coefficients(const Vector<T>& xa,
                                                    const Vector<T>& ya)
      -> bool final;

  /// @brief Interpolation using Steffen's cubic polynomials
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] x The point where the interpolation must be calculated.
  /// @return The interpolated value at the point x.
  [[nodiscard]] constexpr auto interpolate_(const Vector<T>& xa,
                                            const Vector<T>& ya,
                                            const T x) const -> T final;

  /// @brief Return the derivative of the interpolation function at point x.
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] x The point where the derivative must be calculated.
  /// @return The derivative of the interpolation function at the point x.
  [[nodiscard]] constexpr auto derivative_(const Vector<T>& xa,
                                           const Vector<T>& ya, const T x) const
      -> T final;

  /// @brief Return the value x with the sign of y (similar to std::copysign but
  /// for sign transfer)
  [[nodiscard]] static constexpr auto copysign(const T x, const T y) noexcept
      -> T {
    return (x < T{0} && y > T{0}) || (x > T{0} && y < T{0}) ? -x : x;
  }
};

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
constexpr auto Steffen<T>::compute_coefficients(const Vector<T>& xa,
                                                const Vector<T>& ya) -> bool {
  if (!Univariate<T>::compute_coefficients(xa, ya)) [[unlikely]] {
    return false;
  }

  const auto size = xa.size();

  if (y_prime_.size() < size) {
    y_prime_.resize(size);
  }

  // Left boundary: use "simplest possibility" method (section 2.2)
  const T h0 = xa(1) - xa(0);
  const T s0 = (ya(1) - ya(0)) / h0;
  y_prime_(0) = s0;

  // Interior points: compute slopes using Steffen's monotonicity-preserving
  // formula
  for (const auto i : std::views::iota(int64_t{1}, size - 1)) {
    // Interval widths (Eq. 6)
    const T hi = xa(i + 1) - xa(i);
    const T him1 = xa(i) - xa(i - 1);

    // Segment slopes (Eq. 7)
    const T si = (ya(i + 1) - ya(i)) / hi;
    const T sim1 = (ya(i) - ya(i - 1)) / him1;

    // Parabolic slope estimate (Eq. 8)
    const T pi = (sim1 * hi + si * him1) / (him1 + hi);

    // Steffen's monotonicity-preserving slope (combination of sign and
    // magnitude limits)
    const T sign_sum = copysign(T{1}, sim1) + copysign(T{1}, si);
    const T min_slope =
        std::min({std::abs(sim1), std::abs(si), T{0.5} * std::abs(pi)});

    y_prime_(i) = sign_sum * min_slope;
  }

  // Right boundary: use "simplest possibility" method (section 2.2)
  y_prime_(size - 1) =
      (ya(size - 1) - ya(size - 2)) / (xa(size - 1) - xa(size - 2));

  return true;
}

template <std::floating_point T>
constexpr auto Steffen<T>::interpolate_(const Vector<T>& xa,
                                        const Vector<T>& ya, const T x) const
    -> T {
  const auto where = this->search(xa, x);
  if (!where) [[unlikely]] {
    return Fill<T>::value();
  }

  const auto [i0, i1] = *where;

  // Interval properties
  const T h = x - xa(i0);
  const T hi = xa(i1) - xa(i0);
  const T si = (ya(i1) - ya(i0)) / hi;

  // Cubic polynomial coefficients: y = d + c*h + b*h² + a*h³
  const T hi_sq = hi * hi;
  const T a = (y_prime_(i0) + y_prime_(i1) - T{2} * si) / hi_sq;
  const T b = (T{3} * si - T{2} * y_prime_(i0) - y_prime_(i1)) / hi;
  const T c = y_prime_(i0);
  const T d = ya(i0);

  // Evaluate using Horner's method: d + h*(c + h*(b + h*a))
  return d + h * std::fma(h, std::fma(h, a, b), c);
}

template <std::floating_point T>
constexpr auto Steffen<T>::derivative_(const Vector<T>& xa, const Vector<T>& ya,
                                       const T x) const -> T {
  const auto where = this->search(xa, x);
  if (!where) [[unlikely]] {
    return Fill<T>::value();
  }

  const auto [i0, i1] = *where;

  // Interval properties
  const T h = x - xa(i0);
  const T hi = xa(i1) - xa(i0);
  const T si = (ya(i1) - ya(i0)) / hi;

  // Derivative coefficients: dy/dx = c + 2*b*h + 3*a*h²
  const T hi_sq = hi * hi;
  const T a = (y_prime_(i0) + y_prime_(i1) - T{2} * si) / hi_sq;
  const T b = (T{3} * si - T{2} * y_prime_(i0) - y_prime_(i1)) / hi;
  const T c = y_prime_(i0);

  // Evaluate: c + h*(2*b + h*3*a)
  return c + h * std::fma(T{3} * h, a, T{2} * b);
}

}  // namespace pyinterp::math::interpolate::univariate

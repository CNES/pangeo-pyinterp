// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cmath>
#include <concepts>
#include <tuple>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/fill.hpp"
#include "pyinterp/math/interpolate/univariate.hpp"

namespace pyinterp::math::interpolate::univariate {

/// @brief Coefficients of the cubic spline interpolation
/// @tparam T Type of the data to interpolate
template <std::floating_point T>
class CSplineCoefficients {
 public:
  /// @brief Constructor
  /// @param[in] c0 The first derivative at the first point
  /// @param[in] c1 The first derivative at the last point
  constexpr CSplineCoefficients(const T c0, const T c1) noexcept
      : c0_{c0}, c1_{c1} {}

  /// @brief Compute the coefficients of the cubic spline interpolation
  /// @param[in] dx The distance between the two points
  /// @param[in] dy The difference between the two points
  /// @return The coefficients (b, c, d) of the cubic spline interpolation
  [[nodiscard]] constexpr auto operator()(const T dx, const T dy) const noexcept
      -> std::tuple<T, T, T> {
    constexpr T third = T{1} / T{3};

    const T b = (dy / dx) - dx * (c1_ + T{2} * c0_) * third;
    const T c = c0_;
    const T d = (c1_ - c0_) / (T{3} * dx);

    return {b, c, d};
  }

 private:
  T c0_;
  T c1_;
};

/// @brief Base class for cubic spline interpolation
/// @tparam T Type of the data to interpolate
template <std::floating_point T>
class CSplineBase : public Univariate<T> {
 public:
  using Univariate<T>::Univariate;
  using Univariate<T>::operator();
  using Univariate<T>::derivative;

 protected:
  Matrix<T> A_;  ///< Coefficient matrix for the spline system
  Vector<T> b_;  ///< Right-hand side vector for the spline system
  Vector<T> x_;  ///< Solution vector for the spline system

 private:
  /// @brief Interpolation using cubic spline
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] x The point where the interpolation must be calculated.
  /// @return The interpolated value at the point x.
  [[nodiscard]] constexpr auto interpolate_(const Vector<T>& xa,
                                            const Vector<T>& ya,
                                            const T x) const -> T final;

  /// @brief Returns the derivative of the interpolation function at point x.
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] x The point where the derivative must be calculated.
  /// @return The derivative of the interpolation function at the point x.
  [[nodiscard]] constexpr auto derivative_(const Vector<T>& xa,
                                           const Vector<T>& ya, const T x) const
      -> T final;
};

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
constexpr auto CSplineBase<T>::interpolate_(const Vector<T>& xa,
                                            const Vector<T>& ya,
                                            const T x) const -> T {
  const auto where = this->search(xa, x);
  if (!where) [[unlikely]] {
    return Fill<T>::value();
  }

  const auto [i0, i1] = *where;

  // Extract interval endpoints
  const T x_lo = xa(i0);
  const T y_lo = ya(i0);
  const T x_hi = xa(i1);
  const T y_hi = ya(i1);

  // Compute interval properties
  const T dx = x_hi - x_lo;
  const T dy = y_hi - y_lo;
  const T h = x - x_lo;

  // Get cubic coefficients
  const auto [b, c, d] = CSplineCoefficients<T>(x_(i0), x_(i1))(dx, dy);

  // Evaluate cubic polynomial using Horner's method: y_lo + h*(b + h*(c + h*d))
  return y_lo + h * std::fma(h, std::fma(h, d, c), b);
}

template <std::floating_point T>
constexpr auto CSplineBase<T>::derivative_(const Vector<T>& xa,
                                           const Vector<T>& ya, const T x) const
    -> T {
  const auto where = this->search(xa, x);
  if (!where) [[unlikely]] {
    return Fill<T>::value();
  }

  const auto [i0, i1] = *where;

  // Extract interval endpoints
  const T x_lo = xa(i0);
  const T x_hi = xa(i1);
  const T y_hi = ya(i1);
  const T y_lo = ya(i0);

  // Compute interval properties
  const T dx = x_hi - x_lo;
  const T dy = y_hi - y_lo;
  const T h = x - x_lo;

  // Get cubic coefficients
  const auto [b, c, d] = CSplineCoefficients<T>(x_(i0), x_(i1))(dx, dy);

  // Derivative: b + h*(2*c + h*3*d) = b + h*(2*c + 3*h*d)
  return b + h * std::fma(T{3} * h, d, T{2} * c);
}

}  // namespace pyinterp::math::interpolate::univariate

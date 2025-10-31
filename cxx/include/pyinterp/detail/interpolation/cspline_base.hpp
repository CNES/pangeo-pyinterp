// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Dense>

#include "pyinterp/detail/interpolation/interpolator_1d.hpp"

namespace pyinterp::detail::interpolation {

/// Coefficients of the cubic spline interpolation
template <typename T>
class CSplineCoefficients {
 public:
  /// Constructor
  /// @param c0 The first derivative at the first point
  /// @param c1 The first derivative at the last point
  constexpr CSplineCoefficients(const T &c0, const T &c1) : c0_(c0), c1_(c1) {};

  /// Compute the coefficients of the cubic spline interpolation
  /// @param dx The distance between the two points
  /// @param dy The difference between the two points
  /// @return The coefficients of the cubic spline interpolation
  constexpr auto operator()(const T &dx, const T &dy) const
      -> std::tuple<T, T, T> {
    constexpr auto third = T(1) / T(3);
    return {(dy / dx) - dx * (c1_ + 2 * c0_) * third, c0_,
            (c1_ - c0_) / (3 * dx)};
  }

 private:
  T c0_;
  T c1_;
};

/// Base class for cubic spline interpolation
template <typename T>
class CSplineBase : public Interpolator1D<T> {
 public:
  using Interpolator1D<T>::Interpolator1D;
  using Interpolator1D<T>::operator();
  using Interpolator1D<T>::derivative;

 private:
  /// Interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The point where the interpolation must be calculated.
  /// @return The interpolated value at the point x.
  constexpr auto interpolate_(const Vector<T> &xa, const Vector<T> &ya,
                              const T &x) const -> T override;

  /// @brief Returns the derivative of the interpolation function at the point
  ///   x.
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The point where the derivative must be calculated.
  /// @return The derivative of the interpolation function at the point x.
  constexpr auto derivative_(const Vector<T> &xa, const Vector<T> &ya,
                             const T &x) const -> T override;

 protected:
  Matrix<T> A_;
  Vector<T> b_;
  Vector<T> x_;
};

template <typename T>
constexpr auto CSplineBase<T>::interpolate_(const Vector<T> &xa,
                                            const Vector<T> &ya,
                                            const T &x) const -> T {
  auto where = this->search(xa, x);
  if (!where) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  auto x_lo = xa(where->first);
  auto y_lo = ya(where->first);
  auto x_hi = xa(where->second);
  auto y_hi = ya(where->second);
  auto dx = x_hi - x_lo;
  auto dy = y_hi - y_lo;
  auto h = x - x_lo;

  auto [b, c, d] =
      CSplineCoefficients<T>(x_(where->first), x_(where->second))(dx, dy);
  return y_lo + h * (b + h * (c + h * d));
}

template <typename T>
constexpr auto CSplineBase<T>::derivative_(const Vector<T> &xa,
                                           const Vector<T> &ya,
                                           const T &x) const -> T {
  auto where = this->search(xa, x);
  if (!where) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  auto x_lo = xa(where->first);
  auto y_lo = ya(where->first);
  auto x_hi = xa(where->second);
  auto y_hi = ya(where->second);
  auto dx = x_hi - x_lo;
  auto dy = y_hi - y_lo;
  auto h = x - x_lo;

  auto [b, c, d] =
      CSplineCoefficients<T>(x_(where->first), x_(where->second))(dx, dy);
  return b + h * (2 * c + h * 3 * d);
}

}  // namespace pyinterp::detail::interpolation

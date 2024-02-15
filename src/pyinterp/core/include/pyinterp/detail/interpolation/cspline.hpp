// Copyright (c) 2024 CNES
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
  constexpr CSplineCoefficients(const T &c0, const T &c1) : c0_(c0), c1_(c1){};

  /// Compute the coefficients of the cubic spline interpolation
  /// @param dx The distance between the two points
  /// @param dy The difference between the two points
  /// @return The coefficients of the cubic spline interpolation
  constexpr auto operator()(const T &dx, const T &dy) const
      -> std::tuple<T, T, T> {
    return {(dy / dx) - dx * (c1_ + 2 * c0_) / 3, c0_, (c1_ - c0_) / (3 * dx)};
  }

 private:
  T c0_;
  T c1_;
};

/// Cubic spline interpolation
template <typename T>
class CSpline : public Interpolator1D<T> {
 public:
  using Interpolator1D<T>::Interpolator1D;
  using Interpolator1D<T>::operator();
  using Interpolator1D<T>::derivative;

  /// Returns the minimum number of points required for the interpolation.
  auto min_size() const -> Eigen::Index override { return 3; }

 private:
  /// Interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The point where the interpolation must be calculated.
  /// @param i The index of the last point found in the search.
  /// @return The interpolated value at the point x.
  auto operator()(const Eigen::Ref<const Vector<T>> &xa,
                  const Eigen::Ref<const Vector<T>> &ya, const T &x,
                  Eigen::Index *i) const -> T override;

  /// @brief Returns the derivative of the interpolation function at the point
  ///   x.
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The point where the derivative must be calculated.
  /// @param i The index of the last point found in the search.
  /// @return The derivative of the interpolation function at the point x.
  auto derivative(const Eigen::Ref<const Vector<T>> &xa,
                  const Eigen::Ref<const Vector<T>> &ya, const T &x,
                  Eigen::Index *i) const -> T override;

 protected:
  /// @brief Compute the coefficients of the interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  auto compute_coefficients(const Eigen::Ref<const Vector<T>> &xa,
                            const Eigen::Ref<const Vector<T>> &ya)
      -> void override;

  Matrix<T> A_;
  Vector<T> b_;
  Vector<T> x_;
};

template <typename T>
auto CSpline<T>::compute_coefficients(const Eigen::Ref<const Vector<T>> &xa,
                                      const Eigen::Ref<const Vector<T>> &ya)
    -> void {
  Interpolator1D<T>::compute_coefficients(xa, ya);
  auto size = xa.size();
  if (x_.size() != size) {
    A_.resize(size - 2, size - 2);
    b_.resize(size - 2);
    x_.resize(size);
    A_.setZero();
  }

  for (auto i = 0; i < size - 2; i++) {
    const auto h_i0 = xa[i + 1] - xa[i];
    const auto h_i1 = xa[i + 2] - xa[i + 1];
    const auto y_i0 = ya[i + 1] - ya[i];
    const auto y_i1 = ya[i + 2] - ya[i + 1];
    const auto g_i0 = (h_i0 != 0) ? 1 / h_i0 : 0;
    const auto g_i1 = (h_i1 != 0) ? 1 / h_i1 : 0;
    if (i > 0) {
      A_(i, i - 1) = h_i0;
    }
    A_(i, i) = 2 * (h_i0 + h_i1);
    if (i < size - 3) {
      A_(i, i + 1) = h_i1;
    }
    b_(i) = 3 * (y_i1 * g_i1 - y_i0 * g_i0);
  }
  x_.segment(1, size - 2) = A_.fullPivLu().solve(b_);
  x_(0) = x_(size - 1) = 0;
}

template <typename T>
auto CSpline<T>::operator()(const Eigen::Ref<const Vector<T>> &xa,
                            const Eigen::Ref<const Vector<T>> &ya, const T &x,
                            Eigen::Index *i) const -> T {
  auto where = this->search(xa, x, i);
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
auto CSpline<T>::derivative(const Eigen::Ref<const Vector<T>> &xa,
                            const Eigen::Ref<const Vector<T>> &ya, const T &x,
                            Eigen::Index *i) const -> T {
  auto where = this->search(xa, x, i);
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

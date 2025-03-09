// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Dense>

#include "pyinterp/detail/interpolation/cspline_base.hpp"

namespace pyinterp::detail::interpolation {

/// Cubic spline interpolation
template <typename T>
class CSpline : public CSplineBase<T> {
 public:
  using CSplineBase<T>::CSplineBase;
  using CSplineBase<T>::operator();
  using CSplineBase<T>::derivative;

  /// Returns the minimum number of points required for the interpolation.
  auto min_size() const -> Eigen::Index override { return 4; }

 private:
  /// @brief Compute the coefficients of the interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  constexpr auto compute_coefficients(const Vector<T> &xa, const Vector<T> &ya)
      -> void override;

  /// @brief Solve a symmetric tridiagonal system
  /// @param x The solution of the system
  constexpr auto solve_symmetric_tridiagonal(T *x) -> void;

  Vector<T> c_;
  Vector<T> d_;
};

template <typename T>
constexpr auto CSpline<T>::solve_symmetric_tridiagonal(T *x) -> void {
  const auto size = this->A_.rows();
  const auto size_m1 = size - 1;
  const auto size_m2 = size - 2;

  auto item = 1 / this->A_(0, 0);
  c_(0) = this->A_(0, 1) * item;
  d_(0) = this->b_(0) * item;

  for (Eigen::Index i = 1; i < size_m1; ++i) {
    item = this->A_(i, i - 1);
    const auto inv_m = 1 / (this->A_(i, i) - item * c_(i - 1));
    c_(i) = this->A_(i, i + 1) * inv_m;
    d_(i) = (this->b_(i) - item * d_(i - 1)) * inv_m;
  }

  item = this->A_(size_m1, size_m2);
  d_(size_m1) = (this->b_(size_m1) - item * d_(size_m2)) /
                (this->A_(size_m1, size_m1) - item * c_(size_m2));
  x[size_m1] = d_(size_m1);

  for (Eigen::Index i = size_m2; i >= 0; --i) {
    x[i] = d_(i) - c_(i) * x[i + 1];
  }
}

template <typename T>
constexpr auto CSpline<T>::compute_coefficients(const Vector<T> &xa,
                                                const Vector<T> &ya) -> void {
  Interpolator1D<T>::compute_coefficients(xa, ya);
  const auto size = xa.size();
  const auto size_m2 = size - 2;
  if (this->x_.size() != size) {
    this->A_.resize(size_m2, size_m2);
    this->b_.resize(size_m2);
    this->x_.resize(size);
    this->A_.setZero();
    this->x_.setZero();
    c_.resize(size_m2);
    d_.resize(size_m2);
  }

  for (auto i = 0; i < size_m2; ++i) {
    const auto x1 = xa(i + 1);
    const auto y1 = ya(i + 1);
    const auto h_i0 = x1 - xa(i);
    const auto h_i1 = xa(i + 2) - x1;
    const auto y_i0 = y1 - ya(i);
    const auto y_i1 = ya(i + 2) - y1;
    const auto g_i0 = (h_i0 != 0) ? 1 / h_i0 : 0;
    const auto g_i1 = (h_i1 != 0) ? 1 / h_i1 : 0;
    if (i > 0) {
      this->A_(i, i - 1) = h_i0;
    }
    this->A_(i, i) = 2 * (h_i0 + h_i1);
    if (i < size - 3) {
      this->A_(i, i + 1) = h_i1;
    }
    this->b_(i) = 3 * (y_i1 * g_i1 - y_i0 * g_i0);
  }
  solve_symmetric_tridiagonal(this->x_.data() + 1);
}

}  // namespace pyinterp::detail::interpolation

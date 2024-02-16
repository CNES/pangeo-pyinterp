// Copyright (c) 2024 CNES
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
  auto min_size() const -> Eigen::Index override { return 3; }

 private:
  /// @brief Compute the coefficients of the interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  inline auto compute_coefficients(const Vector<T> &xa, const Vector<T> &ya)
      -> void override;

  /// @brief Solve a symmetric tridiagonal system
  /// @param x The solution of the system
  auto solve_symmetric_tridiagonal(T *x) -> void;

  Vector<T> c_;
  Vector<T> d_;
};

template <typename T>
auto CSpline<T>::solve_symmetric_tridiagonal(T *x) -> void {
  auto size = this->A_.rows();
  c_(0) = this->A_(0, 1) / this->A_(0, 0);
  d_(0) = this->b_(0) / this->A_(0, 0);

  for (Eigen::Index i = 1; i < size - 1; ++i) {
    const auto inv_m = 1 / (this->A_(i, i) - this->A_(i, i - 1) * c_(i - 1));
    c_(i) = this->A_(i, i + 1) * inv_m;
    d_(i) = (this->b_(i) - this->A_(i, i - 1) * d_(i - 1)) * inv_m;
  }

  d_(size - 1) =
      (this->b_(size - 1) - this->A_(size - 1, size - 2) * d_(size - 2)) /
      (this->A_(size - 1, size - 1) -
       this->A_(size - 1, size - 2) * c_(size - 2));
  x[size - 1] = d_(size - 1);

  for (Eigen::Index i = size - 2; i >= 0; --i) {
    x[i] = d_(i) - c_(i) * x[i + 1];
  }
}

template <typename T>
auto CSpline<T>::compute_coefficients(const Vector<T> &xa, const Vector<T> &ya)
    -> void {
  Interpolator1D<T>::compute_coefficients(xa, ya);
  auto size = xa.size();
  if (this->x_.size() != size) {
    this->A_.resize(size - 2, size - 2);
    this->b_.resize(size - 2);
    this->x_.resize(size);
    this->A_.setZero();
    this->x_.setZero();
    c_.resize(size - 2);
    d_.resize(size - 2);
  }

  for (auto i = 0; i < size - 2; ++i) {
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

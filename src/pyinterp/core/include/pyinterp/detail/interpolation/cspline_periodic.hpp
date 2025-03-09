// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Dense>

#include "pyinterp/detail/interpolation/cspline_base.hpp"

namespace pyinterp::detail::interpolation {

/// Periodic cubic spline interpolation
template <typename T>
class CSplinePeriodic : public CSplineBase<T> {
 public:
  using CSplineBase<T>::CSplineBase;

  /// Returns the minimum number of points required for the interpolation.
  auto min_size() const -> Eigen::Index override { return 2; }

 private:
  /// Compute the coefficients of the interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  constexpr auto compute_coefficients(const Vector<T> &xa, const Vector<T> &ya)
      -> void override;

  /// Solve a symmetric cyclic tridiagonal system
  /// @param x The solution of the system
  constexpr auto solve_symmetric_cyclic_tridiagonal(T *x) -> void;

  Vector<T> alpha_{};
  Vector<T> gamma_{};
  Vector<T> delta_{};
  Vector<T> c_{};
  Vector<T> z_{};
};

template <typename T>
constexpr auto CSplinePeriodic<T>::solve_symmetric_cyclic_tridiagonal(T *x)
    -> void {
  const auto size = this->A_.rows();
  const auto size_m1 = size - 1;
  const auto size_m2 = size - 2;
  const auto size_m3 = size - 3;
  if (size == 1) {
    x[0] = this->b_(0) / this->A_(0, 0);
    return;
  }

  alpha_(0) = this->A_(0, 0);
  gamma_(0) = this->A_(0, 1) / alpha_(0);
  delta_(0) = this->A_(size_m1, 0) / alpha_(0);

  for (Eigen::Index i = 1; i < size_m2; i++) {
    alpha_(i) = this->A_(i, i) - this->A_(i, i - 1) * gamma_(i - 1);
    gamma_(i) = this->A_(i, i + 1) / alpha_(i);
    delta_(i) = -delta_(i - 1) * this->A_(i, i - 1) / alpha_(i);
  }

  auto sum = alpha_.segment(0, size_m2)
                 .cwiseProduct(delta_.segment(0, size_m2))
                 .cwiseProduct(delta_.segment(0, size_m2))
                 .sum();

  alpha_(size_m2) =
      this->A_(size_m2, size_m2) - this->A_(size_m2, size_m3) * gamma_(size_m3);
  gamma_(size_m2) = (this->A_(size_m2, size_m1) -
                     this->A_(size_m2, size_m3) * delta_(size_m3)) /
                    alpha_(size_m2);
  alpha_(size_m1) = this->A_(size_m1, size_m1) - sum -
                    alpha_(size_m2) * gamma_(size_m2) * gamma_(size_m2);

  z_(0) = this->b_(0);
  for (Eigen::Index i = 1; i < size_m1; i++) {
    z_(i) = this->b_(i) - z_(i - 1) * gamma_(i - 1);
  }

  sum = delta_.segment(0, size_m2).cwiseProduct(z_.segment(0, size_m2)).sum();

  z_(size_m1) = this->b_(size_m1) - sum - gamma_(size_m2) * z_(size_m2);
  c_.segment(0, size) =
      z_.segment(0, size).array() / alpha_.segment(0, size).array();

  x[size_m1] = c_(size_m1);
  x[size_m2] = c_(size_m2) - gamma_(size_m2) * x[size_m1];
  if (size >= 3) {
    for (Eigen::Index i = size_m3, j = 0; j <= size_m3; j++, i--) {
      x[i] = c_(i) - gamma_(i) * x[i + 1] - delta_(i) * x[size_m1];
    }
  }
}

template <typename T>
constexpr auto CSplinePeriodic<T>::compute_coefficients(const Vector<T> &xa,
                                                        const Vector<T> &ya)
    -> void {
  Interpolator1D<T>::compute_coefficients(xa, ya);
  auto size = xa.size();
  if (this->x_.size() < size) {
    this->A_.resize(size - 1, size - 1);
    this->b_.resize(size - 1);
    this->x_.resize(size);
    this->A_.setZero();
    this->x_.setZero();
    alpha_.resize(size - 1);
    gamma_.resize(size - 1);
    delta_.resize(size - 1);
    c_.resize(size);
    z_.resize(size);
  }

  for (Eigen::Index i = 0; i < size - 2; i++) {
    const auto h_i0 = xa[i + 1] - xa[i];
    const auto h_i1 = xa[i + 2] - xa[i + 1];
    const auto y_i0 = ya[i + 1] - ya[i];
    const auto y_i1 = ya[i + 2] - ya[i + 1];
    const auto g_i0 = (h_i0 != 0) ? 1 / h_i0 : 0;
    const auto g_i1 = (h_i1 != 0) ? 1 / h_i1 : 0;
    this->A_(i + 1, i) = h_i1;
    this->A_(i, i) = 2 * (h_i0 + h_i1);
    this->A_(i, i + 1) = h_i1;
    this->b_(i) = 3 * (y_i1 * g_i1 - y_i0 * g_i0);
  }
  auto i = size - 2;
  const auto h_i0 = xa[i + 1] - xa[i];
  const auto h_i1 = xa[1] - xa[0];
  const auto y_i0 = ya[i + 1] - ya[i];
  const auto y_i1 = ya[1] - ya[0];
  const auto g_i0 = (h_i0 != 0) ? 1 / h_i0 : 0;
  const auto g_i1 = (h_i1 != 0) ? 1 / h_i1 : 0;

  this->A_(i, 0) = h_i1;
  this->A_(i, i) = 2 * (h_i0 + h_i1);
  this->A_(0, i) = h_i1;
  this->b_(i) = 3 * (y_i1 * g_i1 - y_i0 * g_i0);
  solve_symmetric_cyclic_tridiagonal(this->x_.data() + 1);
  this->x_(0) = this->x_(size - 1);
}

}  // namespace pyinterp::detail::interpolation

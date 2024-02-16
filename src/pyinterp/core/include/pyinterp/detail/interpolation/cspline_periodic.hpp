// Copyright (c) 2024 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Dense>

#include "pyinterp/detail/interpolation/cspline.hpp"

namespace pyinterp::detail::interpolation {

/// Periodic cubic spline interpolation
template <typename T>
class CSplinePeriodic : public CSpline<T> {
 public:
  using CSpline<T>::CSpline;

  /// Returns the minimum number of points required for the interpolation.
  auto min_size() const -> Eigen::Index override { return 2; }

 private:
  /// Compute the coefficients of the interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  auto compute_coefficients(const Eigen::Ref<const Vector<T>> &xa,
                            const Eigen::Ref<const Vector<T>> &ya)
      -> void override;
};

template <typename T>
auto CSplinePeriodic<T>::compute_coefficients(
    const Eigen::Ref<const Vector<T>> &xa,
    const Eigen::Ref<const Vector<T>> &ya) -> void {
  Interpolator1D<T>::compute_coefficients(xa, ya);
  auto size = xa.size();
  if (this->x_.size() < size) {
    this->A_.resize(size - 1, size - 1);
    this->b_.resize(size - 1);
    this->x_.resize(size);
    this->A_.setZero();
  }

  for (auto i = 0; i < size - 2; i++) {
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
  this->x_.segment(1, size - 1) =
      std::move(this->A_.fullPivLu().solve(this->b_));
  this->x_(0) = this->x_(size - 1);
}

}  // namespace pyinterp::detail::interpolation

// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Dense>

#include "pyinterp/detail/interpolation/cspline_base.hpp"

namespace pyinterp::detail::interpolation {

/// Cubic spline interpolation with not‑a‑knot end conditions.
template <typename T>
class CSplineNotAKnot : public CSplineBase<T> {
 public:
  using CSplineBase<T>::CSplineBase;
  using CSplineBase<T>::operator();

  /// Minimum number of data points required is 4.
  auto min_size() const -> Eigen::Index override { return 4; }

 protected:
  /// Compute the spline coefficients (i.e. the first derivatives at the data
  /// points) by solving an \f$n\times n\f$ system with not‑a‑knot boundary
  /// conditions.
  constexpr auto compute_coefficients(const Vector<T>& xa, const Vector<T>& ya)
      -> void override;
};

template <typename T>
constexpr auto CSplineNotAKnot<T>::compute_coefficients(const Vector<T>& xa,
                                                        const Vector<T>& ya)
    -> void {
  Interpolator1D<T>::compute_coefficients(xa, ya);
  const auto size = xa.size();
  const auto size_m1 = size - 1;
  const auto size_m2 = size - 2;

  // Resize the system: we now have n unknowns (one derivative per knot)
  this->A_.resize(size, size);
  this->b_.resize(size);
  this->x_.resize(size);
  this->A_.setZero();
  this->b_.setZero();
  this->x_.setZero();

  // Compute step sizes: h_i = x[i+1] - x[i] for i = 0 ... n-2.
  Vector<T> h(size_m1);
  for (int64_t ix = 0; ix < size_m1; ++ix) {
    h(ix) = xa(ix + 1) - xa(ix);
    if (h(ix) == T(0)) {
      throw std::invalid_argument("x-coordinates must be distinct");
    }
  }

  // Left not‑a‑knot condition at x0:
  this->A_(0, 0) = -h(1);
  this->A_(0, 1) = h(0) + h(1);
  this->A_(0, 2) = -h(0);
  this->b_(0) = T(0);

  // Interior equations: for i = 1,..., n-2.
  for (int64_t ix = 1; ix <= size_m2; ++ix) {
    this->A_(ix, ix - 1) = h(ix - 1);
    this->A_(ix, ix) = T(2) * (h(ix - 1) + h(ix));
    this->A_(ix, ix + 1) = h(ix);
    const T delta_i = (ya(ix + 1) - ya(ix)) / h(ix);
    const T delta_im1 = (ya(ix) - ya(ix - 1)) / h(ix - 1);
    this->b_(ix) = T(3) * (delta_i - delta_im1);
  }

  // Right not‑a‑knot condition at x[n-1]:
  this->A_(size_m1, size - 3) = h(size_m2);
  this->A_(size_m1, size_m2) = -(h(size_m2) + h(size - 3));
  this->A_(size_m1, size_m1) = h(size_m2);
  this->b_(size_m1) = T(0);

  // Solve the full system A * x_ = b_.
  this->x_ = this->A_.colPivHouseholderQr().solve(this->b_);
}

}  // namespace pyinterp::detail::interpolation

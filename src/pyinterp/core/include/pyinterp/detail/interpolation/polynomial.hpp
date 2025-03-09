// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include "pyinterp/detail/interpolation/interpolator_1d.hpp"

namespace pyinterp::detail::interpolation {

/// Polynomial interpolation
template <typename T>
class Polynomial : public Interpolator1D<T> {
 public:
  using Interpolator1D<T>::Interpolator1D;
  using Interpolator1D<T>::operator();
  using Interpolator1D<T>::derivative;

  /// The minimum size of the arrays to be interpolated.
  auto min_size() const -> Eigen::Index override { return 3; }

 private:
  /// Coefficients of the interpolation
  Vector<T> work_{};

  /// Compute the coefficients of the interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  constexpr auto compute_coefficients(const Vector<T> &xa, const Vector<T> &ya)
      -> void override;

  /// Compute the coefficients of the interpolation
  /// @param xa X-coordinates of the data points.
  /// @param x The point where the interpolation must be calculated.
  auto taylor(const Vector<T> &xa, const T &x) const -> Vector<T>;

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
  auto derivative_(const Vector<T> &xa, const Vector<T> &ya, const T &x) const
      -> T override;
};

template <typename T>
constexpr auto Polynomial<T>::compute_coefficients(const Vector<T> &xa,
                                                   const Vector<T> &ya)
    -> void {
  Interpolator1D<T>::compute_coefficients(xa, ya);
  auto size = xa.size();
  if (work_.size() < size) {
    work_.resize(size);
  }
  work_[0] = ya[0];
  work_.segment(1, size - 1) =
      (ya.segment(1, size - 1) - ya.segment(0, size - 1)).array() /
      (xa.segment(1, size - 1) - xa.segment(0, size - 1)).array();
  for (Eigen::Index ix = 2; ix < size; ix++) {
    work_.segment(ix, size - ix) =
        (work_.segment(ix, size - ix) - work_.segment(ix - 1, size - ix))
            .array() /
        (xa.segment(ix, size - ix) - xa.segment(ix - 1, size - ix)).array();
  }
}

template <typename T>
auto Polynomial<T>::taylor(const Vector<T> &xa, const T &x) const -> Vector<T> {
  auto size = xa.size();
  auto c = Vector<T>(size);
  auto w = Vector<T>(size);
  w(size - 1) = T(1);
  c(0) = work_(0);
  for (Eigen::Index ix = size - 1; ix--;) {
    w(ix) = -w(ix + 1) * (xa(size - 2 - ix) - x);
    for (Eigen::Index jx = ix + 1; jx < size - 1; ++jx) {
      w(jx) -= w(jx + 1) * (xa(size - 2 - ix) - x);
    }
    for (Eigen::Index jx = ix; jx < size; ++jx) {
      c(jx - ix) += w(jx) * work_(size - ix - 1);
    }
  }
  return c;
}

template <typename T>
constexpr auto Polynomial<T>::interpolate_(const Vector<T> &xa,
                                           const Vector<T> &ya,
                                           const T &x) const -> T {
  auto search = this->search(xa, x);
  if (!search) {
    throw std::numeric_limits<T>::quiet_NaN();
  }
  auto size = xa.size();
  auto y = work_[size - 1];
  for (Eigen::Index ix = size - 1; ix--;) {
    y = work_[ix] + (x - xa(ix)) * y;
  }
  return y;
}

template <typename T>
auto Polynomial<T>::derivative_(const Vector<T> &xa, const Vector<T> &ya,
                                const T &x) const -> T {
  auto search = this->search(xa, x);
  if (!search) {
    throw std::numeric_limits<T>::quiet_NaN();
  }
  auto coefficients = taylor(xa, x);
  return coefficients(1);
}

}  // namespace pyinterp::detail::interpolation

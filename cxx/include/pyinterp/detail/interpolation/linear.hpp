// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include "pyinterp/detail/interpolation/interpolator_1d.hpp"

namespace pyinterp::detail::interpolation {

/// Linear interpolation
template <typename T>
class Linear : public Interpolator1D<T> {
 public:
  using Interpolator1D<T>::Interpolator1D;
  using Interpolator1D<T>::operator();
  using Interpolator1D<T>::derivative;

  /// Returns the minimum number of points required for the interpolation.
  auto min_size() const -> Eigen::Index override { return 2; }

 private:
  /// Interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The point where the interpolation must be calculated.
  /// @return The interpolated value at the point x.
  constexpr auto interpolate_(const Vector<T> &xa, const Vector<T> &ya,
                              const T &x) const -> T override {
    auto where = this->search(xa, x);
    if (!where) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    auto [i0, i1] = *where;
    return ya(i0) + (ya(i1) - ya(i0)) / (xa(i1) - xa(i0)) * (x - xa(i0));
  }

  /// @brief Returns the derivative of the interpolation function at the point
  ///   x.
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The point where the derivative must be calculated.
  /// @return The derivative of the interpolation function at the point x.
  constexpr auto derivative_(const Vector<T> &xa, const Vector<T> &ya,
                             const T &x) const -> T override {
    auto where = this->search(xa, x);
    if (!where) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    auto [i0, i1] = *where;
    return (ya(i1) - ya(i0)) / (xa(i1) - xa(i0));
  }
};

}  // namespace pyinterp::detail::interpolation

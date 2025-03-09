// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include "pyinterp/detail/interpolation/interpolator_2d.hpp"

namespace pyinterp::detail::interpolation {

/// Bilinear interpolation
/// @tparam T type of the data
template <typename T>
class Bilinear : public Interpolator2D<T> {
 public:
  using Interpolator2D<T>::Interpolator2D;
  using Interpolator2D<T>::operator();

  /// Returns the minimum number of points required for the interpolation.
  auto min_size() const -> Eigen::Index override { return 2; }

 private:
  /// Interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param za Z-values of the data points.
  /// @param x The point where the interpolation must be calculated.
  /// @param y The point where the interpolation must be calculated.
  constexpr auto interpolate_(const Vector<T> &xa, const Vector<T> &ya,
                              const Matrix<T> &za, const T &x, const T &y) const
      -> T override;
};

template <typename T>
constexpr auto Bilinear<T>::interpolate_(const Vector<T> &xa,
                                         const Vector<T> &ya,
                                         const Matrix<T> &za, const T &x,
                                         const T &y) const -> T {
  auto search_x = this->search(xa, x);
  auto search_y = this->search(ya, y);
  if (!search_x || !search_y) {
    throw std::numeric_limits<T>::quiet_NaN();
  }
  auto [i0, i1] = *search_x;
  auto [j0, j1] = *search_y;
  auto x0 = xa[i0];
  auto x1 = xa[i1];
  auto y0 = ya[j0];
  auto y1 = ya[j1];
  auto dx = x1 - x0;
  auto dy = y1 - y0;
  auto t = (x - x0) / dx;
  auto u = (y - y0) / dy;
  return (T(1) - t) * (T(1) - u) * za(i0, j0) + t * (T(1) - u) * za(i1, j0) +
         (T(1) - t) * u * za(i0, j1) + t * u * za(i1, j1);
}

}  // namespace pyinterp::detail::interpolation

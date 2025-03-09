// Copyright (c) 2025 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include "pyinterp/detail/interpolation/cspline.hpp"
#include "pyinterp/detail/interpolation/interpolator_2d.hpp"

namespace pyinterp::detail::interpolation {

/// Bicubic interpolation
/// @tparam T type of the data
template <typename T>
class Bicubic : public Interpolator2D<T> {
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
  /// @return The interpolated value at the coordinates x, y.
  constexpr auto interpolate_(const Vector<T> &xa, const Vector<T> &ya,
                              const Matrix<T> &za, const T &x, const T &y) const
      -> T override;

  /// Compute the coefficients of the bicubic interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param za Z-values of the data points.
  constexpr auto compute_coefficients(const Vector<T> &xa, const Vector<T> &ya,
                                      const Matrix<T> &za) -> void override;

  Matrix<T> zx_{};
  Matrix<T> zy_{};
  Matrix<T> zxy_{};
  CSpline<T> spline_{};
};

template <typename T>
constexpr auto Bicubic<T>::compute_coefficients(const Vector<T> &xa,
                                                const Vector<T> &ya,
                                                const Matrix<T> &za) -> void {
  Interpolator2D<T>::compute_coefficients(xa, ya, za);
  auto xsize = xa.size();
  auto ysize = ya.size();

  if (zx_.rows() != xsize || zx_.cols() != ysize) {
    zx_ = Matrix<T>(xsize, ysize);
    zy_ = Matrix<T>(xsize, ysize);
    zxy_ = Matrix<T>(xsize, ysize);
  }

  for (Eigen::Index j = 0; j < ysize; ++j) {
    zx_.col(j) = spline_.derivative(xa, za.col(j), xa);
  }
  for (Eigen::Index i = 0; i < xsize; ++i) {
    zy_.row(i) = spline_.derivative(ya, za.row(i), ya);
  }
  for (Eigen::Index j = 0; j < ysize; ++j) {
    zxy_.col(j) = spline_.derivative(xa, zy_.col(j), xa);
  }
}

template <typename T>
constexpr auto Bicubic<T>::interpolate_(const Vector<T> &xa,
                                        const Vector<T> &ya,
                                        const Matrix<T> &za, const T &x,
                                        const T &y) const -> T {
  auto search_x = this->search(xa, x);
  auto search_y = this->search(ya, y);
  if (!search_x || !search_y) {
    throw std::numeric_limits<T>::quiet_NaN();
  }
  const auto [i0, i1] = *search_x;
  const auto [j0, j1] = *search_y;
  const auto x0 = xa(i0);
  const auto x1 = xa(i1);
  const auto y0 = ya(j0);
  const auto y1 = ya(j1);
  const auto z00 = za(i0, j0);
  const auto z01 = za(i0, j1);
  const auto z10 = za(i1, j0);
  const auto z11 = za(i1, j1);
  const auto dx = x1 - x0;
  const auto dy = y1 - y0;
  const auto dxdy = dx * dy;
  const auto t = (x - x0) / dx;
  const auto u = (y - y0) / dy;
  const auto zx00 = zx_(i0, j0) * dx;
  const auto zx01 = zx_(i0, j1) * dx;
  const auto zx10 = zx_(i1, j0) * dx;
  const auto zx11 = zx_(i1, j1) * dx;
  const auto zy00 = zy_(i0, j0) * dy;
  const auto zy01 = zy_(i0, j1) * dy;
  const auto zy10 = zy_(i1, j0) * dy;
  const auto zy11 = zy_(i1, j1) * dy;
  const auto zxy00 = zxy_(i0, j0) * dxdy;
  const auto zxy01 = zxy_(i0, j1) * dxdy;
  const auto zxy10 = zxy_(i1, j0) * dxdy;
  const auto zxy11 = zxy_(i1, j1) * dxdy;
  const auto t2 = t * t;
  const auto t3 = t * t2;
  const auto u2 = u * u;
  const auto u3 = u * u2;

  return (z00 + u * zy00 + u2 * (3 * (z01 - z00) - 2 * zy00 - zy01) +
          u3 * (2 * (z00 - z01) + zy00 + zy01)) +
         t * (zx00 + u * zxy00 + u2 * (3 * (zx01 - zx00) - 2 * zxy00 - zxy01) +
              u3 * (2 * (zx00 - zx01) + zxy00 + zxy01)) +
         t2 * ((3 * (z10 - z00) - 2 * zx00 - zx10) +
               u * (3 * (zy10 - zy00) - 2 * zxy00 - zxy10) +
               u2 * (9 * (z00 - z01 - z10 + z11) +
                     6 * (zx00 - zx01 + zy00 - zy10) +
                     3 * (zx10 - zx11 + zy01 - zy11) + 4 * zxy00 +
                     2 * (zxy01 + zxy10) + zxy11) +
               u3 * (6 * (z01 - z00 + z10 - z11) + 4 * (zx01 - zx00) +
                     3 * (zy10 - zy00 - zy01 + zy11) +
                     2 * (zx11 - zx10 - zxy00 - zxy01) - zxy10 - zxy11)) +
         t3 * ((2 * (z00 - z10) + zx00 + zx10) +
               u * (zxy00 + zxy10 + 2 * (zy00 - zy10)) +
               u2 * (6 * (z01 - z00 + z10 - z11) + 4 * (-zy00 + zy10) +
                     3 * (zx01 - zx00 - zx10 + zx11) +
                     2 * (zy11 - zy01 - zxy00 - zxy10) - zxy01 - zxy11) +
               u3 * (4 * (z00 - z01 - z10 + z11) +
                     2 * (zx00 - zx01 + zx10 - zx11 + zy00 + zy01 - zy10 -
                          zy11) +
                     zxy00 + zxy01 + zxy10 + zxy11));
}

}  // namespace pyinterp::detail::interpolation

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <ranges>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/fill.hpp"
#include "pyinterp/math/interpolate/bivariate.hpp"
#include "pyinterp/math/interpolate/univariate/cspline.hpp"

namespace pyinterp::math::interpolate::bivariate {

/// @brief Bicubic interpolation - smooth interpolation using cubic polynomials
/// in both dimensions with continuous first derivatives
/// @tparam T type of the data (must be floating point)
template <std::floating_point T>
class Bicubic : public Bivariate<T> {
 public:
  using Bivariate<T>::Bivariate;
  using Bivariate<T>::operator();

  /// Returns the minimum number of points required for the interpolation.
  [[nodiscard]] constexpr auto min_size() const -> int64_t final { return 2; }

 private:
  /// @brief Interpolation using bicubic polynomials
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] za Z-values of the data points (2D grid).
  /// @param[in] x The x-coordinate where the interpolation must be calculated.
  /// @param[in] y The y-coordinate where the interpolation must be calculated.
  /// @return The interpolated value at point (x, y).
  [[nodiscard]] constexpr auto interpolate_(
      const Eigen::Ref<const Vector<T>>& xa,
      const Eigen::Ref<const Vector<T>>& ya,
      const Eigen::Ref<const Matrix<T>>& za, const T& x, const T& y) const
      -> T final;

  /// @brief Compute the coefficients of the bicubic interpolation
  /// @param[in] xa X-coordinates of the data points.
  /// @param[in] ya Y-coordinates of the data points.
  /// @param[in] za Z-values of the data points (2D grid).
  /// @return True if coefficients computed successfully
  [[nodiscard]] constexpr auto compute_coefficients(
      const Eigen::Ref<const Vector<T>>& xa,
      const Eigen::Ref<const Vector<T>>& ya,
      const Eigen::Ref<const Matrix<T>>& za) -> bool final;

  /// Partial derivatives needed for bicubic interpolation
  Matrix<T> zx_{};   // ∂z/∂x at grid points
  Matrix<T> zy_{};   // ∂z/∂y at grid points
  Matrix<T> zxy_{};  // ∂²z/∂x∂y at grid points

  /// Cubic spline for computing derivatives
  univariate::CSpline<T> spline_{};
};

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
constexpr auto Bicubic<T>::compute_coefficients(
    const Eigen::Ref<const Vector<T>>& xa,
    const Eigen::Ref<const Vector<T>>& ya,
    const Eigen::Ref<const Matrix<T>>& za) -> bool {
  if (!Bivariate<T>::compute_coefficients(xa, ya, za)) [[unlikely]] {
    return false;
  }

  const auto xsize = xa.size();
  const auto ysize = ya.size();

  // Resize derivative matrices if needed
  if (zx_.rows() != xsize || zx_.cols() != ysize) {
    zx_.resize(xsize, ysize);
    zy_.resize(xsize, ysize);
    zxy_.resize(xsize, ysize);
  }

  // Compute ∂z/∂x at each grid point (derivative along x for each y)
  for (const auto j : std::views::iota(int64_t{0}, ysize)) {
    zx_.col(j) = spline_.derivative(xa, za.col(j), xa);
  }

  // Compute ∂z/∂y at each grid point (derivative along y for each x)
  for (const auto i : std::views::iota(int64_t{0}, xsize)) {
    zy_.row(i) = spline_.derivative(ya, za.row(i), ya);
  }

  // Compute ∂²z/∂x∂y (cross derivatives)
  for (const auto j : std::views::iota(int64_t{0}, ysize)) {
    zxy_.col(j) = spline_.derivative(xa, zy_.col(j), xa);
  }

  return true;
}

template <std::floating_point T>
constexpr auto Bicubic<T>::interpolate_(const Eigen::Ref<const Vector<T>>& xa,
                                        const Eigen::Ref<const Vector<T>>& ya,
                                        const Eigen::Ref<const Matrix<T>>& za,
                                        const T& x, const T& y) const -> T {
  const auto search_x = this->search(xa, x);
  const auto search_y = this->search(ya, y);

  if (!search_x || !search_y) [[unlikely]] {
    return Fill<T>::value();
  }

  const auto [i0, i1] = *search_x;
  const auto [j0, j1] = *search_y;

  // Grid cell corners and dimensions
  const T x0 = xa(i0);
  const T x1 = xa(i1);
  const T y0 = ya(j0);
  const T y1 = ya(j1);
  const T dx = x1 - x0;
  const T dy = y1 - y0;
  const T dxdy = dx * dy;

  // Normalized coordinates [0, 1]
  const T t = (x - x0) / dx;
  const T u = (y - y0) / dy;

  // Function values at corners
  const T z00 = za(i0, j0);
  const T z01 = za(i0, j1);
  const T z10 = za(i1, j0);
  const T z11 = za(i1, j1);

  // Scaled x-derivatives at corners
  const T zx00 = zx_(i0, j0) * dx;
  const T zx01 = zx_(i0, j1) * dx;
  const T zx10 = zx_(i1, j0) * dx;
  const T zx11 = zx_(i1, j1) * dx;

  // Scaled y-derivatives at corners
  const T zy00 = zy_(i0, j0) * dy;
  const T zy01 = zy_(i0, j1) * dy;
  const T zy10 = zy_(i1, j0) * dy;
  const T zy11 = zy_(i1, j1) * dy;

  // Scaled cross-derivatives at corners
  const T zxy00 = zxy_(i0, j0) * dxdy;
  const T zxy01 = zxy_(i0, j1) * dxdy;
  const T zxy10 = zxy_(i1, j0) * dxdy;
  const T zxy11 = zxy_(i1, j1) * dxdy;

  // Powers of normalized coordinates
  const T t2 = t * t;
  const T t3 = t2 * t;
  const T u2 = u * u;
  const T u3 = u2 * u;

  // Bicubic polynomial evaluation
  // Organized as: f(t,u) = Σ(i=0..3) Σ(j=0..3) c_ij * t^i * u^j

  // Terms with t^0 (constant in t)
  const T _t0 = z00 + u * zy00 +
                u2 * std::fma(T{3}, z01 - z00, -T{2} * zy00 - zy01) +
                u3 * (T{2} * (z00 - z01) + zy00 + zy01);

  // Terms with t^1
  const T _t1 = zx00 + u * zxy00 +
                u2 * std::fma(T{3}, zx01 - zx00, -T{2} * zxy00 - zxy01) +
                u3 * (T{2} * (zx00 - zx01) + zxy00 + zxy01);

  // Terms with t^2
  const T _t2_u0 = std::fma(T{3}, z10 - z00, -T{2} * zx00 - zx10);
  const T _t2_u1 = std::fma(T{3}, zy10 - zy00, -T{2} * zxy00 - zxy10);
  const T _t2_u2 = T{9} * (z00 - z01 - z10 + z11) +
                   T{6} * (zx00 - zx01 + zy00 - zy10) +
                   T{3} * (zx10 - zx11 + zy01 - zy11) + T{4} * zxy00 +
                   T{2} * (zxy01 + zxy10) + zxy11;
  const T _t2_u3 = T{6} * (z01 - z00 + z10 - z11) + T{4} * (zx01 - zx00) +
                   T{3} * (zy10 - zy00 - zy01 + zy11) +
                   T{2} * (zx11 - zx10 - zxy00 - zxy01) - zxy10 - zxy11;
  const T _t2 = _t2_u0 + u * _t2_u1 + u2 * _t2_u2 + u3 * _t2_u3;

  // Terms with t^3
  const T _t3_u0 = T{2} * (z00 - z10) + zx00 + zx10;
  const T _t3_u1 = zxy00 + zxy10 + T{2} * (zy00 - zy10);
  const T _t3_u2 = T{6} * (z01 - z00 + z10 - z11) + T{4} * (-zy00 + zy10) +
                   T{3} * (zx01 - zx00 - zx10 + zx11) +
                   T{2} * (zy11 - zy01 - zxy00 - zxy10) - zxy01 - zxy11;
  const T _t3_u3 =
      T{4} * (z00 - z01 - z10 + z11) +
      T{2} * (zx00 - zx01 + zx10 - zx11 + zy00 + zy01 - zy10 - zy11) + zxy00 +
      zxy01 + zxy10 + zxy11;
  const T _t3 = _t3_u0 + u * _t3_u1 + u2 * _t3_u2 + u3 * _t3_u3;

  // Combine all terms
  return _t0 + t * _t1 + t2 * _t2 + t3 * _t3;
}

}  // namespace pyinterp::math::interpolate::bivariate

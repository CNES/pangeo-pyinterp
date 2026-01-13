// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <concepts>
#include <cstdint>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/fill.hpp"
#include "pyinterp/math/interpolate/bivariate.hpp"

namespace pyinterp::math::interpolate::bivariate {

/// @brief Bilinear interpolation
/// Linear interpolation extended to two dimensions using weighted averages of
/// the four surrounding grid points
template <std::floating_point T>
class Bilinear : public Bivariate<T> {
 public:
  using Bivariate<T>::Bivariate;
  using Bivariate<T>::operator();

  /// Returns the minimum number of points required for the interpolation.
  [[nodiscard]] constexpr auto min_size() const -> int64_t final { return 2; }

 private:
  /// Interpolation using bilinear weighting
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param za Z-values of the data points (2D grid).
  /// @param x The x-coordinate where the interpolation must be calculated.
  /// @param y The y-coordinate where the interpolation must be calculated.
  /// @return The interpolated value at point (x, y).
  [[nodiscard]] constexpr auto interpolate_(
      const Eigen::Ref<const Vector<T>>& xa,
      const Eigen::Ref<const Vector<T>>& ya,
      const Eigen::Ref<const Matrix<T>>& za, const T& x, const T& y) const
      -> T final;
};

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
constexpr auto Bilinear<T>::interpolate_(const Eigen::Ref<const Vector<T>>& xa,
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

  // Grid cell corners
  const T x0 = xa(i0);
  const T x1 = xa(i1);
  const T y0 = ya(j0);
  const T y1 = ya(j1);

  // Grid cell dimensions
  const T dx = x1 - x0;
  const T dy = y1 - y0;

  // Normalized coordinates within the cell [0, 1]
  const T t = (x - x0) / dx;
  const T u = (y - y0) / dy;
  // Complementary weights
  const T t_comp = T{1} - t;
  const T u_comp = T{1} - u;

  // Bilinear interpolation using the four corner values
  // z = (1-t)(1-u)*z00 + t(1-u)*z10 + (1-t)u*z01 + tu*z11
  const T z00 = za(i0, j0);
  const T z10 = za(i1, j0);
  const T z01 = za(i0, j1);
  const T z11 = za(i1, j1);

  // Compute weighted sum using FMA for better accuracy
  const T interp_bottom = std::fma(t, z10, t_comp * z00);
  const T interp_top = std::fma(t, z11, t_comp * z01);

  return std::fma(u, interp_top, u_comp * interp_bottom);
}

}  // namespace pyinterp::math::interpolate::bivariate

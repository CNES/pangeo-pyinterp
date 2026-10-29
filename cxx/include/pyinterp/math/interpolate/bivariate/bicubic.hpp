// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <utility>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/fill.hpp"
#include "pyinterp/math/interpolate/bivariate.hpp"
#include "pyinterp/math/interpolate/univariate/cspline.hpp"

namespace pyinterp::math::interpolate::bivariate {

/// @brief Bicubic interpolation - smooth interpolation using cubic polynomials
/// in both dimensions with continuous first derivatives
///
/// The partial derivatives are computed with natural cubic splines fitted along
/// the grid lines. Only the derivatives at the corners of the cell containing
/// the interpolated point are needed: they are computed on demand and kept
/// until a point falls in another cell, or new data are prepared. With the
/// windowed interpolators, the grid is a small window around the point, so
/// that only one cell is ever used.
///
/// @tparam T type of the data (must be floating point)
template <std::floating_point T>
class Bicubic : public Bivariate<T> {
 public:
  using Bivariate<T>::Bivariate;
  using Bivariate<T>::operator();

  /// Returns the minimum number of points required for the interpolation.
  [[nodiscard]] constexpr auto min_size() const -> int64_t final { return 2; }

  /// @brief Invalidate the derivatives computed for the previous data
  /// @return True
  [[nodiscard]] constexpr auto compute_coefficients() -> bool final {
    cell_ = {-1, -1};
    return true;
  }

 private:
  /// @brief Interpolation using bicubic polynomials
  /// @param[in] x The x-coordinate where the interpolation must be calculated.
  /// @param[in] y The y-coordinate where the interpolation must be calculated.
  /// @return The interpolated value at point (x, y).
  [[nodiscard]] constexpr auto interpolate_(const T& x, const T& y) -> T final;

  /// @brief Compute the partial derivatives at the corners of a cell
  /// @param[in] i0 Index of the lower x-coordinate of the cell
  /// @param[in] i1 Index of the upper x-coordinate of the cell
  /// @param[in] j0 Index of the lower y-coordinate of the cell
  /// @param[in] j1 Index of the upper y-coordinate of the cell
  constexpr auto compute_corner_derivatives(int64_t i0, int64_t i1, int64_t j0,
                                            int64_t j1) -> void;

  /// Cell (i0, j0) whose corner derivatives are stored, (-1, -1) if none
  std::pair<int64_t, int64_t> cell_{-1, -1};

  /// Partial derivatives at the corners of the cell, indexed by
  /// [x index (0: i0, 1: i1)][y index (0: j0, 1: j1)]
  std::array<std::array<T, 2>, 2> zx_{};   // ∂z/∂x
  std::array<std::array<T, 2>, 2> zy_{};   // ∂z/∂y
  std::array<std::array<T, 2>, 2> zxy_{};  // ∂²z/∂x∂y

  /// ∂z/∂y along x at the columns j0 and j1 of the cell
  std::array<Vector<T>, 2> zy_columns_{};

  /// Copy of a grid row, contiguous as the splines expect
  Vector<T> row_{};

  /// Cubic splines along x and y. Two instances, so that windows of different
  /// sizes along x and y do not resize the spline buffers on every call.
  univariate::CSpline<T> spline_x_{};
  univariate::CSpline<T> spline_y_{};
};

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
constexpr auto Bicubic<T>::compute_corner_derivatives(const int64_t i0,
                                                      const int64_t i1,
                                                      const int64_t j0,
                                                      const int64_t j1)
    -> void {
  const auto& xa = this->xa();
  const auto& ya = this->ya();
  const auto& za = this->za();
  const auto xa_size = xa.size();
  const std::array<T, 2> x_corners{xa(i0), xa(i1)};
  const std::array<T, 2> y_corners{ya(j0), ya(j1)};
  const std::array<int64_t, 2> columns{j0, j1};
  auto derivatives = std::array<T, 2>{};

  // ∂z/∂y along each row of the grid, kept at the columns of the cell: the
  // cross derivatives are the derivatives of these values along x.
  row_.resize(ya.size());
  for (auto& item : zy_columns_) {
    item.resize(xa_size);
  }
  for (const auto i : std::views::iota(int64_t{0}, xa_size)) {
    row_ = za.row(i).transpose();
    spline_y_.derivative(ya, row_, y_corners, derivatives);
    zy_columns_[0](i) = derivatives[0];
    zy_columns_[1](i) = derivatives[1];
  }

  for (const auto jx : std::views::iota(size_t{0}, size_t{2})) {
    // ∂z/∂x along the column of the corner
    spline_x_.derivative(xa, za.col(columns[jx]), x_corners, derivatives);
    zx_[0][jx] = derivatives[0];
    zx_[1][jx] = derivatives[1];

    // ∂²z/∂x∂y: derivative along x of ∂z/∂y
    spline_x_.derivative(xa, zy_columns_[jx], x_corners, derivatives);
    zxy_[0][jx] = derivatives[0];
    zxy_[1][jx] = derivatives[1];
  }

  zy_[0][0] = zy_columns_[0](i0);
  zy_[0][1] = zy_columns_[1](i0);
  zy_[1][0] = zy_columns_[0](i1);
  zy_[1][1] = zy_columns_[1](i1);
}

template <std::floating_point T>
constexpr auto Bicubic<T>::interpolate_(const T& x, const T& y) -> T {
  const auto& xa = this->xa();
  const auto& ya = this->ya();
  const auto& za = this->za();
  const auto search_x = this->search(xa, x);
  const auto search_y = this->search(ya, y);

  if (!search_x || !search_y) [[unlikely]] {
    return Fill<T>::value();
  }

  const auto [i0, i1] = *search_x;
  const auto [j0, j1] = *search_y;

  if (cell_.first != i0 || cell_.second != j0) {
    compute_corner_derivatives(i0, i1, j0, j1);
    cell_ = {i0, j0};
  }

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
  const T zx00 = zx_[0][0] * dx;
  const T zx01 = zx_[0][1] * dx;
  const T zx10 = zx_[1][0] * dx;
  const T zx11 = zx_[1][1] * dx;

  // Scaled y-derivatives at corners
  const T zy00 = zy_[0][0] * dy;
  const T zy01 = zy_[0][1] * dy;
  const T zy10 = zy_[1][0] * dy;
  const T zy11 = zy_[1][1] * dy;

  // Scaled cross-derivatives at corners
  const T zxy00 = zxy_[0][0] * dxdy;
  const T zxy01 = zxy_[0][1] * dxdy;
  const T zxy10 = zxy_[1][0] * dxdy;
  const T zxy11 = zxy_[1][1] * dxdy;

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

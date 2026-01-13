// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>

#include "pyinterp/eigen.hpp"
#include "pyinterp/math/fill.hpp"
#include "pyinterp/math/interpolate/univariate.hpp"

namespace pyinterp::math::interpolate::univariate {

/// Linear interpolation
/// @tparam T Type of the data to interpolate
template <std::floating_point T>
class Linear : public Univariate<T> {
 public:
  using Univariate<T>::Univariate;
  using Univariate<T>::operator();
  using Univariate<T>::derivative;

  /// Returns the minimum number of points required for the interpolation.
  [[nodiscard]] constexpr auto min_size() const -> int64_t final { return 2; }

 private:
  /// Interpolation
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The point where the interpolation must be calculated.
  /// @return The interpolated value at the point x.
  [[nodiscard]] constexpr auto interpolate_(const Vector<T>& xa,
                                            const Vector<T>& ya, T x) const
      -> T final {
    const auto where = this->search(xa, x);
    if (!where) [[unlikely]] {
      return Fill<T>::value();
    }

    const auto [i0, i1] = *where;
    const T dx = xa(i1) - xa(i0);
    const T dy = ya(i1) - ya(i0);
    const T offset = x - xa(i0);

    return std::fma(dy / dx, offset, ya(i0));
  }

  /// @brief Returns the derivative of the interpolation function at the point
  /// x.
  /// @param xa X-coordinates of the data points.
  /// @param ya Y-coordinates of the data points.
  /// @param x The point where the derivative must be calculated.
  /// @return The derivative of the interpolation function at the point x.
  [[nodiscard]] constexpr auto derivative_(const Vector<T>& xa,
                                           const Vector<T>& ya, T x) const
      -> T final {
    const auto where = this->search(xa, x);
    if (!where) [[unlikely]] {
      return Fill<T>::value();
    }

    const auto [i0, i1] = *where;
    return (ya(i1) - ya(i0)) / (xa(i1) - xa(i0));
  }
};

}  // namespace pyinterp::math::interpolate::univariate

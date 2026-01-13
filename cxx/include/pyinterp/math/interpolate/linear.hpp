// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include "pyinterp/math/fill.hpp"

namespace pyinterp::math::interpolate {

/// @brief Linear interpolation
///
/// @tparam Axis Type of the coordinates
/// @tparam T Type of the point values
/// @param[in] x x_coordinate
/// @param[in] x0 x0 coordinate
/// @param[in] x1 x1 coordinate
/// @param[in] y0 Point value for the coordinate (x0)
/// @param[in] y1 Point value for the coordinate (x1)
/// @return Interpolated value at coordinate x
template <typename Axis, typename T = Axis>
constexpr auto linear(const Axis& x, const Axis& x0, const Axis& x1,
                      const T& y0, const T& y1) noexcept -> T {
  const auto dx = static_cast<T>(x1 - x0);

  if (dx == T{0}) {
    return (x == x0) ? y0 : Fill<T>::value();
  }

  const auto t = static_cast<T>(x - x0) / dx;

  return y0 + t * (y1 - y0);
}

}  // namespace pyinterp::math::interpolate

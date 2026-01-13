// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

namespace pyinterp::math::interpolate {

/// @brief Nearest neighbor interpolation
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
constexpr auto nearest(const Axis &x, const Axis &x0, const Axis &x1,
                       const T &y0, const T &y1) noexcept -> T {
  // Compare absolute differences to avoid overflow from squaring
  // and to correctly compare distances.
  const auto dx0 = x0 - x;
  const auto dx1 = x1 - x;

  const auto abs_dx0 = dx0 >= 0 ? dx0 : -dx0;
  const auto abs_dx1 = dx1 >= 0 ? dx1 : -dx1;

  return abs_dx0 < abs_dx1 ? y0 : y1;
}

}  // namespace pyinterp::math::interpolate

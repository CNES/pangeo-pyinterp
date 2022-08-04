// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <limits>
#include <type_traits>

namespace pyinterp::detail::math {

/// Linear interpolation
///
/// @param x x_coordinate
/// @param x0 x0 coordinate
/// @param x1 x1 coordinate
/// @param y0 Point value for the coordinate (x0)
/// @param y1 Point value for the coordinate (x1)
template <typename T, typename U = T,
          typename std::enable_if<std::is_floating_point<U>::value, U>::type * =
              nullptr>
constexpr auto linear(const T &x, const T &x0, const T &x1, const U &y0,
                      const U &y1) -> U {
  auto dx = static_cast<U>(x1 - x0);
  if (dx == 0) {
    return x == x0 ? y0 : std::numeric_limits<U>::quiet_NaN();
  }

  auto t = static_cast<U>(x1 - x) / dx;
  auto u = static_cast<U>(x - x0) / dx;

  return t * y0 + u * y1 / (t + u);
}

}  // namespace pyinterp::detail::math

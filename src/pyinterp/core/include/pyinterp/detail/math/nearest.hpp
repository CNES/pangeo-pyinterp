// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include "pyinterp/detail/math.hpp"

namespace pyinterp::detail::math {

/// Nearest interpolation
///
/// @param x x_coordinate
/// @param x0 x0 coordinate
/// @param x1 x1 coordinate
/// @param y0 Point value for the coordinate (x0)
/// @param y1 Point value for the coordinate (x1)
template <typename T, typename U = T>
constexpr auto nearest(const T &x, const T &x0, const T &x1, const U &y0,
                       const U &y1) noexcept -> U {
  // we use a comparable distance, there's no need to calculate the square root
  // here
  auto dx0 = sqr(x0 - x);
  auto dx1 = sqr(x1 - x);

  return dx0 < dx1 ? y0 : y1;
}

}  // namespace pyinterp::detail::math

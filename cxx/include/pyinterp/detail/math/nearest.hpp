// Copyright (c) 2025 CNES
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
  // Compare absolute differences to avoid overflow from squaring
  // and to correctly compare distances.
  auto dx0 = x0 - x;
  auto dx1 = x1 - x;

  if (std::abs(dx0) < std::abs(dx1)) {
    return y0;
  } else if (std::abs(dx0) > std::abs(dx1)) {
    return y1;
  } else {
    // If both distances are equal, return the last value
    // to maintain consistency with the original logic.
    return y1;
  }
}

}  // namespace pyinterp::detail::math

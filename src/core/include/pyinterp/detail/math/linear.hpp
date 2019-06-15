#pragma once

namespace pyinterp {
namespace detail {
namespace math {

/// Linear interpolation
///
/// @param x x_coordinate
/// @param x0 x0 coordinate
/// @param x1 x1 coordinate
/// @param y0 Point value for the coordinate (x0)
/// @param y1 Point value for the coordinate (x1)
template <typename T>
inline constexpr T linear(const T& x, const T& x0, const T& x1, const T& y0,
                          const T& y1) {
  auto dx = (x1 - x0);
  auto t = (x1 - x) / dx;
  auto u = (x - x0) / dx;

  return t * y0 + u * y1 / (t + u);
}

}  // namespace math
}  // namespace detail
}  // namespace pyinterp
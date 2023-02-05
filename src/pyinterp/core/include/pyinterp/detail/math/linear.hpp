// Copyright (c) 2023 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <limits>
#include <type_traits>

#include "pyinterp/eigen.hpp"

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

/// @brief Compute the linear interpolation of a point.
/// @tparam T Type of the coordinates.
/// @param coordinates Coordinates of the points: matrix of size (n, m) where n
/// is the number of dimensions and m the number of points.
/// @param values Values of the points.
/// @param query Coordinates of the point to interpolate.
/// @return The interpolated value.
/// @warning The number of dimensions of the coordinates must be equal to the
/// number of dimensions of the query and the number of values must be equal to
/// the number of points. No check is performed on the input parameters.
template <typename T>
auto linear(const Matrix<T> &coordinates, const Vector<T> &values,
            const Vector<T> &query) -> T {
  std::vector<std::pair<T, T>> distances;
  distances.reserve(coordinates.cols());
  for (auto ix = 0; ix < coordinates.cols(); ++ix) {
    auto distance = (coordinates.col(ix) - query).norm();
    distances.emplace_back(distance, values[ix]);
  }
  std::sort(distances.begin(), distances.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  auto sum = T(0);
  auto weight_sum = T(0);
  for (auto &&item : distances) {
    sum += item.second / item.first;
    weight_sum += 1 / item.first;
  }
  return sum / weight_sum;
}

}  // namespace pyinterp::detail::math

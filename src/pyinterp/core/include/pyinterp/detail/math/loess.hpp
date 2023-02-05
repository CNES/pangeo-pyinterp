// Copyright (c) 2023 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Dense>
#include <vector>

#include "pyinterp/detail/math.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp::detail::math {

template <typename T>
constexpr auto gaussian_weight(const T& dx, const T& dy, const T& dz,
                               const T& h) -> T {
  return std::exp(-(dx * dx + dy * dy + dz * dz) / (2 * h * h));
}

template <typename T>
auto weighted_least_squares(const Eigen::Matrix<T, 3, -1>& coordinates,
                            const Eigen::Matrix<T, -1, 1>& values,
                            const Eigen::Vector3<T>& query, const T& h) -> T {
  auto k = coordinates.cols();
  Vector<T> b(k);
  Matrix<T> A(k, 4);
  auto wsum = T(0);
  for (int i = 0; i < k; i++) {
    auto w = gaussian_weight(coordinates(0, i) - query[0],
                             coordinates(1, i) - query[1],
                             coordinates(2, i) - query[2], h);
    wsum += w;
    A(i, 0) = w;
    A(i, 1) = w * coordinates(0, i);
    A(i, 2) = w * coordinates(1, i);
    A(i, 3) = w * coordinates(2, i);
    b[i] = w * values[i];
  }
  A /= wsum;
  b /= wsum;
  Eigen::Matrix<T, 4, 1> result =
      (A.transpose() * A).inverse() * A.transpose() * b;
  return result[0] + result[1] * query[0] + result[2] * query[1] +
         result[3] * query[2];
}

template <typename T>
auto loess(const Eigen::Matrix<T, 3, -1>& points,
           const Eigen::Matrix<T, -1, 1>& values,
           const Eigen::Vector3<T>& query, const T& h) -> T {
  return weighted_least_squares(points, values, query, h);
}

}  // namespace pyinterp::detail::math

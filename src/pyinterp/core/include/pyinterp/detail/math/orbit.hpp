// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <cmath>
#include <tuple>

#include "pyinterp/detail/math.hpp"

namespace pyinterp::detail::math {

/// Calculate the rotation matrix for a rotation of angle θ around the axis
template <typename T>
inline auto rotation_3d_matrix(const T& theta,
                               const Eigen::Vector3<T>& axis) noexcept
    -> Eigen::Matrix<T, 3, 3> {
  auto coefs = -axis.normalized() * std::sin(theta * static_cast<T>(0.5));

  auto result = Eigen::Matrix<T, 3, 3>();
  const auto a = std::cos(theta * 0.5);
  const auto b = coefs[0];
  const auto c = coefs[1];
  const auto d = coefs[2];
  const auto a2 = a * a;
  const auto b2 = b * b;
  const auto c2 = c * c;
  const auto d2 = d * d;
  const auto bc = b * c;
  const auto ad = a * d;
  const auto bd = b * d;
  const auto ac = a * c;
  const auto cd = c * d;
  const auto ab = a * b;
  result << a2 + b2 - c2 - d2, 2 * (bc - ad), 2 * (bd + ac), 2 * (bc + ad),
      a2 - b2 + c2 - d2, 2 * (cd - ab), 2 * (bd - ac), 2 * (cd + ab),
      a2 - b2 - c2 + d2;
  return result;
}

/// Convert cartesian coordinates to spherical coordinates
template <typename T>
inline auto cartesian2spherical(const T& x, const T& y, const T& z)
    -> std::tuple<T, T> {
  if (is_almost_zero(x, std::numeric_limits<T>::epsilon()) &&
      is_almost_zero(y, std::numeric_limits<T>::epsilon())) {
    return std::make_tuple(0, static_cast<T>(90) * (std::signbit(z) ? -1 : 1));
  }
  auto lat = std::atan2(z, std::sqrt(x * x + y * y));
  auto lon = std::atan2(y, x);
  return std::make_tuple(degrees<T>(lon), degrees<T>(lat));
}

}  // namespace pyinterp::detail::math

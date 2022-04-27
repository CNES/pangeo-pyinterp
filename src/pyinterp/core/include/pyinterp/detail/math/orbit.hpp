// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <limits>
#include <tuple>

#include "pyinterp/detail/math.hpp"
#include "pyinterp/eigen.hpp"

namespace pyinterp::detail::math {

template <typename T>
auto satellite_direction(const Eigen::Matrix<T, Eigen::Dynamic, 3>& location)
    -> Eigen::Matrix<T, Eigen::Dynamic, 3> {
  auto result = Eigen::Matrix<T, Eigen::Dynamic, 3>(location.rows(), 3);
  auto denominator = location.block(1, 0, location.rows() - 2, 3)
                         .array()
                         .pow(2)
                         .rowwise()
                         .sum()
                         .sqrt();
  for (auto ix = 0; ix < 3; ++ix) {
    result.block(1, ix, location.rows() - 2, 1) =
        (location.block(2, ix, location.rows() - 2, 1) -
         location.block(0, ix, location.rows() - 2, 1))
            .array() /
        denominator.array();
  }

  result.row(0) = result.row(1);
  result.row(result.rows() - 1) = result.row(result.rows() - 2);
  return result;
}

template <typename T>
auto spherical2cartesian(const Vector<T>& lon, const Vector<T>& lat)
    -> Eigen::Matrix<T, Eigen::Dynamic, 3> {
  auto result = Eigen::Matrix<T, Eigen::Dynamic, 3>(lon.rows(), 3);
  for (auto ix = 0; ix < lon.rows(); ++ix) {
    auto rlon = radians(lon(ix));
    auto rlat = radians(lat(ix));
    result(ix, 0) = std::cos(rlat) * std::cos(rlon);
    result(ix, 1) = std::cos(rlat) * std::sin(rlon);
    result(ix, 2) = std::sin(rlat);
  }
  return result;
}

/// Calculate the rotation matrix for a rotation of angle θ around the axis
template <typename T>
inline auto rotation_3d_matrix(const T& theta,
                               const Eigen::Matrix<T, 3, 1>& axis) noexcept
    -> Eigen::Matrix<T, 3, 3> {
  auto angle_axis = Eigen::AngleAxis<T>(theta, -axis.normalized());
  return angle_axis.toRotationMatrix();
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

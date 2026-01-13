// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <concepts>

namespace pyinterp::geometry::satellite::rotation {

/// @brief Calculate the direction of the satellite movement
/// @tparam T Numeric type
/// @param[in] location Satellite locations
/// @return Direction of movement vectors (normalized)
template <std::floating_point T>
auto satellite_direction(const Eigen::Matrix<T, Eigen::Dynamic, 3>& location)
    -> Eigen::Matrix<T, Eigen::Dynamic, 3> {
  auto result = Eigen::Matrix<T, Eigen::Dynamic, 3>(location.rows(), 3);

  // Compute the difference vectors: location[i+1] - location[i-1]
  auto diff = location.block(2, 0, location.rows() - 2, 3) -
              location.block(0, 0, location.rows() - 2, 3);

  // Compute the norm of each difference vector
  auto denominator = diff.array().pow(2).rowwise().sum().sqrt();

  // Normalize each component of the difference vectors
  for (auto ix = 0; ix < 3; ++ix) {
    result.block(1, ix, location.rows() - 2, 1) =
        diff.col(ix).array() / denominator.array();
  }

  // Copy boundary values
  result.row(0) = result.row(1);
  result.row(result.rows() - 1) = result.row(result.rows() - 2);
  return result;
}

/// @brief Create a 3D rotation matrix
/// @tparam T Numeric type
/// @param[in] theta Rotation angle (radian)
/// @param[in] axis Rotation axis
/// @return 3D rotation matrix
template <std::floating_point T>
inline auto matrix_3d(const T& theta,
                      const Eigen::Matrix<T, 3, 1>& axis) noexcept
    -> Eigen::Matrix<T, 3, 3> {
  auto angle_axis = Eigen::AngleAxis<T>(theta, -axis.normalized());
  return angle_axis.toRotationMatrix();
}

}  // namespace pyinterp::geometry::satellite::rotation

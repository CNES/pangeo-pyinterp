// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Core>
#include <boost/geometry.hpp>

#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/geographic/algorithms/distance.hpp"
#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/geographic/multi_point.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"

namespace pyinterp::geometry::geographic {

/// @brief Calculate distance from each point to target using compile-time
/// strategy
/// @tparam SourceGeometry Source geometry type (MultiPoint, LineString, Ring)
/// @tparam TargetGeometry Target geometry type
/// @tparam Method Strategy method to use
/// @param[in] source Source geometry containing points
/// @param[in] target Target geometry to measure distance to
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @return Vector of distances for each point in meters
template <typename SourceGeometry, typename TargetGeometry,
          StrategyMethod Method>
[[nodiscard]] inline auto for_each_point_distance(
    const SourceGeometry& source, const TargetGeometry& target,
    const std::optional<Spheroid>& spheroid) -> Eigen::VectorXd {
  Eigen::VectorXd result(source.size());
  auto strategy = make_distance_strategy<Method>(make_spheroid(spheroid));
  for (std::size_t i = 0; i < source.size(); ++i) {
    result(i) = boost::geometry::distance(source[i], target, strategy);
  }
  return result;
}

/// @brief Calculate distance from each point to target using runtime strategy
/// @tparam SourceGeometry Source geometry type (MultiPoint, LineString, Ring)
/// @tparam TargetGeometry Target geometry type
/// @param[in] source Source geometry containing points
/// @param[in] target Target geometry to measure distance to
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @param[in] strategy Strategy method to use
/// @return Vector of distances for each point in meters
template <typename SourceGeometry, typename TargetGeometry>
[[nodiscard]] inline auto for_each_point_distance(
    const SourceGeometry& source, const TargetGeometry& target,
    const std::optional<Spheroid>& spheroid, const StrategyMethod strategy)
    -> Eigen::VectorXd {
  using enum StrategyMethod;
  switch (strategy) {
    case kAndoyer:
      return for_each_point_distance<SourceGeometry, TargetGeometry, kAndoyer>(
          source, target, spheroid);
    case kKarney:
      return for_each_point_distance<SourceGeometry, TargetGeometry, kKarney>(
          source, target, spheroid);
    case kThomas:
      return for_each_point_distance<SourceGeometry, TargetGeometry, kThomas>(
          source, target, spheroid);
    case kVincenty:
      return for_each_point_distance<SourceGeometry, TargetGeometry, kVincenty>(
          source, target, spheroid);
  }
  std::unreachable();
}

}  // namespace pyinterp::geometry::geographic

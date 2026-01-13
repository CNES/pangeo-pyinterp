// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <Eigen/Core>
#include <boost/geometry.hpp>

#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"

namespace pyinterp::geometry::geographic {

/// @brief Create distance strategy based on the specified method
/// and spheroid
/// @tparam Method Strategy method to use
/// @param[in] spheroid Spheroid for geodetic calculations
/// @return distance strategy
template <StrategyMethod Method>
[[nodiscard]] constexpr auto make_distance_strategy(
    const boost::geometry::srs::spheroid<double> &spheroid) {
  if constexpr (Method == StrategyMethod::kAndoyer) {
    return boost::geometry::strategy::distance::geographic<
        boost::geometry::strategy::andoyer>(spheroid);
  } else if constexpr (Method == StrategyMethod::kKarney) {
    return boost::geometry::strategy::distance::geographic<
        boost::geometry::strategy::karney>(spheroid);
  } else if constexpr (Method == StrategyMethod::kThomas) {
    return boost::geometry::strategy::distance::geographic<
        boost::geometry::strategy::thomas>(spheroid);
  } else if constexpr (Method == StrategyMethod::kVincenty) {
    return boost::geometry::strategy::distance::geographic<
        boost::geometry::strategy::vincenty>(spheroid);
  } else {
    static_assert(false, "Unhandled StrategyMethod");
  }
}

/// @brief Calculate the distance using a compile-time strategy
/// @tparam Geometry Geometry type
/// @tparam Method Strategy method to use
/// @param[in] geometry1 First geometric object
/// @param[in] geometry2 Second geometric object
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @return Calculated distance
template <typename Geometry1, typename Geometry2, StrategyMethod Method>
[[nodiscard]] inline auto distance(const Geometry1 &geometry1,
                                   const Geometry2 &geometry2,
                                   const std::optional<Spheroid> &spheroid)
    -> double {
  return boost::geometry::distance(
      geometry1, geometry2,
      make_distance_strategy<Method>(make_spheroid(spheroid)));
}

/// @brief Calculate the distance using a runtime strategy
/// @tparam Geometry Geometry type
/// @param[in] geometry1 First geometric object
/// @param[in] geometry2 Second geometric object
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @param[in] strategy Strategy method to use
/// @return Calculated distance
template <typename Geometry1, typename Geometry2>
[[nodiscard]] inline auto distance(const Geometry1 &geometry1,
                                   const Geometry2 &geometry2,
                                   const std::optional<Spheroid> &spheroid,
                                   const StrategyMethod strategy) -> double {
  using enum StrategyMethod;
  switch (strategy) {
    case kAndoyer:
      return distance<Geometry1, Geometry2, kAndoyer>(geometry1, geometry2,
                                                      spheroid);
    case kKarney:
      return distance<Geometry1, Geometry2, kKarney>(geometry1, geometry2,
                                                     spheroid);
    case kThomas:
      return distance<Geometry1, Geometry2, kThomas>(geometry1, geometry2,
                                                     spheroid);
    case kVincenty:
      return distance<Geometry1, Geometry2, kVincenty>(geometry1, geometry2,
                                                       spheroid);
  }
  std::unreachable();
}

/// @brief Calculate the curvilinear distance along the geometry
/// @tparam Geometry Geometry type
/// @tparam Method Strategy method to use
/// @param[in] geometry Geometric object
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @return Calculated distance
template <typename Geometry, StrategyMethod Method>
[[nodiscard]] inline auto curvilinear_distance(
    const Geometry &geometry, const std::optional<Spheroid> &spheroid)
    -> Eigen::VectorXd {
  Eigen::VectorXd distances(geometry.size());
  if (geometry.size() == 0) {
    return distances;
  }
  auto strategy = make_distance_strategy<Method>(make_spheroid(spheroid));
  auto acc = 0.0;
  distances(0) = acc;
  for (std::size_t i = 1; i < geometry.size(); ++i) {
    acc += boost::geometry::distance(geometry[i - 1], geometry[i], strategy);
    distances(i) = acc;
  }
  return distances;
}

/// @brief Calculate the curvilinear distance along the geometry
/// @tparam Geometry Geometry type
/// @param[in] geometry Geometric object
/// @param[in] spheroid Optional Spheroid for geodetic calculations
/// @param[in] strategy Strategy method to use
/// @return Calculated distance
template <typename Geometry>
[[nodiscard]] inline auto curvilinear_distance(
    const Geometry &geometry, const std::optional<Spheroid> &spheroid,
    const StrategyMethod strategy) -> Eigen::VectorXd {
  using enum StrategyMethod;
  switch (strategy) {
    case kAndoyer:
      return curvilinear_distance<Geometry, kAndoyer>(geometry, spheroid);
    case kKarney:
      return curvilinear_distance<Geometry, kKarney>(geometry, spheroid);
    case kThomas:
      return curvilinear_distance<Geometry, kThomas>(geometry, spheroid);
    case kVincenty:
      return curvilinear_distance<Geometry, kVincenty>(geometry, spheroid);
  }
  std::unreachable();
}

}  // namespace pyinterp::geometry::geographic

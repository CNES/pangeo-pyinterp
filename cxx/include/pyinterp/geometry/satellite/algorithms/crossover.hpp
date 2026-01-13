// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Core>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#include "pyinterp/broadcast.hpp"
#include "pyinterp/geometry/cartesian/crossover.hpp"
#include "pyinterp/geometry/cartesian/linestring.hpp"
#include "pyinterp/geometry/cartesian/point.hpp"
#include "pyinterp/geometry/geographic/algorithms/distance.hpp"
#include "pyinterp/geometry/geographic/crossover.hpp"
#include "pyinterp/geometry/geographic/linestring.hpp"
#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"
#include "pyinterp/math.hpp"

namespace pyinterp::geometry::satellite::algorithms {

/// @brief Result of a crossover detection
struct CrossoverResult {
  geographic::Point point;  ///< The crossover point
  size_t index1;            ///< Index of nearest vertex in first linestring
  size_t index2;            ///< Index of nearest vertex in second linestring
};

namespace detail {

/// @brief Check that coordinates of the half-orbits have same shape.
/// @param[in] lon1 Longitude array of the first half-orbit
/// @param[in] lat1 Latitude array of the first half-orbit
/// @param[in] lon2 Longitude array of the second half-orbit
/// @param[in] lat2 Latitude array of the second half-orbit
/// @param[in] predicate Maximum acceptable distance
/// @throws std::invalid_argument if the shapes are not compatible
inline auto check_linestring_shapes(
    const Eigen::Ref<const Eigen::VectorXd>& lon1,
    const Eigen::Ref<const Eigen::VectorXd>& lat1,
    const Eigen::Ref<const Eigen::VectorXd>& lon2,
    const Eigen::Ref<const Eigen::VectorXd>& lat2, const double predicate)
    -> void {
  broadcast::check_eigen_shape("lon1", lon1, "lat1", lat1);
  broadcast::check_eigen_shape("lon2", lon2, "lat2", lat2);
  if (lon1.size() < 3) {
    throw std::invalid_argument(
        "The first linestring must have at least 3 points.");
  }
  if (lon2.size() < 3) {
    throw std::invalid_argument(
        "The second linestring must have at least 3 points.");
  }
  if (predicate <= 0.0 || !std::isfinite(predicate)) {
    throw std::invalid_argument(
        "The predicate distance must be a positive finite value.");
  }
}

/// @brief Check if a crossover point is acceptable based on the distance to
/// the nearest vertices in both linestrings.
/// @param[in] xover Crossover handler
/// @param[in] point Crossover point found
/// @param[in] predicate Maximum acceptable distance to consider a vertex as
/// nearest
/// @param[in] strategy Calculation strategy
/// @param[in] spheroid Optional spheroid for geodetic calculations
/// @return The crossover point if acceptable; std::nullopt otherwise
template <typename CrossoverType, typename PointType>
inline auto filter_crossover(
    const CrossoverType& xover, PointType point, const double predicate,
    const geographic::StrategyMethod strategy,
    const std::optional<geographic::Spheroid>& spheroid)
    -> std::optional<CrossoverResult> {
  auto [ix1, ix2] = xover.nearest(point);

  geographic::Point geographic_point;
  if constexpr (std::is_same_v<CrossoverType, geographic::Crossover>) {
    geographic_point = std::move(point);
  } else {
    // Convert the Cartesian crossover point to geographic coordinates in order
    // to compute distances to the nearest vertices.
    geographic_point = geographic::Point(
        math::normalize_period(point.template get<0>(), -180.0, 360.0),
        point.template get<1>());
  }

  const auto& line1 = xover.line1();
  const auto& line2 = xover.line2();

  auto calculate_distance = [&](size_t ix, const auto& line) -> double {
    return geographic::distance(geographic_point,
                                geographic::Point(line[ix].template get<0>(),
                                                  line[ix].template get<1>()),
                                spheroid, strategy);
  };

  if (calculate_distance(ix1, line1) > predicate) {
    return std::nullopt;
  }
  if (calculate_distance(ix2, line2) > predicate) {
    return std::nullopt;
  }
  return CrossoverResult{
      .point = std::move(geographic_point), .index1 = ix1, .index2 = ix2};
}

/// @brief Find crossovers using geographic calculations.
inline auto find_crossovers_geographic(
    const Eigen::Ref<const Eigen::VectorXd>& lon1,
    const Eigen::Ref<const Eigen::VectorXd>& lat1,
    const Eigen::Ref<const Eigen::VectorXd>& lon2,
    const Eigen::Ref<const Eigen::VectorXd>& lat2, const double predicate,
    const bool allow_multiple, const geographic::StrategyMethod strategy,
    const std::optional<geographic::Spheroid>& spheroid)
    -> std::vector<CrossoverResult> {
  auto xover = geographic::Crossover(geographic::LineString(lon1, lat1),
                                     geographic::LineString(lon2, lat2));
  std::vector<CrossoverResult> result;
  if (allow_multiple) {
    for (auto point : xover.find_all(spheroid, strategy)) {
      if (auto filtered = filter_crossover(xover, std::move(point), predicate,
                                           strategy, spheroid)) {
        result.push_back(std::move(*filtered));
      }
    }
  } else {
    if (auto point = xover.find_unique(spheroid, strategy)) {
      if (auto filtered = filter_crossover(xover, std::move(*point), predicate,
                                           strategy, spheroid)) {
        result.push_back(std::move(*filtered));
      }
    }
  }
  return result;
}

/// @brief Find crossovers using Cartesian calculations.
inline auto find_crossovers_cartesian(
    const Eigen::Ref<const Eigen::VectorXd>& lon1,
    const Eigen::Ref<const Eigen::VectorXd>& lat1,
    const Eigen::Ref<const Eigen::VectorXd>& lon2,
    const Eigen::Ref<const Eigen::VectorXd>& lat2, const double predicate,
    const bool allow_multiple, const geographic::StrategyMethod strategy,
    const std::optional<geographic::Spheroid>& spheroid)
    -> std::vector<CrossoverResult> {
  auto xover = cartesian::Crossover(cartesian::LineString(lon1, lat1),
                                    cartesian::LineString(lon2, lat2));
  std::vector<CrossoverResult> result;
  if (allow_multiple) {
    for (auto point : xover.find_all()) {
      if (auto filtered = filter_crossover(xover, std::move(point), predicate,
                                           strategy, spheroid)) {
        result.push_back(std::move(*filtered));
      }
    }
  } else {
    if (auto point = xover.find_unique()) {
      if (auto filtered = filter_crossover(xover, std::move(*point), predicate,
                                           strategy, spheroid)) {
        result.push_back(std::move(*filtered));
      }
    }
  }
  return result;
}

}  // namespace detail

/// @brief Finds crossover points between two satellite half-orbits.
/// @param[in] lon1 Longitude array of the first half-orbit
/// @param[in] lat1 Latitude array of the first half-orbit
/// @param[in] lon2 Longitude array of the second half-orbit
/// @param[in] lat2 Latitude array of the second half-orbit
/// @param[in] predicate Maximum acceptable distance to consider a vertex as
/// nearest
/// @param[in] allow_multiple If true, find all crossover points; otherwise,
/// expect exactly one crossover point
/// @param[in] use_cartesian If true, use Cartesian calculations for faster
/// results; otherwise, use geographic calculations
/// @param[in] strategy Calculation strategy
/// @param[in] spheroid Optional spheroid for geodetic calculations
/// @return All crossover points found that pass the predicate filter
/// @throws std::runtime_error if allow_multiple is false and multiple crossover
/// points are found
/// @note If use_cartesian is true, the intersection search is performed on a
/// Cartesian plane, which provides faster results than geodetic calculations.
/// However, this approach is only appropriate when an approximate determination
/// is acceptable. The accuracy depends on the vertices of the linestrings
/// being close to each other; if they are widely spaced, the determined
/// geographical point may be significantly incorrect due to Cartesian
/// approximation errors.
inline auto find_crossovers(const Eigen::Ref<const Eigen::VectorXd>& lon1,
                            const Eigen::Ref<const Eigen::VectorXd>& lat1,
                            const Eigen::Ref<const Eigen::VectorXd>& lon2,
                            const Eigen::Ref<const Eigen::VectorXd>& lat2,
                            const double predicate, const bool allow_multiple,
                            const bool use_cartesian,
                            const geographic::StrategyMethod strategy,
                            const std::optional<geographic::Spheroid>& spheroid)
    -> std::vector<CrossoverResult> {
  detail::check_linestring_shapes(lon1, lat1, lon2, lat2, predicate);
  return use_cartesian
             ? detail::find_crossovers_cartesian(lon1, lat1, lon2, lat2,
                                                 predicate, allow_multiple,
                                                 strategy, spheroid)
             : detail::find_crossovers_geographic(lon1, lat1, lon2, lat2,
                                                  predicate, allow_multiple,
                                                  strategy, spheroid);
}

}  // namespace pyinterp::geometry::satellite::algorithms

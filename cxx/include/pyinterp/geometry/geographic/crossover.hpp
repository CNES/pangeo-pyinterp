// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <boost/geometry.hpp>
#include <unordered_set>
#include <utility>

#include "pyinterp/geometry/crossover.hpp"
#include "pyinterp/geometry/geographic/algorithms/intersection.hpp"
#include "pyinterp/geometry/geographic/algorithms/strategy.hpp"
#include "pyinterp/geometry/geographic/linestring.hpp"
#include "pyinterp/geometry/geographic/multi_point.hpp"
#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"

namespace pyinterp::geometry::geographic {

/// @brief Detects crossover points between two geographic linestrings and
/// retrieves their properties.
class Crossover : public geometry::Crossover<Point> {
 public:
  /// @brief Constructs a crossover object from two linestrings
  /// @param[in] line1 First linestring
  /// @param[in] line2 Second linestring
  Crossover(LineString line1, LineString line2)
      : geometry::Crossover<Point>(std::move(line1), std::move(line2)) {}

  /// @brief Constructs a crossover object from a base Crossover object
  /// @param[in] base Base Crossover object
  explicit Crossover(geometry::Crossover<Point> base)
      : geometry::Crossover<Point>(std::move(base)) {}

  /// @brief Find a unique crossover point between the two linestrings
  /// @param[in] spheroid Optional spheroid for geodetic calculations
  /// @param[in] strategy Calculation strategy
  /// @return The unique crossover point if found; std::nullopt otherwise
  /// @throws std::runtime_error if multiple crossover points are found
  [[nodiscard]] auto find_unique(const std::optional<Spheroid>& spheroid,
                                 const StrategyMethod strategy) const
      -> std::optional<Point> {
    auto points = intersection_point(line1_, line2_, spheroid, strategy);
    if (points.empty()) {
      return std::nullopt;
    }
    if (points.size() == 1) {
      return points.front();
    }
    // If there is a merged point between lines #1 and #2 then the method will
    // find this point for each of the segments tested.
    auto set = std::unordered_set<Point>{};
    for (const auto& point : points) {
      set.insert(point);
    }
    if (set.size() == 1) {
      return *set.begin();
    }
    throw std::runtime_error(
        "Multiple crossover points found between the two lines.");
  }

  /// @brief Find all crossover points between the two linestrings
  /// @param[in] spheroid Optional spheroid for geodetic calculations
  /// @param[in] strategy Calculation strategy
  /// @return All crossover points found as a MultiPoint object
  [[nodiscard]] auto find_all(const std::optional<Spheroid>& spheroid,
                              const StrategyMethod strategy) const
      -> MultiPoint {
    auto points = intersection_point(line1_, line2_, spheroid, strategy);
    return MultiPoint(std::move(points));
  }
};

}  // namespace pyinterp::geometry::geographic

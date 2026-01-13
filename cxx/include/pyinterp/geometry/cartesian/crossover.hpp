// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <algorithm>
#include <boost/geometry.hpp>
#include <unordered_set>
#include <utility>

#include "pyinterp/geometry/cartesian/linestring.hpp"
#include "pyinterp/geometry/cartesian/multi_point.hpp"
#include "pyinterp/geometry/cartesian/point.hpp"
#include "pyinterp/geometry/crossover.hpp"
#include "pyinterp/math.hpp"

namespace pyinterp::geometry::cartesian {

/// @brief Detects crossover points between two geographic linestrings and
/// retrieves their properties.
class Crossover : public geometry::Crossover<Point> {
 public:
  /// @brief Constructs a crossover object from two linestrings
  /// @param[in] line1 First linestring
  /// @param[in] line2 Second linestring
  Crossover(LineString line1, LineString line2)
      : geometry::Crossover<Point>(std::move(line1), std::move(line2)) {
    normalize_line_strings(line1_, line2_);
  }

  /// @brief Constructs a crossover object from a base Crossover object
  /// @param[in] base Base Crossover object
  explicit Crossover(geometry::Crossover<Point> base)
      : geometry::Crossover<Point>(std::move(base)) {
    normalize_line_strings(line1_, line2_);
  }

  /// @brief Find a unique crossover point between the two linestrings
  /// @return The unique crossover point if found; std::nullopt otherwise
  /// @throws std::runtime_error if multiple crossover points are found
  [[nodiscard]] auto find_unique() const -> std::optional<Point> {
    auto points = Crossover::intersection_point(line1_, line2_);
    if (points.empty()) {
      return std::nullopt;
    }
    if (points.size() == 1) {
      return points.front();
    }
    // If there is a merged point between lines #1 and #2 then the method will
    // find this point for each of the segments tested.
    auto set = std::unordered_set<Point>{};
    for (const auto &point : points) {
      set.insert(point);
    }
    if (set.size() == 1) {
      return *set.begin();
    }
    throw std::runtime_error(
        "Multiple crossover points found between the two lines.");
  }

  /// @brief Find all crossover points between the two linestrings
  /// @return All crossover points found as a MultiPoint object
  [[nodiscard]] auto find_all() const -> MultiPoint {
    auto points = Crossover::intersection_point(line1(), line2());
    return MultiPoint(std::move(points));
  }

 private:
  /// @brief Computes the intersection points between two linestrings.
  /// @param[in] geometry1 First linestring
  /// @param[in] geometry2 Second linestring
  /// @return A vector of intersection points
  [[nodiscard]] static inline auto intersection_point(
      const LineString &geometry1, const LineString &geometry2)
      -> std::vector<Point> {
    std::vector<Point> result;
    boost::geometry::intersection(geometry1, geometry2, result);
    return result;
  }

  /// @brief Checks if the sequence of longitudes indicates a retrograde motion.
  /// @param[in] x1 Longitude of the first point
  /// @param[in] x2 Longitude of the second point
  /// @param[in] x3 Longitude of the third point
  /// @return True if the motion is retrograde; false otherwise
  __CONSTEXPR auto is_retrograde(const double x1, const double x2,
                                 const double x3) -> bool {
    constexpr auto k180 = 180.0;
    auto delta = std::min(std::max(x1 - x2, -k180), k180);
    if (std::abs(delta) == k180) {
      delta = std::min(std::max(x2 - x3, -k180), k180);
    }
    return delta > 0;
  }

  /// @brief Shift longitude values in the line string by adding 360 degrees
  /// from points with longitude less than the crossing point.
  /// @param[in] line The linestring to modify
  /// @param[in] crossing The crossing longitude value
  constexpr auto shift_plus_360(LineString &line, const double crossing)
      -> void {
    auto shift = [crossing](Point &item) {
      const auto lon = item.get<0>();
      if (lon < crossing) {
        item.set<0>(lon + 360);
      }
    };
    std::ranges::for_each(line, shift);
  }

  /// @brief Shift longitude values in the line string by subtracting 360
  /// degrees from points with longitude greater than the crossing point.
  /// @param[in] line The linestring to modify
  /// @param[in] crossing The crossing longitude value
  constexpr auto shift_minus_360(LineString &line, const double crossing)
      -> void {
    auto shift = [crossing](Point &item) {
      const auto lon = item.get<0>();
      if (lon > crossing) {
        item.set<0>(lon - 360);
      }
    };
    std::ranges::for_each(line, shift);
  }

  /// @brief Normalize the two linestrings to handle longitude wrap-around at
  /// the antimeridian.
  /// @param[in] l1 First linestring
  /// @param[in] l2 Second linestring
  inline auto normalize_line_strings(LineString &l1, LineString &l2) -> void {
    if (l1.size() < 3 || l2.size() < 3) {
      throw std::runtime_error("LineString must have at least 3 points");
    }

    // Determine if the given linestrings are retrograde
    auto retrograde1 =
        is_retrograde(l1[0].get<0>(), l1[1].get<0>(), l1[2].get<0>());
    auto retrograde2 =
        is_retrograde(l2[0].get<0>(), l2[1].get<0>(), l2[2].get<0>());

    double crossing;

    if (!retrograde1 && !retrograde2) {
      crossing = std::min(l1.front().get<0>(), l2.front().get<0>());
    } else if (!retrograde1 && retrograde2) {
      crossing = std::min(l1.front().get<0>(), l2.back().get<0>());
    } else if (retrograde1 && !retrograde2) {
      crossing = std::min(l1.back().get<0>(), l2.front().get<0>());
    } else { /* if (retrograde1 && retrograde2) */
      crossing = std::min(l1.back().get<0>(), l2.back().get<0>());
    }

    shift_plus_360(l1, crossing);
    shift_plus_360(l2, crossing);

    if (!retrograde1 && !retrograde2) {
      crossing = std::max(l1.back().get<0>(), l2.back().get<0>());
    } else if (!retrograde1 && retrograde2) {
      crossing = std::max(l1.back().get<0>(), l2.front().get<0>());
    } else if (retrograde1 && !retrograde2) {
      crossing = std::max(l1.front().get<0>(), l2.back().get<0>());
    } else { /* if (retrograde1 && retrograde2) */
      crossing = std::max(l1.front().get<0>(), l2.front().get<0>());
    }

    shift_minus_360(l1, crossing);
    shift_minus_360(l2, crossing);
  }
};

}  // namespace pyinterp::geometry::cartesian

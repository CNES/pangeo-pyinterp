// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

#include "pyinterp/detail/geometry/line_string.hpp"
#include "pyinterp/detail/geometry/point.hpp"

namespace pyinterp::detail::geometry {

/// Determine if the satellite is retrograde
///
/// @param x1 first longitude
/// @param x2 second longitude
/// @param x3 third longitude
/// @return true if the satellite is retrograde
template <typename T>
constexpr auto is_retrograde(const T& x1, const T& x2, const T& x3) -> bool {
  auto delta = std::min(std::max(x1 - x2, -180.0), 180.0);
  if (std::abs(delta) == 180) {
    delta = std::min(std::max(x2 - x3, -180.0), 180.0);
  }
  return delta > 0;
}

/// Find the nearest index of a point in this linestring to a given point.
///
/// @param line the linestring to search.
/// @param point the point to search.
/// @return the index of the nearest point.
template <typename T>
auto nearest_point(const LineString<T>& line, const Point2D<T>& point)
    -> size_t {
  auto min_distance = std::numeric_limits<T>::max();
  auto index = size_t(0);
  for (auto it = line.begin(); it != line.end(); ++it) {
    auto distance = boost::geometry::comparable_distance(*it, point);
    if (distance < min_distance) {
      min_distance = distance;
      index = it - line.begin();
    }
  }
  return index;
}

/// Normalize the line strings if one of the line crosses the dateline.
///
/// @param l1 the first line string.
/// @param l2 the second line string.
template <typename T>
auto normalize_line_strings(LineString<T>& l1, LineString<T>& l2) -> void {
  if (l1.size() < 3 || l2.size() < 3) {
    throw std::runtime_error("LineString must have at least 3 points");
  }

  auto retrograde1 =
      is_retrograde(l1[0].template get<0>(), l1[1].template get<0>(),
                    l1[2].template get<0>());
  auto retrograde2 =
      is_retrograde(l2[0].template get<0>(), l2[1].template get<0>(),
                    l2[2].template get<0>());

  T crossing;

  if (!retrograde1 && !retrograde2) {
    crossing =
        std::min(l1.front().template get<0>(), l1.front().template get<0>());
  } else if (!retrograde1 && retrograde2) {
    crossing =
        std::min(l1.front().template get<0>(), l2.back().template get<0>());
  } else if (retrograde1 && !retrograde2) {
    crossing =
        std::min(l1.back().template get<0>(), l2.front().template get<0>());
  } else { /* if (retrograde1 && retrograde2) */
    crossing =
        std::min(l1.back().template get<0>(), l2.back().template get<0>());
  }

  std::for_each(l1.begin(), l1.end(), [crossing](Point2D<T>& item) {
    if (item.template get<0>() < crossing) {
      item.template set<0>(item.template get<0>() + 360);
    }
  });
  std::for_each(l2.begin(), l2.end(), [crossing](Point2D<T>& item) {
    if (item.template get<0>() < crossing) {
      item.template set<0>(item.template get<0>() + 360);
    }
  });

  if (!retrograde1 && !retrograde2) {
    crossing =
        std::max(l1.back().template get<0>(), l2.back().template get<0>());
  } else if (!retrograde1 && retrograde2) {
    crossing =
        std::max(l1.back().template get<0>(), l2.front().template get<0>());
  } else if (retrograde1 && !retrograde2) {
    crossing =
        std::max(l1.front().template get<0>(), l2.back().template get<0>());
  } else { /* if (retrograde1 && retrograde2) */
    crossing =
        std::max(l1.front().template get<0>(), l2.front().template get<0>());
  }

  std::for_each(l1.begin(), l1.end(), [crossing](Point2D<T>& item) {
    if (item.template get<0>() > crossing) {
      item.template set<0>(item.template get<0>() - 360);
    }
  });
  std::for_each(l2.begin(), l2.end(), [crossing](Point2D<T>& item) {
    if (item.template get<0>() > crossing) {
      item.template set<0>(item.template get<0>() - 360);
    }
  });
}

/// Calculate the crossover between two half-orbits.
template <typename T>
class Crossover {
 public:
  /// Default constructor
  ///
  /// @param[in] half_orbit_1 first half-orbit.
  /// @param[in] half_orbit_2 second half-orbit.
  Crossover(LineString<T> half_orbit_1, LineString<T> half_orbit_2)
      : half_orbit_1_(std::move(half_orbit_1)),
        half_orbit_2_(std::move(half_orbit_2)) {
    normalize_line_strings(half_orbit_1_, half_orbit_2_);
  }

  /// Get the crossover point between the two passes.
  ///
  /// @return the crossover location.
  [[nodiscard]] auto search() const -> std::optional<Point2D<T>> {
    auto line_string = half_orbit_1_.intersection(half_orbit_2_);
    if (line_string.empty()) {
      // There is no intersection.
      return {};
    }

    if (line_string.size() != 1) {
      // If there is a merged point between lines #1 and #2 then the method will
      // find this point for each of the segments tested.
      std::set<std::tuple<double, double>> points;
      for (auto& item : line_string) {
        points.insert(
            std::make_tuple(item.template get<0>(), item.template get<1>()));
      }
      if (points.size() != 1) {
        // If the intersection is not a point then an exception is thrown.
        throw std::runtime_error(
            "The geometry of the intersection is not a point");
      }
    }
    return line_string.front();
  }

  /// Search nearest indexes from a given point.
  ///
  /// @param point the point to search.
  /// @param predicate the distance predicate, in meters.
  /// @return the indices found on the half-orbits #1 & #2.
  [[nodiscard]] inline auto nearest(const Point2D<T>& point) const
      -> std::tuple<size_t, size_t> {
    return std::make_tuple(nearest_point(half_orbit_1_, point),
                           nearest_point(half_orbit_2_, point));
  }

 private:
  LineString<T> half_orbit_1_;
  LineString<T> half_orbit_2_;
};

}  // namespace pyinterp::detail::geometry

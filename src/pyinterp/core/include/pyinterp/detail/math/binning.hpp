// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <tuple>

namespace pyinterp::detail::math {

namespace detail {

/// Calculates the area of a polygon.
template <template <class> class Point, typename Strategy, typename T>
inline auto calculate_area(boost::geometry::model::polygon<Point<T>> &polygon,
                           const Strategy &strategy) -> T {
  auto result = boost::geometry::area(polygon, strategy);
  if (result < 0) {
    boost::geometry::reverse(polygon);
    result = boost::geometry::area(polygon, strategy);
  }
  return result;
}

/// Calculate the area of a polygon. If the area is less than one epsilon, the
/// calculated area is set to zero.
template <template <class> class Point, typename Strategy, typename T>
inline auto calculate_and_normalize_area(
    boost::geometry::model::polygon<Point<T>> &polygon,
    const Strategy &strategy, const double total_area) -> T {
  auto result = calculate_area<Point, Strategy, T>(polygon, strategy);
  if (result > total_area || std::fabs(result) < 1e-12) {
    result = 0;
  }
  return result;
}

}  // namespace detail

/// Linear binning 2D
///
/// p01 (D/ABCD)         p11 (C/ABCD)
///   ┌────────────┰──────┐
///   │     A      ┃ B    │
///   ┝━━━━━━━━━━━━╋━━━━━━┥ j
///   │            ┃      │
///   │     C      ┃ D    │
///   │            ┃      │
///   └────────────┸──────┘
/// p00 (B/ABCD)   i     p10 (A/ABCD)
///
/// @param p Query point (i, j)
/// @param p00 Point of coordinate (x0, y0)
/// @param p11 Point of coordinate (x1, y1)
/// @return a tuple that contains
///   * w00 Weight for the coordinate (x0, y0)
///   * w01 Weight for the coordinate (x0, y1)
///   * w11 Weight for the coordinate (x1, y1)
///   * w10 Weight for the coordinate (x1, y0)
template <template <class> class Point, typename Strategy, typename T>
auto binning_2d(const Point<T> &pij, const Point<T> &p00, const Point<T> &p11,
                Strategy const &strategy) -> std::tuple<T, T, T, T> {
  // Coordinates of the grid points deducted.
  const auto p01 =
      Point<T>{boost::geometry::get<0>(p00), boost::geometry::get<1>(p11)};
  const auto p10 =
      Point<T>{boost::geometry::get<0>(p11), boost::geometry::get<1>(p00)};

  // Coordinates of intersections between the point of interest and the grid
  // points.
  const auto p0j =
      Point<T>{boost::geometry::get<0>(p00), boost::geometry::get<1>(pij)};
  const auto pi1 =
      Point<T>{boost::geometry::get<0>(pij), boost::geometry::get<1>(p11)};
  const auto p1j =
      Point<T>{boost::geometry::get<0>(p11), boost::geometry::get<1>(pij)};
  const auto pi0 =
      Point<T>{boost::geometry::get<0>(pij), boost::geometry::get<1>(p00)};

  // Polygon to process
  auto polygon_a =
      boost::geometry::model::polygon<Point<T>>{{pij, p0j, p01, pi1, pij}};
  auto polygon_b =
      boost::geometry::model::polygon<Point<T>>{{pij, pi1, p11, p1j, pij}};
  auto polygon_c =
      boost::geometry::model::polygon<Point<T>>{{pij, pi0, p00, p0j, pij}};
  auto polygon_d =
      boost::geometry::model::polygon<Point<T>>{{pij, p1j, p10, pi0, pij}};
  auto grid_cell =
      boost::geometry::model::polygon<Point<T>>{{p00, p01, p11, p10, p00}};

  // Area calculation.
  const auto total_area =
      detail::calculate_area<Point, Strategy, T>(grid_cell, strategy);
  const auto area_a = detail::calculate_and_normalize_area<Point, Strategy, T>(
      polygon_a, strategy, total_area);
  const auto area_b = detail::calculate_and_normalize_area<Point, Strategy, T>(
      polygon_b, strategy, total_area);
  const auto area_c = detail::calculate_and_normalize_area<Point, Strategy, T>(
      polygon_c, strategy, total_area);
  const auto area_d = detail::calculate_and_normalize_area<Point, Strategy, T>(
      polygon_d, strategy, total_area);

  return std::make_tuple(area_b / total_area, area_d / total_area,
                         area_c / total_area, area_a / total_area);
}

}  // namespace pyinterp::detail::math

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include <array>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <concepts>

namespace pyinterp::math::interpolate {
namespace detail {

/// @brief Calculates the area of a polygon, ensuring positive orientation
/// @tparam Point Point type used in the polygon
/// @tparam Strategy Boost.Geometry strategy for area calculation
/// @tparam T Floating-point type
/// @param[in] polygon The polygon whose area is to be calculated
/// @param[in] strategy Boost.Geometry strategy for area calculation
/// @return The area of the polygon
template <template <class> class Point, typename Strategy,
          std::floating_point T>
[[nodiscard]] auto calculate_area(
    boost::geometry::model::polygon<Point<T>>& polygon,
    const Strategy& strategy) -> T {
  auto result = boost::geometry::area(polygon, strategy);

  if (result < T{0}) [[unlikely]] {
    boost::geometry::reverse(polygon);
    result = boost::geometry::area(polygon, strategy);
  }

  return result;
}

/// @brief Calculate the area of a polygon.
///
/// If the area is less than epsilon or greater than total area, the calculated
/// area is set to zero.
///
/// @tparam Point Point type used in the polygon
/// @tparam Strategy Boost.Geometry strategy for area calculation
/// @tparam T Floating-point type
/// @param[in] polygon The polygon whose area is to be calculated
/// @param[in] strategy Boost.Geometry strategy for area calculation
/// @param[in] total_area The total area for normalization
/// @return The calculated and normalized area of the polygon
template <template <class> class Point, typename Strategy,
          std::floating_point T>
[[nodiscard]] auto calculate_and_normalize_area(
    boost::geometry::model::polygon<Point<T>>& polygon,
    const Strategy& strategy, const T total_area) -> T {
  auto result = calculate_area<Point, Strategy, T>(polygon, strategy);

  // Handle numerical errors: area too large or negligibly small
  constexpr T epsilon = T{1e-12};
  if (result > total_area || std::abs(result) < epsilon) [[unlikely]] {
    result = T{0};
  }

  return result;
}

}  // namespace detail

/// @brief Bilinear weights
///
/// Computes bilinear interpolation weights based on geometric areas
///
/// Grid cell layout:
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
/// @param[in] pij Query point (i, j)
/// @param[in] p00 Grid corner at (x0, y0)
/// @param[in] p11 Grid corner at (x1, y1)
/// @param[in] strategy Boost.Geometry strategy for area calculation
/// @return A normalized array of weights {w00, w01, w11, w10}, where each
/// weight corresponds to a grid corner, and the sum of all weights equals 1.
template <template <class> class Point, typename Strategy,
          std::floating_point T>
[[nodiscard]] auto bilinear_weights(const Point<T>& pij, const Point<T>& p00,
                                    const Point<T>& p11,
                                    const Strategy& strategy)
    -> std::array<T, 4> {
  // Deduce the other two grid corners
  const auto p01 =
      Point<T>{boost::geometry::get<0>(p00), boost::geometry::get<1>(p11)};
  const auto p10 =
      Point<T>{boost::geometry::get<0>(p11), boost::geometry::get<1>(p00)};

  // Create intersection points between query point and grid edges
  const auto p0j =
      Point<T>{boost::geometry::get<0>(p00), boost::geometry::get<1>(pij)};
  const auto pi1 =
      Point<T>{boost::geometry::get<0>(pij), boost::geometry::get<1>(p11)};
  const auto p1j =
      Point<T>{boost::geometry::get<0>(p11), boost::geometry::get<1>(pij)};
  const auto pi0 =
      Point<T>{boost::geometry::get<0>(pij), boost::geometry::get<1>(p00)};

  // Create the four sub-polygons around the query point
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

  // Calculate total grid cell area
  const T total_area =
      detail::calculate_area<Point, Strategy, T>(grid_cell, strategy);

  // Calculate areas of the four sub-polygons
  const T area_a = detail::calculate_and_normalize_area<Point, Strategy, T>(
      polygon_a, strategy, total_area);
  const T area_b = detail::calculate_and_normalize_area<Point, Strategy, T>(
      polygon_b, strategy, total_area);
  const T area_c = detail::calculate_and_normalize_area<Point, Strategy, T>(
      polygon_c, strategy, total_area);
  const T area_d = detail::calculate_and_normalize_area<Point, Strategy, T>(
      polygon_d, strategy, total_area);

  // Normalize areas to get weights (opposite corners get corresponding areas)
  // w00 corresponds to area_b (opposite corner from p11)
  // w01 corresponds to area_d (opposite corner from p10)
  // w11 corresponds to area_c (opposite corner from p00)
  // w10 corresponds to area_a (opposite corner from p01)
  return {
      area_b / total_area,  // w00: weight for p00
      area_d / total_area,  // w01: weight for p01
      area_c / total_area,  // w11: weight for p11
      area_a / total_area   // w10: weight for p10
  };
}

}  // namespace pyinterp::math::interpolate

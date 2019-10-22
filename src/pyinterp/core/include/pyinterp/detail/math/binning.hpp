#pragma once
#include <tuple>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>

namespace pyinterp::detail::math {

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
/// p00 (B/ABCD)   i     p01 (A/ABCD)
///
/// @param p Query point (i, j)
/// @param p00 Point of coordinate (x0, y0)
/// @param p11 Point of coordinate (x1, y1)
/// @return a tuple that contains
///   * w00 Weight for the coordinate (x0, y0)
///   * w01 Weight for the coordinate (x0, y1)
///   * w10 Weight for the coordinate (x1, y0)
///   * w11 Weight for the coordinate (x1, y1)
template <template <class> class Point, typename T>
std::tuple<T, T, T, T> binning(const Point<T>& pij, const Point<T>& p00,
                               const Point<T>& p11) {
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
  // Area calculation.
  auto total_area = boost::geometry::area(
      boost::geometry::model::polygon<Point<T>>{{p00, p01, p11, p10, p00}});
  auto a = boost::geometry::area(
      boost::geometry::model::polygon<Point<T>>{{pij, p0j, p01, pi1, pij}});
  auto b = boost::geometry::area(
      boost::geometry::model::polygon<Point<T>>{{pij, pi1, p11, p1j, pij}});
  auto c = boost::geometry::area(
      boost::geometry::model::polygon<Point<T>>{{pij, pi0, p00, p0j, pij}});
  auto d = boost::geometry::area(
      boost::geometry::model::polygon<Point<T>>{{pij, p1j, p10, pi0, pij}});

  return std::make_tuple(b / total_area, d / total_area, c / total_area,
                         a / total_area);
}

}  // namespace pyinterp::detail::math
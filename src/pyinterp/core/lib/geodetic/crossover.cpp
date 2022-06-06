// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geodetic/crossover.hpp"

#include "pyinterp/detail/broadcast.hpp"
#include "pyinterp/detail/geometry/crossover.hpp"
#include "pyinterp/detail/math.hpp"

namespace pyinterp::geodetic {

/// Search for the nearest index of a point in this linestring to a given point.
class NearestPoint {
 public:
  /// Default constructor
  ///
  /// @param line the line string to search.
  explicit NearestPoint(const LineString& line_string) {
    size_t ix = 0;
    auto data = std::vector<std::pair<Point, size_t>>();
    data.reserve(line_string.size());
    for (const auto& item : line_string) {
      data.emplace_back(std::make_pair(item, ix++));
    }
    rtree_ = RTree(std::move(data));
  }

  /// Find the nearest index of a point in this linestring to a given
  /// point.
  ///
  /// @param point the point to search.
  /// @return the index of the nearest point or none if no intersection is
  ///         found.
  [[nodiscard]] inline auto operator()(const Point& point) const -> size_t {
    std::vector<std::pair<Point, size_t>> result;
    rtree_.query(boost::geometry::index::nearest(point, 1),
                 std::back_inserter(result));
    return result[0].second;
  }

 private:
  using RTree =
      boost::geometry::index::rtree<std::pair<Point, size_t>,
                                    boost::geometry::index::quadratic<16>>;
  RTree rtree_;
};

Crossover::Crossover(LineString half_orbit_1, LineString half_orbit_2)
    : half_orbit_1_(std::move(half_orbit_1)),
      half_orbit_2_(std::move(half_orbit_2)) {}

auto Crossover::search(const std::optional<Spheroid>& wgs) const
    -> std::optional<Point> {
  auto line_string = half_orbit_1_.intersection(half_orbit_2_, wgs);
  if (line_string.empty()) {
    // There is no intersection.
    return {};
  }

  if (line_string.size() != 1) {
    // If there is a merged point between lines #1 and #2 then the method will
    // find this point for each of the segments tested.
    std::set<std::tuple<double, double>> points;
    for (auto& item : line_string) {
      points.insert(std::make_tuple(item.get<0>(), item.get<1>()));
    }
    if (points.size() != 1) {
      // If the intersection is not a point then an exception is thrown.
      throw std::runtime_error(
          "The geometry of the intersection is not a point");
    }
  }
  return line_string[0];
}

auto Crossover::nearest(const Point& point, const double predicate,
                        const DistanceStrategy strategy,
                        const std::optional<Spheroid>& wgs) const
    -> std::optional<std::tuple<size_t, size_t>> {
  auto ix1 = NearestPoint(half_orbit_1_)(point);
  if (half_orbit_1_[ix1].distance(point, strategy, wgs) > predicate) {
    return {};
  }

  auto ix2 = NearestPoint(half_orbit_2_)(point);
  if (half_orbit_2_[ix2].distance(point, strategy, wgs) > predicate) {
    return {};
  }

  return std::make_tuple(ix1, ix2);
}

auto crossover(const Eigen::Ref<const Eigen::VectorXd>& lon1,
               const Eigen::Ref<const Eigen::VectorXd>& lat1,
               const Eigen::Ref<const Eigen::VectorXd>& lon2,
               const Eigen::Ref<const Eigen::VectorXd>& lat2, double predicate,
               const DistanceStrategy strategy,
               const std::optional<Spheroid>& wgs, bool cartesian_plane)
    -> std::optional<std::tuple<Point, std::tuple<size_t, size_t>>> {
  detail::check_container_size("lon1", lon1, "lat1", lat1);
  detail::check_container_size("lon2", lon2, "lat2", lat2);
  if (cartesian_plane) {
    // The intersection is computed in the cartesian plane.
    auto xover = detail::geometry::Crossover<double>(
        std::move(detail::geometry::LineString<double>(lon1, lat1)),
        std::move(detail::geometry::LineString<double>(lon2, lat2)));

    auto point = xover.search();
    if (!point) {
      return {};
    }
    auto [ix1, ix2] = xover.nearest(*point);
    // From this point on, we start working in geodetic coordinates.
    auto geodetic_point =
        Point(detail::math::normalize_angle(point->get<0>(), -180.0, 360.0),
              point->get<1>());
    if (geodetic_point.distance(Point(lon1[ix1], lat1[ix1]), strategy, wgs) >
        predicate) {
      return {};
    }
    if (geodetic_point.distance(Point(lon2[ix2], lat2[ix2]), strategy, wgs) >
        predicate) {
      return {};
    }
    return std::make_tuple(geodetic_point, std::make_tuple(ix1, ix2));
  }
  // The intersection is computed on the geodetic sherical plane.
  auto xover = Crossover(LineString(lon1, lat1), LineString(lon2, lat2));
  auto point = xover.search(wgs);
  if (!point) {
    return {};
  }
  auto nearest = xover.nearest(*point, predicate, strategy, wgs);
  if (!nearest) {
    return {};
  }
  return std::make_tuple(*point, *nearest);
}

}  // namespace pyinterp::geodetic

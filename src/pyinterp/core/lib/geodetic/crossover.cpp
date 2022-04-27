// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/geodetic/crossover.hpp"

namespace pyinterp::geodetic {

class NearestPoint {
 public:
  /// Default constructor
  explicit NearestPoint(const LineString& line_string) {
    size_t ix = 0;
    for (const auto& item : line_string) {
      rtree_.insert(std::make_pair(item, ix));
      ++ix;
    }
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
  boost::geometry::index::rtree<std::pair<Point, size_t>,
                                boost::geometry::index::quadratic<16>>
      rtree_;
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

}  // namespace pyinterp::geodetic

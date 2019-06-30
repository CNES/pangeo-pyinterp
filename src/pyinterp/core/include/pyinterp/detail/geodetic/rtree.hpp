// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include "pyinterp/detail/geodetic/coordinates.hpp"
#include "pyinterp/detail/geometry/rtree.hpp"
#include "pyinterp/detail/thread.hpp"
#include <Eigen/Core>
#include <optional>

namespace pyinterp {
namespace detail {
namespace geodetic {

/// RTree spatial index for geodetic point
///
/// @note
/// The tree of the "boost" library allows to directly handle the geodetic
/// coordinates, but it is much less efficient than the use of the tree in a
/// Cartesian space.
/// @tparam Coordinate The class of storage for a point's coordinates.
/// @tparam Type The type of data stored in the tree.
template <typename Coordinate, typename Type>
class RTree : public geometry::RTree<Coordinate, Type, 3> {
 public:
  /// Type of distances between two points
  using distance_t = typename boost::geometry::default_distance_result<
      geometry::EquatorialPoint3D<Coordinate>,
      geometry::EquatorialPoint3D<Coordinate>>::type;

  /// Type of query results.
  using result_t = std::pair<distance_t, Type>;

  /// Default constructor
  explicit RTree(const std::optional<System> &wgs)
      : geometry::RTree<Coordinate, Type, 3>(),
        coordinates_(wgs.value_or(System())),
        strategy_(boost::geometry::strategy::distance::haversine<Coordinate>{
            Coordinate(wgs.value_or(System()).semi_major_axis())}) {}

  /// Default destructor
  virtual ~RTree() = default;

  /// Default copy constructor
  RTree(const RTree &) = default;

  /// Default copy assignment operator
  RTree &operator=(const RTree &) = default;

  /// Move constructor
  RTree(RTree &&) noexcept = default;

  /// Move assignment operator
  RTree &operator=(RTree &&) noexcept = default;

  /// Returns the box able to contain all values stored in the container.
  ///
  /// @returns The box able to contain all values stored in the container or an
  /// invalid box if there are no values in the container.
  std::optional<geometry::EquatorialBox3D<Coordinate>> equatorial_bounds()
      const {
    if (this->empty()) {
      return {};
    }

    Coordinate x0 = std::numeric_limits<Coordinate>::max();
    Coordinate x1 = std::numeric_limits<Coordinate>::min();
    Coordinate y0 = std::numeric_limits<Coordinate>::max();
    Coordinate y1 = std::numeric_limits<Coordinate>::min();
    Coordinate z0 = std::numeric_limits<Coordinate>::max();
    Coordinate z1 = std::numeric_limits<Coordinate>::min();

    std::for_each(this->tree_->begin(), this->tree_->end(),
                  [&](const auto &item) {
                    auto lla = coordinates_.ecef_to_lla(item.first);
                    x0 = std::min(x0, boost::geometry::get<0>(lla));
                    x1 = std::max(x1, boost::geometry::get<0>(lla));
                    y0 = std::min(y0, boost::geometry::get<1>(lla));
                    y1 = std::max(y1, boost::geometry::get<1>(lla));
                    z0 = std::min(z0, boost::geometry::get<2>(lla));
                    z1 = std::max(z1, boost::geometry::get<2>(lla));
                  });

    return geometry::EquatorialBox3D<Coordinate>({x0, y0, z0}, {x1, y1, z1});
  }

  /// Search for the K nearest neighbors of a given point.
  ///
  /// @param point Point of interest
  /// @param k The number of nearest neighbors to search.
  /// @return the k nearest neighbors
  std::vector<result_t> query(
      const geometry::EquatorialPoint3D<Coordinate> &point,
      const uint32_t k) const {
    std::vector<result_t> result;
    std::for_each(
        this->tree_->qbegin(boost::geometry::index::nearest(
            coordinates_.lla_to_ecef(point), k)),
        this->tree_->qend(), [&](const auto &item) {
          result.emplace_back(std::make_pair(
              boost::geometry::distance(
                  point, coordinates_.ecef_to_lla(item.first), strategy_),
              item.second));
        });
    return result;
  }

  /// Search for the nearest neighbors of a given point within a radius r.
  ///
  /// @param point Point of interest
  /// @param radius distance within which neighbors are returned
  /// @return the k nearest neighbors
  std::vector<result_t> query_ball(
      const geometry::EquatorialPoint3D<Coordinate> &point,
      const double radius) const {
    std::vector<result_t> result;
    std::for_each(
        this->tree_->qbegin(
            boost::geometry::index::satisfies([&](const auto &item) {
              return boost::geometry::distance(
                         coordinates_.ecef_to_lla(item.first), point,
                         strategy_) < radius;
            })),
        this->tree_->qend(), [&](const auto &item) {
          result.emplace_back(std::make_pair(
              boost::geometry::distance(
                  point, coordinates_.ecef_to_lla(item.first), strategy_),
              item.second));
        });
    return result;
  }

  /// Search for the K nearest neighbors around a given point.
  ///
  /// @param point Point of interest
  /// @param k The number of nearest neighbors to search.
  /// @return the k nearest neighbors if the point is within by its
  /// neighbors.
  std::vector<result_t> query_within(
      const geometry::EquatorialPoint3D<Coordinate> &point,
      const uint32_t k) const {
    std::vector<result_t> result;
    auto ecef =
        boost::geometry::model::multi_point<geometry::Point3D<Coordinate>>();
    ecef.reserve(k);
    std::for_each(
        this->tree_->qbegin(boost::geometry::index::nearest(
            coordinates_.lla_to_ecef(point), k)),
        this->tree_->qend(), [&](const auto &item) {
          ecef.emplace_back(item.first);
          result.emplace_back(std::make_pair(
              boost::geometry::distance(
                  point, coordinates_.ecef_to_lla(item.first), strategy_),
              item.second));
        });
    if (!boost::geometry::covered_by(
            coordinates_.lla_to_ecef(point),
            boost::geometry::return_envelope<
                boost::geometry::model::box<geometry::Point3D<Coordinate>>>(
                ecef))) {
      result.clear();
    }
    return result;
  }

  /// Interpolation of the value at the requested position.
  ///
  /// @param point Point of interrest
  /// @param radius The maximum radius of the search (m).
  /// @param k The number of nearest neighbors to be used for calculating the
  /// interpolated value.
  /// @return a tuple containing the interpolated value and the number of
  /// neighbors used in the calculation.
  /// @param within If true, the method ensures that the neighbors found are
  /// located around the point of interest. In other words, this parameter
  /// ensures that the calculated values will not be extrapolated.
  std::pair<Type, uint32_t> inverse_distance_weighting(
      const geometry::EquatorialPoint3D<Coordinate> &point,
      distance_t radius = std::numeric_limits<distance_t>::max(),
      uint32_t k = 4, bool within = true) const {
    Type result = 0;
    Type total_weight = 0;

    // We're looking for the nearest k points.
    auto nearest = within ? query(point, k) : query_within(point, k);
    uint32_t neighbors = 0;

    // For each point, the distance between the point requested and the point
    // found is calculated and the information required for the Inverse distance
    // weighting interpolation method is updated.
    for (const auto &item : nearest) {
      const auto distance = item.first;
      if (distance < 1e-6) {
        // If the user has requested a grid point, the mesh value is returned.
        return std::make_pair(item.second, k);
      }

      if (distance <= radius) {
        // If the neighbor found is within an acceptable radius it can be taken
        // into account in the calculation.
        auto wk = 1 / math::sqr(distance);
        total_weight += wk;
        result += item.second * wk;
        ++neighbors;
      }
    }

    // Finally the interpolated value is returned if there are selected points
    // otherwise one returns an undefined value.
    return total_weight != 0
               ? std::make_pair(static_cast<Type>(result / total_weight),
                                neighbors)
               : std::make_pair(std::numeric_limits<Type>::quiet_NaN(),
                                static_cast<uint32_t>(0));
  }

 protected:
  /// System for converting Geodetic coordinates into Cartesian coordinates.
  Coordinates coordinates_;

  /// Distance calculation formulae on lat/lon coordinates
  boost::geometry::strategy::distance::haversine<Coordinate> strategy_;
};

}  // namespace geodetic
}  // namespace detail
}  // namespace pyinterp

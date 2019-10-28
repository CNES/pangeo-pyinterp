// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <algorithm>
#include <boost/geometry.hpp>
#include <optional>
#include "pyinterp/detail/geometry/box.hpp"
#include "pyinterp/detail/geometry/point.hpp"

namespace pyinterp::detail::geometry {

/// Index points in the Cartesian space at N dimensions.
///
/// @tparam Coordinate The class of storage for a point's coordinates.
/// @tparam Type The type of data stored in the tree.
/// @tparam N Number of dimensions in the Cartesian space handled.
template <typename Coordinate, typename Type, size_t N>
class RTree {
 public:
  /// Type of distances between two points.
  using distance_t = typename boost::geometry::default_distance_result<
      geometry::PointND<Coordinate, N>, geometry::PointND<Coordinate, N>>::type;

  /// Type of query results.
  using result_t = std::pair<distance_t, Type>;

  /// Value handled by this object
  using value_t = std::pair<geometry::PointND<Coordinate, N>, Type>;

  /// Spatial index used
  using rtree_t =
      boost::geometry::index::rtree<value_t, boost::geometry::index::rstar<16>>;

  /// Default constructor
  RTree() : tree_(new rtree_t{}) {}

  /// Default destructor
  virtual ~RTree() = default;

  /// Default copy constructor
  RTree(const RTree &) = default;

  /// Default copy assignment operator
  auto operator=(const RTree &) -> RTree & = default;

  /// Move constructor
  RTree(RTree &&) noexcept = default;

  /// Move assignment operator
  auto operator=(RTree &&) noexcept -> RTree & = default;

  /// Returns the box able to contain all values stored in the container.
  ///
  /// @returns The box able to contain all values stored in the container or an
  /// invalid box if there are no values in the container.
  virtual auto bounds() const -> std::optional<geometry::BoxND<Coordinate, N>> {
    if (empty()) {
      return {};
    }
    return tree_->bounds();
  }

  /// Returns the number of points of this mesh
  ///
  /// @return the number of points
  [[nodiscard]] inline auto size() const -> size_t { return tree_->size(); }

  /// Query if the container is empty.
  ///
  /// @return true if the container is empty.
  [[nodiscard]] inline auto empty() const -> bool { return tree_->empty(); }

  /// Removes all values stored in the container.
  inline void clear() { tree_->clear(); }

  /// The tree is created using packing algorithm (The old data is erased before
  /// construction.)
  ///
  /// @param points
  void packing(const std::vector<value_t> &points) { *tree_ = rtree_t(points); }

  /// Insert new data into the search tree
  ///
  /// @param point
  void insert(const value_t &value) { tree_->insert(value); }

  /// Search for the K nearest neighbors of a given point.
  ///
  /// @param point Point of interest
  /// @param k The number of nearest neighbors to search.
  /// @return the k nearest neighbors
  auto query(const geometry::PointND<Coordinate, N> &point,
             const uint32_t k) const -> std::vector<result_t> {
    auto result = std::vector<result_t>();
    std::for_each(
        tree_->qbegin(boost::geometry::index::nearest(point, k)), tree_->qend(),
        [&point, &result](const auto &item) {
          result.emplace_back(std::make_pair(
              boost::geometry::distance(point, item.first), item.second));
        });
    return result;
  }

  /// Search for the nearest neighbors of a given point within a radius r.
  ///
  /// @param point Point of interest
  /// @param radius distance within which neighbors are returned
  /// @return the k nearest neighbors
  auto query_ball(const geometry::PointND<Coordinate, N> &point,
                  const double radius) const -> std::vector<result_t> {
    auto result = std::vector<result_t>();
    std::for_each(
        tree_->qbegin(boost::geometry::index::satisfies([&](const auto &item) {
          return boost::geometry::distance(item.first, point) <= radius;
        })),
        tree_->qend(), [&point, &result](const auto &item) {
          result.emplace_back(std::make_pair(
              boost::geometry::distance(point, item.first), item.second));
        });
    return result;
  }

  /// Search for the nearest K neighbors around a given point.
  ///
  /// @param point Point of interest
  /// @param k The number of nearest neighbors to search.
  /// @return the k nearest neighbors if the point is within by its
  /// neighbors.
  auto query_within(const geometry::PointND<Coordinate, N> &point,
                    const uint32_t k) const -> std::vector<result_t> {
    auto result = std::vector<result_t>();
    auto points =
        boost::geometry::model::multi_point<geometry::PointND<Coordinate, N>>();
    points.reserve(k);

    std::for_each(
        tree_->qbegin(boost::geometry::index::nearest(point, k)), tree_->qend(),
        [&points, &point, &result](const auto &item) {
          points.emplace_back(item.first);
          result.emplace_back(std::make_pair(
              boost::geometry::distance(point, item.first), item.second));
        });

    // Are found points located around the requested point?
    if (!boost::geometry::covered_by(
            point,
            boost::geometry::return_envelope<
                boost::geometry::model::box<geometry::PointND<Coordinate, N>>>(
                points))) {
      return {};
    }
    return result;
  }

 protected:
  /// Geographic index used to store data and their searches.
  std::shared_ptr<rtree_t> tree_;
};

}  // namespace pyinterp::detail::geometry

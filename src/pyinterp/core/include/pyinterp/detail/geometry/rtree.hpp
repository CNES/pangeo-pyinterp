// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <boost/geometry.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "pyinterp/detail/geometry/box.hpp"
#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/detail/math/radial_basis_functions.hpp"
#include "pyinterp/detail/math/window_functions.hpp"

namespace pyinterp::detail::geometry {

/// Index points in the Cartesian space at N dimensions.
///
/// @tparam CoordinateType The class of storage for a point's coordinates.
/// @tparam Type The type of data stored in the tree.
/// @tparam N Number of dimensions in the Cartesian space handled.
template <typename CoordinateType, typename Type, size_t N>
class RTree {
 public:
  /// Type of the point handled by this instance.
  using point_t = geometry::PointND<CoordinateType, N>;

  /// Type of distances between two points.
  using distance_t = typename boost::geometry::default_distance_result<
      point_t, geometry::PointND<CoordinateType, N>>::type;

  /// Type of query results.
  using result_t = std::pair<distance_t, Type>;

  /// Value handled by this object
  using value_t = std::pair<point_t, Type>;

  /// Spatial index used
  using rtree_t =
      boost::geometry::index::rtree<value_t, boost::geometry::index::rstar<16>>;

  /// Type of the implicit conversion between the type of coordinates and values
  using promotion_t =
      decltype(std::declval<CoordinateType>() + std::declval<Type>());

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
  virtual inline auto bounds() const
      -> std::optional<geometry::BoxND<CoordinateType, N>> {
    if (empty()) {
      return {};
    }
    return tree_->bounds();
  }

  /// Returns the number of points of this mesh
  ///
  /// @return the number of points
  [[nodiscard]] constexpr auto size() const -> size_t { return tree_->size(); }

  /// Query if the container is empty.
  ///
  /// @return true if the container is empty.
  [[nodiscard]] constexpr auto empty() const -> bool { return tree_->empty(); }

  /// Removes all values stored in the container.
  inline auto clear() -> void { tree_->clear(); }

  /// The tree is created using packing algorithm (The old data is erased before
  /// construction.)
  ///
  /// @param points
  inline auto packing(const std::vector<value_t> &points) -> void {
    *tree_ = rtree_t(points);
  }

  /// Insert new data into the search tree
  ///
  /// @param point
  inline auto insert(const value_t &value) -> void { tree_->insert(value); }

  /// Search for the K nearest neighbors of a given point.
  ///
  /// @param point Point of interest
  /// @param k The number of nearest neighbors to search.
  /// @return the k nearest neighbors:
  auto query(const point_t &point, const uint32_t k) const
      -> std::vector<result_t> {
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
  auto query_ball(const point_t &point, const distance_t radius) const
      -> std::vector<result_t> {
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
  auto query_within(const point_t &point, const uint32_t k) const
      -> std::vector<result_t> {
    auto result = std::vector<result_t>();
    auto points = boost::geometry::model::multi_point<point_t>();
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
            point, boost::geometry::return_envelope<
                       boost::geometry::model::box<point_t>>(points))) {
      return {};
    }
    return result;
  }

  /// Interpolation of the value at the requested position.
  ///
  /// @param point Point of interrest
  /// @param radius The maximum radius of the search.
  /// @param k The number of nearest neighbors to be used for calculating the
  /// interpolated value.
  /// @param p the power parameter.
  /// @param within If true, the method ensures that the neighbors found are
  /// located around the point of interest. In other words, this parameter
  /// ensures that the calculated values will not be extrapolated.
  /// @return a tuple containing the interpolated value and the number of
  /// neighbors used in the calculation.
  auto inverse_distance_weighting(const point_t &point, distance_t radius,
                                  uint32_t k, uint32_t p, bool within) const
      -> std::pair<distance_t, uint32_t> {
    distance_t result = 0;
    distance_t total_weight = 0;

    // We're looking for the nearest k points.
    auto nearest = within ? query_within(point, k) : query(point, k);
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
        auto wk =
            static_cast<Type>(1 / std::pow(distance, static_cast<Type>(p)));
        total_weight += wk;
        result += item.second * wk;
        ++neighbors;
      }
    }

    // Finally the interpolated value is returned if there are selected points
    // otherwise one returns an undefined value.
    return total_weight != 0
               ? std::make_pair(static_cast<distance_t>(result / total_weight),
                                neighbors)
               : std::make_pair(std::numeric_limits<distance_t>::quiet_NaN(),
                                static_cast<uint32_t>(0));
  }

  /// Search for the nearest K neighbors of a given point.
  ///
  /// @param point Point of interest
  /// @param radius The maximum radius of the search.
  /// @param k The number of nearest neighbors to be used for calculating the
  /// interpolated value.
  /// @return A tuple containing the matrix describing the coordinates of the
  /// selected points and a vector of the values of the points. The arrays will
  /// be empty if no points are selected.
  auto nearest(const point_t &point, const distance_t radius,
               const uint32_t k) const
      -> std::tuple<Matrix<promotion_t>, Vector<promotion_t>> {
    auto coordinates = Matrix<promotion_t>(N, k);
    auto values = Vector<promotion_t>(k);
    auto jx = 0U;

    std::for_each(
        tree_->qbegin(boost::geometry::index::nearest(point, k)), tree_->qend(),
        [&](const auto &item) {
          if (boost::geometry::distance(point, item.first) <= radius) {
            // If the point is not too far away, it is inserted and
            // its coordinates and value are stored.
            for (size_t ix = 0; ix < N; ++ix) {
              coordinates(ix, jx) = geometry::point::get(item.first, ix);
            }
            values(jx++) = item.second;
          }
        });

    // The arrays are resized according to the number of selected points. This
    // number can be zero.
    coordinates.conservativeResize(N, jx);
    values.conservativeResize(jx);
    return std::make_tuple(coordinates, values);
  }

  /// Search for the nearest K neighbors around a given point.
  ///
  /// @param point Point of interest
  /// @param radius The maximum radius of the search.
  /// @param k The number of nearest neighbors to be used for calculating the
  /// interpolated value.
  /// @return A tuple containing the matrix describing the coordinates of the
  /// selected points and a vector of the values of the points. The arrays will
  /// be empty if no points are selected.
  auto nearest_within(const point_t &point, const distance_t radius,
                      const uint32_t k) const
      -> std::tuple<Matrix<promotion_t>, Vector<promotion_t>> {
    auto points = boost::geometry::model::multi_point<point_t>();
    auto coordinates = Matrix<promotion_t>(N, k);
    auto values = Vector<promotion_t>(k);
    auto jx = 0U;

    // List of selected points ()
    points.reserve(k);

    std::for_each(
        tree_->qbegin(boost::geometry::index::nearest(point, k)), tree_->qend(),
        [&](const auto &item) {
          if (boost::geometry::distance(point, item.first) <= radius) {
            // If the point is not too far away, it is inserted and
            // its coordinates and value are stored.
            points.emplace_back(item.first);
            for (size_t ix = 0; ix < N; ++ix) {
              coordinates(ix, jx) = geometry::point::get(item.first, ix);
            }
            values(jx++) = item.second;
          }
        });

    // If the point is not covered by its closest neighbors, an empty set will
    // be returned.
    if (!boost::geometry::covered_by(
            point, boost::geometry::return_envelope<
                       boost::geometry::model::box<point_t>>(points))) {
      jx = 0;
    }

    // The arrays are resized according to the number of selected points. This
    // number can be zero.
    coordinates.conservativeResize(N, jx);
    values.conservativeResize(jx);
    return std::make_tuple(coordinates, values);
  }

  /// Interpolate the value of a point using a Radial Basis Function.
  ///
  /// @param point Point of interest
  /// @param rbf The radial basis function to be used.
  /// @param radius The maximum radius of the search.
  /// @param k The number of nearest neighbors to be used for calculating the
  /// interpolated value.
  /// @param within If true, the method ensures that the neighbors found are
  /// located around the point of interest. In other words, this parameter
  /// ensures that the calculated values will not be extrapolated.
  /// @return A pair containing the interpolated value and the number of
  /// neighbors used in the calculation.
  auto radial_basis_function(const point_t &point,
                             const math::RBF<promotion_t> &rbf,
                             distance_t radius, uint32_t k, bool within) const
      -> std::pair<promotion_t, uint32_t> {
    auto [coordinates, values] =
        within ? nearest_within(point, radius, k) : nearest(point, radius, k);
    if (values.size() == 0) {
      return std::make_pair(std::numeric_limits<promotion_t>::quiet_NaN(), 0);
    }
    auto xi = Eigen::Matrix<promotion_t, N, 1>();
    for (size_t ix = 0; ix < N; ++ix) {
      xi(ix, 0) = geometry::point::get(point, ix);
    }
    auto interpolated = rbf.interpolate(coordinates, values, xi);
    return std::make_pair(interpolated(0),
                          static_cast<uint32_t>(values.size()));
  }

  /// Interpolate the value of a point using a Window Function.
  ///
  /// @param point Point of interest
  /// @param wf The window function to be used.
  /// @param radius The maximum radius of the search.
  /// @param k The number of nearest neighbors to be used for calculating the
  /// interpolated value.
  /// @param within If true, the method ensures that the neighbors found are
  /// located around the point of interest. In other words, this parameter
  /// ensures that the calculated values will not be extrapolated.
  /// @return A pair containing the interpolated value and the number of
  /// neighbors used in the calculation.
  auto window_function(const point_t &point,
                       const math::WindowFunction<distance_t> &wf,
                       const distance_t arg, distance_t radius, uint32_t k,
                       bool within) const -> std::pair<distance_t, uint32_t> {
    distance_t result = 0;
    distance_t total_weight = 0;

    auto nearest = within ? query_within(point, k) : query(point, k);
    uint32_t neighbors = 0;

    for (const auto &item : nearest) {
      const auto distance = item.first;

      auto wk = wf(distance, radius, arg);
      total_weight += wk;
      result += item.second * wk;
      ++neighbors;
    }

    return total_weight != 0
               ? std::make_pair(static_cast<distance_t>(result / total_weight),
                                neighbors)
               : std::make_pair(std::numeric_limits<distance_t>::quiet_NaN(),
                                static_cast<uint32_t>(0));
  }

 protected:
  /// Geographic index used to store data and their searches.
  std::shared_ptr<rtree_t> tree_;
};

}  // namespace pyinterp::detail::geometry

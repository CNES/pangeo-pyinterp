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

#include "pyinterp/detail/geometry/point.hpp"
#include "pyinterp/detail/math/radial_basis_functions.hpp"
#include "pyinterp/detail/math/window_functions.hpp"

namespace pyinterp::detail::geometry {

/// Index points.
///
/// @tparam Point point type
/// @tparam Type of value associated to each point
template <typename Point, typename Type>
class RTree {
 public:
  using dimension_t = typename boost::geometry::traits::dimension<Point>;

  /// Type of point coordinates.
  using coordinate_t =
      typename boost::geometry::traits::coordinate_type<Point>::type;

  /// Type of distance function
  using distance_t =
      typename boost::geometry::default_distance_result<Point, Point>::type;

  /// Type of query results.
  using result_t = std::pair<distance_t, Type>;

  /// Value handled by this object
  using value_t = std::pair<Point, Type>;

  /// Spatial index used
  using rtree_t =
      boost::geometry::index::rtree<value_t, boost::geometry::index::rstar<16>>;

  /// Type of the implicit conversion between the type of coordinates and values
  using promotion_t =
      decltype(std::declval<coordinate_t>() + std::declval<Type>());

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
      -> std::optional<typename rtree_t::bounds_type> {
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

  /// Search for the K nearest neighbors of a given point using the given
  /// strategy.
  ///
  /// @param point Point of interest
  /// @param strategy Strategy used to calculate the distance between points.
  /// @param k The number of nearest neighbors to search.
  /// @return the k nearest neighbors:
  template <typename Strategy>
  auto query(const Point &point, const Strategy &strategy,
             const uint32_t k) const -> std::vector<result_t> {
    auto result = std::vector<result_t>();
    std::for_each(tree_->qbegin(boost::geometry::index::nearest(point, k)),
                  tree_->qend(),
                  [&point, &result, &strategy](const auto &item) {
                    result.emplace_back(std::make_pair(
                        boost::geometry::distance(point, item.first, strategy),
                        item.second));
                  });
    return result;
  }

  /// @overload query(const Point &, const Strategy &, const uint32_t) const
  ///
  /// Overload of the query method with the default strategy.
  auto query(const Point &point, const uint32_t k) const
      -> std::vector<result_t> {
    return query(point, boost::geometry::default_strategy(), k);
  }

  /// Search for the nearest neighbors of a given point within a radius r using
  /// the given strategy.
  ///
  /// @param point Point of interest
  /// @param strategy strategy used to compute the distance
  /// @param radius distance within which neighbors are returned
  /// @return the k nearest neighbors
  template <typename Strategy>
  auto query_ball(const Point &point, const Strategy &strategy,
                  const coordinate_t radius) const -> std::vector<result_t> {
    auto result = std::vector<result_t>();
    std::for_each(
        tree_->qbegin(boost::geometry::index::satisfies([&](const auto &item) {
          return boost::geometry::distance(item.first, point) <= radius;
        })),
        tree_->qend(), [&point, &result, &strategy](const auto &item) {
          result.emplace_back(std::make_pair(
              boost::geometry::distance(point, item.first, strategy),
              item.second));
        });
    return result;
  }

  /// @overload query_ball(const Point &, const Strategy &, const coordinate_t)
  ///
  /// Overload of the query_ball method with the default strategy.
  auto query_ball(const Point &point, const coordinate_t radius) const
      -> std::vector<result_t> {
    return query_ball(point, boost::geometry::default_strategy(), radius);
  }

  /// Search for the nearest K neighbors around a given point using the given
  /// strategy.
  ///
  /// @param point Point of interest
  /// @param strategy strategy used to compute the distance
  /// @param k The number of nearest neighbors to search.
  /// @return the k nearest neighbors if the point is within by its
  /// neighbors.
  template <typename Strategy>
  auto query_within(const Point &point, const Strategy &strategy,
                    const uint32_t k) const -> std::vector<result_t> {
    auto result = std::vector<result_t>();
    auto points = boost::geometry::model::multi_point<Point>();
    points.reserve(k);

    std::for_each(tree_->qbegin(boost::geometry::index::nearest(point, k)),
                  tree_->qend(),
                  [&points, &point, &result, &strategy](const auto &item) {
                    points.emplace_back(item.first);
                    result.emplace_back(std::make_pair(
                        boost::geometry::distance(point, item.first, strategy),
                        item.second));
                  });

    // Are found points located around the requested point?
    if (!boost::geometry::covered_by(
            point, boost::geometry::return_envelope<
                       boost::geometry::model::box<Point>>(points))) {
      return {};
    }
    return result;
  }

  /// @overload query_within(const Point &, const Strategy &, const uint32_t)
  ///
  /// Overload of the query_within method with the default strategy.
  auto query_within(const Point &point, const uint32_t k) const
      -> std::vector<result_t> {
    return query_within(point, boost::geometry::default_strategy(), k);
  }

  /// Search for the nearest K neighbors around a given point.
  ///
  /// @param point Point of interest
  /// @param strategy strategy used to compute the distance
  /// @param radius distance within which neighbors are returned
  /// @param k The number of nearest neighbors to search.
  /// @param within if true, the method returns the k nearest neighbors if the
  /// point is within by its neighbors.
  /// @return the k nearest neighbors.
  template <typename Strategy>
  auto value(const Point &point, const Strategy &strategy,
             const std::optional<coordinate_t> &radius, const uint32_t k,
             const bool within) const -> std::vector<value_t> {
    auto result = std::vector<value_t>();
    std::for_each(tree_->qbegin(boost::geometry::index::nearest(point, k)),
                  tree_->qend(),
                  [&result](const auto &item) { result.emplace_back(item); });

    // Remove points outside the radius
    if (radius.has_value()) {
      result.erase(std::remove_if(result.begin(), result.end(),
                                  [&](const auto &item) {
                                    return boost::geometry::distance(
                                               item.first, point, strategy) >
                                           radius.value();
                                  }),
                   result.end());
    }

    // If the point is not within the neighbors, return an empty vector
    if (within) {
      auto points = boost::geometry::model::multi_point<Point>();
      points.reserve(result.size());
      std::for_each(result.begin(), result.end(), [&points](const auto &item) {
        points.emplace_back(item.first);
      });

      if (!boost::geometry::covered_by(
              point, boost::geometry::return_envelope<
                         boost::geometry::model::box<Point>>(points))) {
        return {};
      }
    }
    return result;
  }

  /// @overload value(const Point &, const Strategy &, const
  /// std::optional<coordinate_t> &, const uint32_t, const bool) const
  ///
  /// Overload of the value method with the default strategy.
  auto value(const Point &point, const std::optional<coordinate_t> &radius,
             const uint32_t k, const bool within) const
      -> std::vector<value_t> {
    return value(point, boost::geometry::default_strategy(), radius, k, within);
  }

  /// Interpolation of the value at the requested position.
  ///
  /// @param point Point of interest.
  /// @param strategy strategy used to compute the distance.
  /// @param radius The maximum radius of the search.
  /// @param k The number of nearest neighbors to be used for calculating the
  /// interpolated value.
  /// @param p the power parameter.
  /// @param within If true, the method ensures that the neighbors found are
  /// located around the point of interest. In other words, this parameter
  /// ensures that the calculated values will not be extrapolated.
  /// @return a tuple containing the interpolated value and the number of
  /// neighbors used in the calculation.
  template <typename Strategy>
  auto inverse_distance_weighting(const Point &point, const Strategy &strategy,
                                  coordinate_t radius, uint32_t k, uint32_t p,
                                  bool within) const
      -> std::pair<coordinate_t, uint32_t> {
    coordinate_t result = 0;
    coordinate_t total_weight = 0;

    // We're looking for the nearest k points.
    auto nearest =
        within ? query_within(point, strategy, k) : query(point, strategy, k);
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
               ? std::make_pair(
                     static_cast<coordinate_t>(result / total_weight),
                     neighbors)
               : std::make_pair(std::numeric_limits<coordinate_t>::quiet_NaN(),
                                static_cast<uint32_t>(0));
  }

  /// @overload inverse_distance_weighting(const Point &, const Strategy &,
  /// const coordinate_t, const uint32_t, const uint32_t, const bool)
  ///
  /// Overload of the inverse_distance_weighting method with the default
  /// strategy.
  auto inverse_distance_weighting(const Point &point, coordinate_t radius,
                                  uint32_t k, uint32_t p, bool within) const {
    return inverse_distance_weighting(
        point, boost::geometry::default_strategy(), radius, k, p, within);
  }

  /// Search for the nearest K neighbors of a given point using the given
  /// strategy.
  ///
  /// @param point Point of interest.
  /// @param strategy strategy used to compute the distance.
  /// @param radius The maximum radius of the search.
  /// @param k The number of nearest neighbors to be used for calculating the
  /// interpolated value.
  /// @return A tuple containing the matrix describing the coordinates of the
  /// selected points and a vector of the values of the points. The arrays will
  /// be empty if no points are selected.
  template <typename Strategy>
  auto nearest(const Point &point, const Strategy &strategy,
               const coordinate_t radius, const uint32_t k) const
      -> std::tuple<Matrix<promotion_t>, Vector<promotion_t>> {
    auto coordinates = Matrix<promotion_t>(dimension_t::value, k);
    auto values = Vector<promotion_t>(k);
    auto jx = 0U;

    std::for_each(tree_->qbegin(boost::geometry::index::nearest(point, k)),
                  tree_->qend(), [&](const auto &item) {
                    if (boost::geometry::distance(point, item.first,
                                                  strategy) <= radius) {
                      // If the point is not too far away, it is inserted and
                      // its coordinates and value are stored.
                      for (size_t ix = 0; ix < dimension_t::value; ++ix) {
                        coordinates(ix, jx) =
                            geometry::point::get(item.first, ix);
                      }
                      values(jx++) = item.second;
                    }
                  });

    // The arrays are resized according to the number of selected points. This
    // number can be zero.
    coordinates.conservativeResize(dimension_t::value, jx);
    values.conservativeResize(jx);
    return std::make_tuple(coordinates, values);
  }

  /// @overload nearest(const Point &, const Strategy &, const coordinate_t,
  /// const uint32_t)
  ///
  /// Overload of the nearest method with the default strategy.
  auto nearest(const Point &point, const coordinate_t radius,
               const uint32_t k) const {
    return nearest(point, boost::geometry::default_strategy(), radius, k);
  }

  /// Search for the nearest K neighbors around a given point using the given
  /// strategy.
  ///
  /// @param point Point of interest.
  /// @param strategy strategy used to compute the distance.
  /// @param radius The maximum radius of the search.
  /// @param k The number of nearest neighbors to be used for calculating the
  /// interpolated value.
  /// @return A tuple containing the matrix describing the coordinates of the
  /// selected points and a vector of the values of the points. The arrays will
  /// be empty if no points are selected.
  template <typename Strategy>
  auto nearest_within(const Point &point, const Strategy &strategy,
                      const coordinate_t radius, const uint32_t k) const
      -> std::tuple<Matrix<promotion_t>, Vector<promotion_t>> {
    auto points = boost::geometry::model::multi_point<Point>();
    auto coordinates = Matrix<promotion_t>(dimension_t::value, k);
    auto values = Vector<promotion_t>(k);
    auto jx = 0U;

    // List of selected points ()
    points.reserve(k);

    std::for_each(tree_->qbegin(boost::geometry::index::nearest(point, k)),
                  tree_->qend(), [&](const auto &item) {
                    if (boost::geometry::distance(point, item.first,
                                                  strategy) <= radius) {
                      // If the point is not too far away, it is inserted and
                      // its coordinates and value are stored.
                      points.emplace_back(item.first);
                      for (size_t ix = 0; ix < dimension_t::value; ++ix) {
                        coordinates(ix, jx) =
                            geometry::point::get(item.first, ix);
                      }
                      values(jx++) = item.second;
                    }
                  });

    // If the point is not covered by its closest neighbors, an empty set will
    // be returned.
    if (!boost::geometry::covered_by(
            point, boost::geometry::return_envelope<
                       boost::geometry::model::box<Point>>(points))) {
      jx = 0;
    }

    // The arrays are resized according to the number of selected points. This
    // number can be zero.
    coordinates.conservativeResize(dimension_t::value, jx);
    values.conservativeResize(jx);
    return std::make_tuple(coordinates, values);
  }

  /// @overload nearest_within(const Point &, const Strategy &, const
  /// coordinate_t, const uint32_t)
  ///
  /// Overload of the nearest_within method with the default strategy.
  auto nearest_within(const Point &point, const coordinate_t radius,
                      const uint32_t k) const
      -> std::tuple<Matrix<promotion_t>, Vector<promotion_t>> {
    return nearest_within(point, boost::geometry::default_strategy(), radius,
                          k);
  }

  /// Interpolate the value of a point using a Radial Basis Function using the
  /// given strategy.
  ///
  /// @param point Point of interest
  /// @param strategy strategy used to compute the distance.
  /// @param rbf The radial basis function to be used.
  /// @param radius The maximum radius of the search.
  /// @param k The number of nearest neighbors to be used for calculating the
  /// interpolated value.
  /// @param within If true, the method ensures that the neighbors found are
  /// located around the point of interest. In other words, this parameter
  /// ensures that the calculated values will not be extrapolated.
  /// @return A pair containing the interpolated value and the number of
  /// neighbors used in the calculation.
  template <typename Strategy>
  auto radial_basis_function(const Point &point, const Strategy &strategy,
                             const math::RBF<promotion_t> &rbf,
                             coordinate_t radius, uint32_t k, bool within) const
      -> std::pair<promotion_t, uint32_t> {
    auto [coordinates, values] =
        within ? nearest_within(point, strategy, radius, k)
               : nearest(point, strategy, radius, k);
    if (values.size() == 0) {
      return std::make_pair(std::numeric_limits<promotion_t>::quiet_NaN(), 0);
    }
    auto xi = Eigen::Matrix<promotion_t, dimension_t::value, 1>();
    for (size_t ix = 0; ix < dimension_t::value; ++ix) {
      xi(ix, 0) = geometry::point::get(point, ix);
    }
    auto interpolated = rbf.interpolate(coordinates, values, xi);
    return std::make_pair(interpolated(0),
                          static_cast<uint32_t>(values.size()));
  }

  /// @overload radial_basis_function(const Point &, const Strategy &, const
  /// math::RBF<promotion_t> &, const coordinate_t, const uint32_t, const bool)
  ///
  /// Overload of the radial_basis_function method with the default strategy.
  auto radial_basis_function(const Point &point,
                             const math::RBF<promotion_t> &rbf,
                             coordinate_t radius, uint32_t k, bool within) const
      -> std::pair<promotion_t, uint32_t> {
    return radial_basis_function(point, boost::geometry::default_strategy(),
                                 rbf, radius, k, within);
  }

  /// Interpolate the value of a point using a Window Function using the given
  /// strategy.
  ///
  /// @param point Point of interest
  /// @param strategy strategy used to compute the distance.
  /// @param wf The window function to be used.
  /// @param radius The maximum radius of the search.
  /// @param k The number of nearest neighbors to be used for calculating the
  /// interpolated value.
  /// @param within If true, the method ensures that the neighbors found are
  /// located around the point of interest. In other words, this parameter
  /// ensures that the calculated values will not be extrapolated.
  /// @return A pair containing the interpolated value and the number of
  /// neighbors used in the calculation.
  template <typename Strategy>
  auto window_function(const Point &point, const Strategy &strategy,
                       const math::WindowFunction<coordinate_t> &wf,
                       const coordinate_t arg, coordinate_t radius, uint32_t k,
                       bool within) const -> std::pair<coordinate_t, uint32_t> {
    coordinate_t result = 0;
    coordinate_t total_weight = 0;

    auto nearest =
        within ? query_within(point, strategy, k) : query(point, strategy, k);
    uint32_t neighbors = 0;

    for (const auto &item : nearest) {
      const auto distance = item.first;

      auto wk = wf(static_cast<coordinate_t>(distance), radius, arg);
      total_weight += wk;
      result += item.second * wk;
      ++neighbors;
    }

    return total_weight != 0
               ? std::make_pair(
                     static_cast<coordinate_t>(result / total_weight),
                     neighbors)
               : std::make_pair(std::numeric_limits<coordinate_t>::quiet_NaN(),
                                static_cast<uint32_t>(0));
  }

  /// @overload window_function(const Point &, const Strategy &, const
  /// math::WindowFunction<coordinate_t> &, const coordinate_t, const
  /// coordinate_t, const uint32_t, const bool)
  ///
  /// Overload of the window_function method with the default strategy.
  auto window_function(const Point &point,
                       const math::WindowFunction<coordinate_t> &wf,
                       const coordinate_t arg, coordinate_t radius, uint32_t k,
                       bool within) const -> std::pair<coordinate_t, uint32_t> {
    return window_function(point, boost::geometry::default_strategy(), wf, arg,
                           radius, k, within);
  }

 protected:
  /// Geographic index used to store data and their searches.
  std::shared_ptr<rtree_t> tree_;
};

}  // namespace pyinterp::detail::geometry

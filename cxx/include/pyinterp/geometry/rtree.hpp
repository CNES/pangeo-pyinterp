// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/convex_hull.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/multi_point.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/index/parameters.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/strategies/default_strategy.hpp>
#include <concepts>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/point_traits.hpp"
#include "pyinterp/math/interpolate/kriging.hpp"
#include "pyinterp/math/interpolate/rbf.hpp"
#include "pyinterp/math/interpolate/window_function.hpp"
#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::geometry {

/// Defines how the validity of the neighborhood is checked.
enum class BoundaryCheck : uint8_t {
  /// No verification is performed.
  kNone,
  /// Checks if the point is within the Axis Aligned Bounding Box (AABB) of the
  /// neighbors.
  kEnvelope,
  /// Checks if the point is within the Convex Hull of the neighbors (most
  /// restrictive and geometrically accurate).
  kConvexHull
};

/// Spatial index for point data with various interpolation methods.
///
/// @tparam Point Boost.Geometry point type
/// @tparam Type Value type associated with each point
template <typename Point, std::floating_point Type>
class RTree {
 public:
  using dimension_t = typename boost::geometry::traits::dimension<Point>;
  static_assert(dimension_t::value == 2 || dimension_t::value == 3,
                "Only 2D and 3D points are supported");

  /// Type of point coordinates
  using coordinate_t =
      typename boost::geometry::traits::coordinate_type<Point>::type;

  // Use the correct strategy type for distances
  using default_strategy_t = typename boost::geometry::default_strategy;

  /// Type of distance function result
  using distance_t =
      typename boost::geometry::default_distance_result<Point, Point>::type;

  /// Type of query results (distance, value)
  using result_t = std::pair<distance_t, Type>;

  /// Value handled by this object (point, value)
  using value_t = std::pair<Point, Type>;

  // R*-tree with fanout 16
  using rtree_t =
      boost::geometry::index::rtree<value_t, boost::geometry::index::rstar<16>>;

  /// Type promotion for mixed coordinate/value arithmetic
  using promotion_t =
      decltype(std::declval<coordinate_t>() + std::declval<Type>());

  /// Default constructor
  RTree() = default;

  /// Destructor
  virtual ~RTree() = default;

  /// Copy constructor
  RTree(const RTree&) = default;
  /// Copy assignment operator
  auto operator=(const RTree&) -> RTree& = default;

  /// Move constructor
  RTree(RTree&&) noexcept = default;
  /// Move assignment operator
  auto operator=(RTree&&) noexcept -> RTree& = default;

  /// Returns the bounding box containing all stored values.
  /// @return The bounding box or nullopt if the container is empty
  [[nodiscard]] virtual auto bounds() const
      -> std::optional<typename rtree_t::bounds_type> {
    if (empty()) [[unlikely]] {
      return std::nullopt;
    }
    return tree_.bounds();
  }

  /// Returns the number of points in the tree
  [[nodiscard]] constexpr auto size() const noexcept -> size_t {
    return tree_.size();
  }

  /// Query if the container is empty
  [[nodiscard]] constexpr auto empty() const noexcept -> bool {
    return tree_.empty();
  }

  /// Removes all values from the container
  auto clear() -> void { tree_.clear(); }

  /// Rebuild the tree using packing algorithm (erases old data)
  /// @param[in] points Vector of (point, value) pairs
  auto packing(const std::vector<value_t>& points) -> void {
    tree_ = rtree_t(points);
  }

  /// Insert new data into the search tree
  /// @param value (point, value) pair to insert
  auto insert(const value_t& value) -> void { tree_.insert(value); }

  /// Search for the K nearest neighbors.
  /// @tparam Strategy Strategy to calculate distances
  /// @param[in] point Point of interest
  /// @param[in] k Number of nearest neighbors
  /// @param[in] radius Maximum search distance
  /// @param[in] check Type of boundary verification to apply
  /// @param[in] strategy Strategy to calculate distances
  /// @return Vector of (distance, value) pairs
  template <typename Strategy = default_strategy_t>
  [[nodiscard]] auto query(const Point& point, const uint32_t k,
                           const coordinate_t& radius,
                           const BoundaryCheck check = BoundaryCheck::kNone,
                           const Strategy& strategy = Strategy()) const
      -> std::vector<result_t>;

  /// Search for neighbors within a radius.
  /// @tparam Strategy Strategy to calculate distances
  /// @param[in] point Point of interest
  /// @param[in] strategy Strategy to calculate distances
  /// @param[in] radius Maximum search distance
  /// @return Vector of (distance, value) pairs within radius
  template <typename Strategy = default_strategy_t>
  [[nodiscard]] auto query_ball(const Point& point, const coordinate_t radius,
                                const Strategy& strategy = Strategy()) const
      -> std::vector<result_t>;

  /// Get K nearest neighbors, optionally filtered by radius and boundary check.
  /// @tparam Strategy Strategy to calculate distances
  /// @param[in] point Point of interest
  /// @param[in] strategy Strategy to calculate distances
  /// @param[in] radius Optional maximum search distance
  /// @param[in] k Number of nearest neighbors
  /// @param[in] within If true, ensure point is surrounded by neighbors
  /// @return Vector of (point, value) pairs
  template <typename Strategy = default_strategy_t>
  [[nodiscard]] auto value(const Point& point, const coordinate_t& radius,
                           const uint32_t k, const BoundaryCheck check,
                           const Strategy& strategy = Strategy()) const
      -> std::vector<value_t>;

  /// Inverse Distance Weighting interpolation.
  /// @tparam Strategy Strategy to calculate distances
  /// @param[in] point Point of interest
  /// @param[in] radius Maximum search distance
  /// @param[in] k Number of nearest neighbors
  /// @param[in] p Power parameter
  /// @param[in] check Type of boundary verification to apply
  /// @param[in] strategy Strategy to calculate distances
  /// @return Pair of (interpolated value, number of neighbors used)
  template <typename Strategy = default_strategy_t>
  [[nodiscard]] auto inverse_distance_weighting(
      const Point& point, const coordinate_t& radius, const uint32_t k,
      const uint32_t p, const BoundaryCheck check,
      const Strategy& strategy = Strategy()) const
      -> std::pair<coordinate_t, uint32_t>;

  /// Kriging interpolation.
  /// @param[in] model Kriging model
  /// @param[in] point Point of interest
  /// @param[in] radius Maximum search distance
  /// @param[in] k Number of nearest neighbors
  /// @param[in] check Type of boundary verification to apply
  /// @return Pair of (interpolated value, number of neighbors used)
  template <typename Strategy = default_strategy_t>
  [[nodiscard]] auto kriging(
      const math::interpolate::Kriging<promotion_t>& model, const Point& point,
      const coordinate_t& radius, const uint32_t k, const BoundaryCheck check,
      const Strategy& strategy = Strategy()) const
      -> std::pair<coordinate_t, uint32_t>;

  /// Radial Basis Function interpolation.
  /// @tparam Strategy Strategy to calculate distances
  /// @param[in] model Radial basis function
  /// @param[in] point Point of interest
  /// @param[in] strategy Strategy to calculate distances
  /// @param[in] radius Maximum search distance
  /// @param[in] k Number of nearest neighbors
  /// @param[in] check Type of boundary verification to apply
  /// @return Pair of (interpolated value, number of neighbors used)
  template <typename Strategy = default_strategy_t>
  [[nodiscard]] auto radial_basis_function(
      const math::interpolate::RBF<promotion_t>& model, const Point& point,
      const coordinate_t& radius, const uint32_t k, const BoundaryCheck check,
      const Strategy& strategy = Strategy()) const
      -> std::pair<coordinate_t, uint32_t>;

  /// Window Function interpolation.
  /// @tparam Strategy Strategy to calculate distances
  /// @param[in] model Window function
  /// @param[in] point Point of interest
  /// @param[in] strategy Strategy to calculate distances
  /// @param[in] radius Maximum search distance
  /// @param[in] k Number of nearest neighbors
  /// @param[in] check Type of boundary verification to apply
  /// @return Pair of (interpolated value, number of neighbors used)
  template <typename Strategy = default_strategy_t>
  [[nodiscard]] auto window_function(
      const math::interpolate::InterpolationWindow<coordinate_t>& model,
      const Point& point, const coordinate_t& radius, const uint32_t k,
      const BoundaryCheck check, const Strategy& strategy = Strategy()) const
      -> std::pair<coordinate_t, uint32_t>;

  /// Get K nearest neighbors with their coordinates and values.
  /// @tparam Strategy Strategy to calculate distances
  /// @param[in] point Point of interest
  /// @param[in] strategy Strategy to calculate distances
  /// @param[in] radius Maximum search distance
  /// @param[in] k Number of nearest neighbors
  /// @param[in] check Type of boundary verification to apply
  /// @return Pair of (coordinates matrix, values vector)
  template <typename Strategy = default_strategy_t>
  [[nodiscard]] auto nearest(const Point& point, const coordinate_t& radius,
                             const uint32_t k, const BoundaryCheck check,
                             const Strategy& strategy = Strategy()) const
      -> std::tuple<Matrix<promotion_t>, Vector<promotion_t>>;

  /// Serialize the RTree state for storage or transmission.
  /// @return Serialized state as a Writer object
  [[nodiscard]] virtual auto pack() const -> serialization::Writer;

  /// Deserialize an RTree from serialized state.
  /// @param[in] state Reference to serialization Reader containing encoded
  /// RTree data
  /// @return New RTree instance with restored properties
  [[nodiscard]] static auto unpack(serialization::Reader& state)
      -> RTree<Point, Type>;

 protected:
  rtree_t tree_;

 private:
  /// Magic number for RTree serialization
  static constexpr uint32_t kMagicNumber = 0x52545452 + dimension_t::value;

  /// Verifies if the point satisfies the requested boundary condition.
  [[nodiscard]] auto is_boundary_valid(
      const Point& point,
      const boost::geometry::model::multi_point<Point>& points,
      const BoundaryCheck check) const -> bool;
};

// ============================================================================
// Implementation
// ============================================================================

template <typename Point, std::floating_point Type>
template <typename Strategy>
auto RTree<Point, Type>::query(const Point& point, const uint32_t k,
                               const coordinate_t& radius,
                               const BoundaryCheck check,
                               const Strategy& strategy) const
    -> std::vector<result_t> {
  std::vector<result_t> result;
  result.reserve(k);

  boost::geometry::model::multi_point<Point> points;
  if (check != BoundaryCheck::kNone) {
    points.reserve(k);
  }

  auto query_range = tree_.qbegin(boost::geometry::index::nearest(point, k));
  std::ranges::for_each(query_range, tree_.qend(), [&](const auto& item) {
    auto distance = boost::geometry::distance(point, item.first, strategy);
    if (distance <= radius) {
      result.emplace_back(distance, item.second);
      if (check != BoundaryCheck::kNone) {
        points.emplace_back(item.first);
      }
    }
  });

  if (!is_boundary_valid(point, points, check)) {
    return {};
  }
  return result;
}

// ============================================================================

template <typename Point, std::floating_point Type>
template <typename Strategy>
auto RTree<Point, Type>::query_ball(const Point& point,
                                    const coordinate_t radius,
                                    const Strategy& strategy) const
    -> std::vector<result_t> {
  std::vector<result_t> result;

  auto satisfies_radius = [&](const auto& item) {
    return boost::geometry::distance(item.first, point) <= radius;
  };

  auto query_range =
      tree_.qbegin(boost::geometry::index::satisfies(satisfies_radius));

  std::ranges::for_each(query_range, tree_.qend(), [&](const auto& item) {
    // Calculate exact distance using the provided strategy
    result.emplace_back(boost::geometry::distance(point, item.first, strategy),
                        item.second);
  });

  return result;
}

// ============================================================================

template <typename Point, std::floating_point Type>
template <typename Strategy>
auto RTree<Point, Type>::value(const Point& point, const coordinate_t& radius,
                               const uint32_t k, const BoundaryCheck check,
                               const Strategy& strategy) const
    -> std::vector<value_t> {
  std::vector<value_t> result;
  result.reserve(k);

  boost::geometry::model::multi_point<Point> points;
  if (check != BoundaryCheck::kNone) {
    points.reserve(k);
  }

  auto query_range = tree_.qbegin(boost::geometry::index::nearest(point, k));
  std::ranges::for_each(query_range, tree_.qend(), [&](const auto& item) {
    if (boost::geometry::distance(item.first, point, strategy) <= radius) {
      result.emplace_back(item);
      if (check != BoundaryCheck::kNone) {
        points.emplace_back(item.first);
      }
    }
  });

  if (!is_boundary_valid(point, points, check)) {
    return {};
  }
  return result;
}

// ============================================================================

template <typename Point, std::floating_point Type>
template <typename Strategy>
auto RTree<Point, Type>::inverse_distance_weighting(
    const Point& point, const coordinate_t& radius, const uint32_t k,
    const uint32_t p, const BoundaryCheck check, const Strategy& strategy) const
    -> std::pair<coordinate_t, uint32_t> {
  constexpr coordinate_t epsilon = coordinate_t{1e-6};
  coordinate_t result{0};
  coordinate_t total_weight{0};

  // Filter by radius directly within query
  const auto nearest = query(point, k, radius, check, strategy);
  const auto neighbors = static_cast<uint32_t>(nearest.size());

  for (const auto& [distance, value] : nearest) {
    if (distance < epsilon) [[unlikely]] {
      return {static_cast<coordinate_t>(value), k};
    }

    const auto weight = static_cast<Type>(
        Type{1} / std::pow(static_cast<Type>(distance), static_cast<Type>(p)));
    total_weight += weight;
    result += value * weight;
  }

  return total_weight != coordinate_t{0}
             ? std::pair{static_cast<coordinate_t>(result / total_weight),
                         neighbors}
             : std::pair{std::numeric_limits<coordinate_t>::quiet_NaN(),
                         uint32_t{0}};
}

// ============================================================================

template <typename Point, std::floating_point Type>
template <typename Strategy>
auto RTree<Point, Type>::kriging(
    const math::interpolate::Kriging<promotion_t>& model, const Point& point,
    const coordinate_t& radius, const uint32_t k, const BoundaryCheck check,
    const Strategy& strategy) const -> std::pair<coordinate_t, uint32_t> {
  const auto [coords, values] = nearest(point, radius, k, check, strategy);

  if (values.size() == 0) [[unlikely]] {
    return {std::numeric_limits<promotion_t>::quiet_NaN(), uint32_t{0}};
  }

  // Kriging requires 3D coordinates
  if constexpr (dimension_t::value == 3) {
    const Eigen::Vector3<promotion_t> point_3d(boost::geometry::get<0>(point),
                                               boost::geometry::get<1>(point),
                                               boost::geometry::get<2>(point));
    return {model(coords, values, point_3d),
            static_cast<uint32_t>(coords.cols())};
  } else {
    Eigen::Matrix<promotion_t, 3, Eigen::Dynamic> coords_3d(3, coords.cols());
    coords_3d.template topRows<2>() = coords;
    coords_3d.row(2).setZero();

    const Eigen::Vector3<promotion_t> point_3d(boost::geometry::get<0>(point),
                                               boost::geometry::get<1>(point),
                                               promotion_t{0});
    return {model(coords_3d, values, point_3d),
            static_cast<uint32_t>(coords_3d.cols())};
  }
}

// ============================================================================

template <typename Point, std::floating_point Type>
template <typename Strategy>
auto RTree<Point, Type>::radial_basis_function(
    const math::interpolate::RBF<promotion_t>& model, const Point& point,
    const coordinate_t& radius, const uint32_t k, const BoundaryCheck check,
    const Strategy& strategy) const -> std::pair<coordinate_t, uint32_t> {
  const auto [coordinates, values] = nearest(point, radius, k, check, strategy);

  if (values.size() == 0) [[unlikely]] {
    return {std::numeric_limits<promotion_t>::quiet_NaN(), uint32_t{0}};
  }

  auto xi = Eigen::Matrix<promotion_t, dimension_t::value, 1>();
  for (size_t ix = 0; ix < dimension_t::value; ++ix) {
    xi(ix, 0) = geometry::point::get(point, ix);
  }

  const auto interpolated = model.interpolate(coordinates, values, xi);
  return {interpolated(0), static_cast<uint32_t>(values.size())};
}

// ============================================================================

template <typename Point, std::floating_point Type>
template <typename Strategy>
auto RTree<Point, Type>::window_function(
    const math::interpolate::InterpolationWindow<coordinate_t>& model,
    const Point& point, const coordinate_t& radius, const uint32_t k,
    const BoundaryCheck check, const Strategy& strategy) const
    -> std::pair<coordinate_t, uint32_t> {
  coordinate_t result{0};
  coordinate_t total_weight{0};

  const auto nearest = query(point, k, radius, check, strategy);
  const auto neighbors = static_cast<uint32_t>(nearest.size());

  // Calculate furthest neighbor for the window scaling.
  // If nearest is not empty, the last element is guaranteed <= radius
  // because query() handles the filtering.
  const coordinate_t furthest_neighbor =
      radius == std::numeric_limits<coordinate_t>::max()
          ? (nearest.empty() ? coordinate_t{0}
                             : static_cast<coordinate_t>(nearest.back().first))
          : radius;

  for (const auto& [distance, value] : nearest) {
    const auto weight =
        model(static_cast<coordinate_t>(distance), furthest_neighbor);
    total_weight += weight;
    result += value * weight;
  }

  return total_weight != coordinate_t{0}
             ? std::pair{static_cast<coordinate_t>(result / total_weight),
                         neighbors}
             : std::pair{std::numeric_limits<coordinate_t>::quiet_NaN(),
                         uint32_t{0}};
}

// ============================================================================

template <typename Point, std::floating_point Type>
template <typename Strategy>
auto RTree<Point, Type>::nearest(const Point& point, const coordinate_t& radius,
                                 const uint32_t k, const BoundaryCheck check,
                                 const Strategy& strategy) const
    -> std::tuple<Matrix<promotion_t>, Vector<promotion_t>> {
  boost::geometry::model::multi_point<Point> points;
  if (check != BoundaryCheck::kNone) {
    points.reserve(k);
  }

  auto coordinates = Matrix<promotion_t>(dimension_t::value, k);
  auto values = Vector<promotion_t>(k);
  uint32_t count{0};

  auto query_range = tree_.qbegin(boost::geometry::index::nearest(point, k));
  std::ranges::for_each(query_range, tree_.qend(), [&](const auto& item) {
    if (boost::geometry::distance(point, item.first, strategy) <= radius) {
      if (check != BoundaryCheck::kNone) {
        points.emplace_back(item.first);
      }

      for (size_t ix = 0; ix < dimension_t::value; ++ix) {
        coordinates(ix, count) = geometry::point::get(item.first, ix);
      }
      values(count++) = item.second;
    }
  });

  if (check != BoundaryCheck::kNone &&
      !is_boundary_valid(point, points, check)) {
    count = 0;
  }

  coordinates.conservativeResize(dimension_t::value, count);
  values.conservativeResize(count);
  return {coordinates, values};
}

// ============================================================================

template <typename Point, std::floating_point Type>
auto RTree<Point, Type>::is_boundary_valid(
    const Point& point,
    const boost::geometry::model::multi_point<Point>& points,
    const BoundaryCheck check) const -> bool {
  if (check == BoundaryCheck::kNone) {
    return true;
  }

  if (points.empty()) {
    return false;
  }

  if (check == BoundaryCheck::kEnvelope) {
    // Fast: Axis Aligned Bounding Box
    auto box =
        boost::geometry::return_envelope<boost::geometry::model::box<Point>>(
            points);
    return boost::geometry::covered_by(point, box);

  } else if (check == BoundaryCheck::kConvexHull) {
    // Slow but accurate: Convex Hull
    boost::geometry::model::polygon<Point> hull;
    boost::geometry::convex_hull(points, hull);
    return boost::geometry::covered_by(point, hull);
  }

  return true;
}

// ============================================================================

template <typename Point, std::floating_point Type>
auto RTree<Point, Type>::pack() const -> serialization::Writer {
  serialization::Writer buffer;
  // Write magic number for validation
  buffer.write(kMagicNumber);
  // Serialize the number of points as a size_t
  const auto num_points = static_cast<size_t>(tree_.size());
  buffer.write(num_points);
  // Serialize each point and its associated value
  for (const auto& item : tree_) {
    // Serialize point coordinates
    for (size_t dim = 0; dim < dimension_t::value; ++dim) {
      buffer.write(geometry::point::get(item.first, dim));
    }
    // Serialize the associated value
    buffer.write(item.second);
  }

  return buffer;
}

// ============================================================================

template <typename Point, std::floating_point Type>
auto RTree<Point, Type>::unpack(serialization::Reader& state)
    -> RTree<Point, Type> {
  if (state.size() < sizeof(uint32_t) + sizeof(size_t)) {
    throw std::invalid_argument("Cannot restore RTree from incomplete state.");
  }
  const auto magic_number = state.read<uint32_t>();
  if (magic_number != kMagicNumber) {
    throw std::invalid_argument(
        "Invalid magic number for RTree serialization.");
  }

  const auto num_points = state.read<size_t>();

  std::vector<value_t> points;
  points.reserve(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    Point point;
    for (size_t dim = 0; dim < dimension_t::value; ++dim) {
      const auto coord = state.read<coordinate_t>();
      geometry::point::set(point, coord, dim);
    }
    const auto value = state.read<Type>();
    points.emplace_back(point, value);
  }
  RTree<Point, Type> rtree;
  rtree.packing(points);
  return rtree;
}

}  // namespace pyinterp::geometry

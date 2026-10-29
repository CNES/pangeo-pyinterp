// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <array>
#include <boost/geometry.hpp>
#include <boost/geometry/index/detail/rtree/utilities/view.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace pyinterp::geometry::detail {

/// @brief Ranks the neighbors by Euclidean distance.
///
/// The keys are the comparable (squared) distances computed by Boost, as in
/// its own nearest query: both searches select the same neighbors.
template <typename Point>
struct EuclideanRanking {
  /// Type of the keys
  using key_t =
      typename boost::geometry::default_comparable_distance_result<Point,
                                                                   Point>::type;

  /// Point of interest
  const Point& point;

  /// Lower bound of the keys of the points contained in a box
  template <typename Box>
  [[nodiscard]] auto bound(const Box& box) const -> key_t {
    return static_cast<key_t>(boost::geometry::comparable_distance(point, box));
  }

  /// Key of a point; a key not smaller than @p limit is not used.
  [[nodiscard]] auto key(const Point& other, const key_t& /*limit*/) const
      -> key_t {
    return static_cast<key_t>(
        boost::geometry::comparable_distance(point, other));
  }
};

/// @brief Ranks the neighbors by a distance never smaller than the Euclidean
/// distance multiplied by `1 - Strategy::kMargin` (see EcefGeodesic).
template <typename Point, typename Strategy, typename Distance>
struct ChordBoundedRanking {
  /// Type of the keys
  using key_t = Distance;

  /// Point of interest
  const Point& point;
  /// Strategy computing the distances
  const Strategy& strategy;

  /// Lower bound of the keys of the points contained in a box
  template <typename Box>
  [[nodiscard]] auto bound(const Box& box) const -> key_t {
    return static_cast<key_t>(boost::geometry::distance(point, box) *
                              (1 - Strategy::kMargin));
  }

  /// Key of a point; the exact distance is not computed when its lower bound
  /// already exceeds @p limit.
  [[nodiscard]] auto key(const Point& other, const key_t& limit) const
      -> key_t {
    if (boost::geometry::distance(point, other) * (1 - Strategy::kMargin) >
        limit) {
      return limit;
    }
    return static_cast<key_t>(strategy(point, other));
  }
};

/// @brief Depth-first search of the k nearest neighbors in a Boost R-tree.
///
/// Boost's nearest query runs a best-first search that allocates its queues
/// for every point. This visitor descends into the children of a node in
/// increasing order of distance, skips those that cannot hold a closer
/// neighbor, and keeps the neighbors found in a buffer supplied by the
/// caller, sorted by increasing key. Neighbors with equal keys are kept in
/// the order they are found.
///
/// @tparam MembersHolder Internals of the Boost R-tree
/// @tparam Ranking Ranking of the neighbors (EuclideanRanking or
/// ChordBoundedRanking)
template <typename MembersHolder, typename Ranking>
class NearestVisitor : public MembersHolder::visitor_const {
 public:
  using value_type = typename MembersHolder::value_type;
  using internal_node = typename MembersHolder::internal_node;
  using leaf = typename MembersHolder::leaf;
  using node_pointer = typename MembersHolder::node_pointer;
  using key_t = typename Ranking::key_t;
  /// A neighbor found: (key, value)
  using neighbor_t = std::pair<key_t, const value_type*>;

  /// @param[in] ranking Ranking of the neighbors
  /// @param[in] k Number of neighbors searched (> 0)
  /// @param[out] neighbors Buffer of @p k items receiving the neighbors
  NearestVisitor(const Ranking& ranking, const uint32_t k,
                 neighbor_t* neighbors)
      : ranking_(ranking), k_(k), neighbors_(neighbors) {}

  /// Number of neighbors found
  [[nodiscard]] auto size() const noexcept -> uint32_t { return size_; }

  /// Visits the children of an internal node, closest first.
  void operator()(const internal_node& node) {
    std::array<std::pair<key_t, node_pointer>, kMaxElements> children;
    size_t count = 0;
    for (const auto& [box, child] :
         boost::geometry::index::detail::rtree::elements(node)) {
      const auto bound = ranking_.bound(box);
      if (pruned(bound)) {
        continue;
      }
      auto ix = count++;
      for (; ix > 0 && children[ix - 1].first > bound; --ix) {
        children[ix] = children[ix - 1];
      }
      children[ix] = {bound, child};
    }
    for (size_t ix = 0; ix < count && !pruned(children[ix].first); ++ix) {
      boost::geometry::index::detail::rtree::apply_visitor(
          *this, *children[ix].second);
    }
  }

  /// Inserts the values of a leaf closer than the neighbors found.
  void operator()(const leaf& node) {
    for (const auto& value :
         boost::geometry::index::detail::rtree::elements(node)) {
      const auto full = size_ == k_;
      const auto key = full
                           ? ranking_.key(value.first, neighbors_[k_ - 1].first)
                           : ranking_.key(value.first, kInfinity);
      uint32_t ix;
      if (!full) {
        ix = size_++;
      } else if (key < neighbors_[k_ - 1].first) {
        ix = k_ - 1;
      } else {
        continue;
      }
      for (; ix > 0 && neighbors_[ix - 1].first > key; --ix) {
        neighbors_[ix] = neighbors_[ix - 1];
      }
      neighbors_[ix] = {key, std::addressof(value)};
    }
  }

 private:
  /// Maximum number of children of a node
  static constexpr size_t kMaxElements =
      MembersHolder::parameters_type::max_elements + 1;
  /// Key larger than any other
  static constexpr key_t kInfinity = std::numeric_limits<key_t>::infinity();

  const Ranking& ranking_;
  uint32_t k_;
  neighbor_t* neighbors_;
  uint32_t size_{0};

  /// True if a node whose keys are not smaller than @p bound cannot hold a
  /// closer neighbor.
  [[nodiscard]] auto pruned(const key_t& bound) const noexcept -> bool {
    return size_ == k_ && neighbors_[k_ - 1].first <= bound;
  }
};

/// @brief Searches the k nearest neighbors of a point in a Boost R-tree.
///
/// @param[in] tree The R-tree
/// @param[in] ranking Ranking of the neighbors
/// @param[in] k Number of neighbors searched (> 0)
/// @param[out] neighbors Buffer of @p k items receiving the (key, value)
/// pairs, sorted by increasing key
/// @return The number of neighbors found
template <typename RTree, typename Ranking, typename Neighbor>
auto nearest(const RTree& tree, const Ranking& ranking, const uint32_t k,
             Neighbor* neighbors) -> uint32_t {
  using view_t = boost::geometry::index::detail::rtree::utilities::view<RTree>;
  using visitor_t = NearestVisitor<typename view_t::members_holder, Ranking>;
  static_assert(std::is_same_v<Neighbor, typename visitor_t::neighbor_t>);
  auto visitor = visitor_t(ranking, k, neighbors);
  view_t(tree).apply_visitor(visitor);
  return visitor.size();
}

}  // namespace pyinterp::geometry::detail

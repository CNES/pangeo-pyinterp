#pragma once
#include "pyinterp/detail/geometry/rtree.hpp"

namespace pyinterp {
namespace detail {
namespace cartesian {

/// Index points in the Cartesian space at N dimensions.
///
/// @tparam Coordinate The class of storage for a point's coordinates.
/// @tparam Type The type of data stored in the tree.
/// @tparam N Number of dimensions in the Cartesian space handled.
template <typename Coordinate, typename Type, size_t N>
class RTree : public geometry::RTree<Coordinate, Type, N> {
 public:
  using geometry::RTree<Coordinate, Type, N>::RTree;

  /// Type of query results.
  using result_t = typename geometry::RTree<Coordinate, Type, N>::result_t;

  /// Inserting data into the tree using the packaging algorithm (the old data
  /// is deleted before construction).
  ///
  /// @param coordinates Coordinates to be inserted in the tree.
  /// @param values Values associated with the different coordinates to be
  /// inserted in the tree.
  void packing(
      const Eigen::Matrix<Coordinate, Eigen::Dynamic, Eigen::Dynamic>
          &coordinates,
      const Eigen::Ref<const Eigen::Matrix<Type, Eigen::Dynamic, 1>> &values) {
    auto size = coordinates.rows();
    auto points = std::vector<typename RTree<Coordinate, Type>::value_t>();
    auto point = geometry::PointND<Coordinate, N>();

    points.reserve(size);

    for (auto ix = 0; ix < size; ++ix) {
      for (auto dim = 0LL; dim < N; ++dim) {
        geometry::point::set(point, coordinates(ix, dim), dim);
      }
      points.emplace_back(std::make_pair(point, values(ix)));
    }
    geometry::RTree<Coordinate, Type, N>::packing(points);
  }

  /// Inserting data into the tree.
  ///
  /// @param coordinates Coordinates to be inserted in the tree.
  /// @param values Values associated with the different coordinates to be
  /// inserted in the tree.
  void insert(
      const Eigen::Matrix<Coordinate, Eigen::Dynamic, Eigen::Dynamic>
          &coordinates,
      const Eigen::Ref<const Eigen::Matrix<Type, Eigen::Dynamic, 1>> &values) {
    auto size = coordinates.rows();
    auto point = geometry::PointND<Coordinate, N>();

    for (auto ix = 0; ix < size; ++ix) {
      for (auto dim = 0ULL; dim < N; ++dim) {
        geometry::point::set(point, coordinates(ix, dim), dim);
      }
      geometry::RTree<Coordinate, Type, N>::insert(point);
    }
  }

  /// Search for the nearest K nearest neighbors of a given point.
  ///
  /// @param coordinates Matrix describing the coordinates of the points to be
  /// searched.
  /// @param k The maximum number of neighbors to search for.
  /// @param within If true, the method ensures that the neighbors found are
  ///   located within the point of interest
  /// @param num_threads The number of threads to use for the computation
  std::tuple<Eigen::Matrix<distance_t, Eigen::Dynamic, Eigen::Dynamic>,
             Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>>
  query(const Eigen::Ref<const Eigen::Matrix<Coordinate, Eigen::Dynamic,
                                             Eigen::Dynamic>> &coordinates,
        const uint32_t k, const bool within, const size_t num_threads) {
    // Signature of the function of the base class to be called.
    using query_t = std::vector<result_t> (RTree::*)(
        const geometry::PointND<Coordinate, N> &, uint32_t) const;

    // Selection of the method performing the calculation.
    const std::function<std::vector<result_t>(
        const RTree &, const geometry::PointND<Coordinate, N> &, uint32_t)>
        method =
            within ? &RTree::query_within : static_cast<query_t>(&RTree::query);

    auto size = coordinates.rows();
    auto distance =
        Eigen::Matrix<distance_t, Eigen::Dynamic, Eigen::Dynamic>(size, k);
    auto value = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>(size, k);
    auto point = geometry::PointND<Coordinate, N>();

    // Captures the detected exceptions in the calculation function
    // (only the last exception captured is kept)
    auto except = std::exception_ptr(nullptr);

    // Dispatch calculation on defined cores
    dispatch(
        [&](const size_t start, const size_t stop) {
          auto point = geometry::PointND<Coordinate, N>();
          try {
            for (auto ix = start; ix < stop; ++ix) {
              for (auto dim = 0ULL; dim < N; ++dim) {
                geometry::point::set(point, coordinates(ix, dim), dim);
              }

              auto nearest = method(*this, point, k);
              auto jx = 0ULL;

              // Fill in the calculation result for all neighbors found
              for (; jx < nearest.size(); ++jx) {
                distance(ix, jx) = std::get<0>(nearest[jx]);
                value(ix, jx) = std::get<1>(nearest[jx]);
              }

              // The rest of the result is filled with invalid values
              for (; jx < k; ++jx) {
                distance(ix, jx) = -1;
                value(ix, jx) = -1;
              }
            }
          } catch (...) {
            except = std::current_exception();
          }
        },
        size, num_threads);

    if (except != nullptr) {
      std::rethrow_exception(except);
    }
    return std::make_tuple(distance, value);
  }
};

}  // namespace cartesian
}  // namespace detail
}  // namespace pyinterp
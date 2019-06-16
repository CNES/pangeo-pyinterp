#pragma once
#include "pyinterp/detail/geometry/rtree.hpp"
#include "pyinterp/detail/geodetic/coordinates.hpp"
#include "Eigen/Core"
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
  /// Type of query results.
  using result_t = typename geometry::RTree<Coordinate, Type, 3>::result_t;

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

  /// Populates the RTree using the packaging algorithm
  ///
  /// @param coordinates Coordinates to be inserted in the tree.
  /// @param values Values associated with the different coordinates to be
  /// inserted in the tree.
  void packing(
      const Eigen::Ref<const Eigen::Matrix<Coordinate, Eigen::Dynamic,
                                           Eigen::Dynamic>> &coordinates,
      const Eigen::Ref<const Eigen::Matrix<Type, Eigen::Dynamic, 1>> &values) {
    if (coordinates.rows() != values.size()) {
      throw std::invalid_argument(
          "coordinates, values could not be broadcast together with shape (" +
          std::to_string(coordinates.rows()) + ", " +
          std::to_string(coordinates.cols()) + ") (" +
          std::to_string(values.size()) + ")");
    }
    switch (coordinates.cols()) {
      case 2:
        packing<2>(coordinates, values);
        break;
      case 3:
        packing<3>(coordinates, values);
        break;
      default:
        throw std::invalid_argument(
            "coordinates must be a matrix (n, 2) to add points defined by "
            "their longitudes and latitudes or a matrix (n, 3) to add points "
            "defined by their longitudes, latitudes and altitudes.");
    }
  }

  /// Insert new data into the RTree
  ///
  /// @param coordinates Coordinates to be inserted in the tree.
  /// @param values Values associated with the different coordinates to be
  /// inserted in the tree.
  void insert(
      const Eigen::Ref<const Eigen::Matrix<Coordinate, Eigen::Dynamic,
                                           Eigen::Dynamic>> &coordinates,
      const Eigen::Ref<const Eigen::Matrix<Type, Eigen::Dynamic, 1>> &values) {
    if (coordinates.rows() != values.size()) {
      throw std::invalid_argument(
          "coordinates, values could not be broadcast together with shape (" +
          std::to_string(coordinates.rows()) + ", " +
          std::to_string(coordinates.cols()) + ") (" +
          std::to_string(values.size()) + ")");
    }
    switch (coordinates.cols()) {
      case 2:
        insert<2>(coordinates, values);
        break;
      case 3:
        insert<3>(coordinates, values);
        break;
      default:
        throw std::invalid_argument(
            "coordinates must be a matrix (n, 2) to add points defined by "
            "their longitudes and latitudes or a matrix (n, 3) to add points "
            "defined by their longitudes, latitudes and altitudes.");
    }
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

 private:
  /// System for converting Geodetic coordinates into Cartesian coordinates.
  Coordinates coordinates_;

  /// Distance calculation formulae on lat/lon coordinates
  boost::geometry::strategy::distance::haversine<Coordinate> strategy_;

  /// Inserting data into the tree using the packaging algorithm (the old data
  /// is deleted before construction).
  ///
  /// @param coordinates Coordinates to be inserted in the tree.
  /// @param values Values associated with the different coordinates to be
  /// inserted in the tree.
  /// @tparam Dimensions Number of dimensions provided by the user: 2 if
  /// altitude is not specified, otherwise 3.
  template <size_t Dimensions>
  void packing(
      const Eigen::Matrix<Coordinate, Eigen::Dynamic, Eigen::Dynamic>
          &coordinates,
      const Eigen::Ref<const Eigen::Matrix<Type, Eigen::Dynamic, 1>> &values) {
    auto size = coordinates.rows();
    auto points = std::vector<typename RTree<Coordinate, Type>::value_t>();
    auto point = geometry::EquatorialPoint3D<Coordinate>();

    points.reserve(size);

    for (auto ix = 0; ix < size; ++ix) {
      auto dim = 0ULL;
      for (; dim < Dimensions; ++dim) {
        geometry::point::set(point, coordinates(ix, dim), dim);
      }
      for (; dim < 3ULL; ++dim) {
        geometry::point::set(point, Coordinate(0), dim);
      }
      points.emplace_back(
          std::make_pair(coordinates_.lla_to_ecef(point), values(ix)));
    }
    geometry::RTree<Coordinate, Type, 3>::packing(points);
  }

  /// Inserting data into the tree.
  ///
  /// @param coordinates Coordinates to be inserted in the tree.
  /// @param values Values associated with the different coordinates to be
  /// inserted in the tree.
  /// @tparam Dimensions Number of dimensions provided by the user: 2 if
  /// altitude is not specified, otherwise 3.
  template <size_t Dimensions>
  void insert(
      const Eigen::Matrix<Coordinate, Eigen::Dynamic, Eigen::Dynamic>
          &coordinates,
      const Eigen::Ref<const Eigen::Matrix<Type, Eigen::Dynamic, 1>> &values) {
    auto size = coordinates.rows();
    auto point = geometry::EquatorialPoint3D<Coordinate>();

    for (auto ix = 0; ix < size; ++ix) {
      auto dim = 0ULL;
      for (; dim < Dimensions; ++dim) {
        geometry::point::set(point, coordinates(ix, dim), dim);
      }
      for (; dim < 3ULL; ++dim) {
        geometry::point::set(point, Coordinate(0), dim);
      }
      geometry::RTree<Coordinate, Type, 3>::insert(
          coordinates_.lla_to_ecef(point));
    }
  }
};

}  // namespace geodetic
}  // namespace detail
}  // namespace pyinterp
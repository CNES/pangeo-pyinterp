// Copyright (c) 2022 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once
#include <optional>
#include <vector>

#include "pyinterp/detail/geometry/rtree.hpp"
#include "pyinterp/geodetic/box.hpp"

namespace pyinterp::geodetic {

/// RTree spatial index for geodetic point
class RTree : public detail::geometry::RTree<
                  detail::geometry::GeographicPoint2D<double>, double> {
 public:
  /// Point type
  using point_t = detail::geometry::GeographicPoint2D<double>;

  /// Base class
  using base_t = detail::geometry::RTree<point_t, double>;

  /// Type of point coordinates
  using coordinate_t = typename base_t::coordinate_t;

  /// Type of distance between two points
  using distance_t = typename base_t::distance_t;

  /// Type of query results.
  using result_t = typename base_t::result_t;

  /// Type of the implicit conversion between the type of coordinates and values
  using promotion_t = typename base_t::promotion_t;

  /// Strategy used to compute the distance between two points with the
  /// geodetic spheroid specified.
  using strategy_t = boost::geometry::strategy::distance::geographic<
      boost::geometry::strategy::andoyer,
      boost::geometry::srs::spheroid<coordinate_t>, void>;

  /// Pointer on the method to Search for the nearest K nearest neighbors
  using Requester = std::vector<result_t> (base_t::*)(const point_t &,
                                                      const strategy_t &,
                                                      const uint32_t) const;

  /// Default constructor
  RTree() = default;

  /// Create a new RTree with the specified spheroid.
  explicit RTree(const std::optional<detail::geodetic::Spheroid> &wgs);

  /// Returns the box able to contain all values stored in the container.
  [[nodiscard]] auto equatorial_bounds() const -> std::optional<Box>;

  /// Populates the RTree with coordinates using the packaging algorithm
  ///
  void packing(const Eigen::Ref<const Vector<double>> &lon,
               const Eigen::Ref<const Vector<double>> &lat,
               const Eigen::Ref<const Vector<double>> &values);

  /// Insert new data into the search tree
  ///
  void insert(const Eigen::Ref<const Vector<double>> &lon,
              const Eigen::Ref<const Vector<double>> &lat,
              const Eigen::Ref<const Vector<double>> &values);

  /// Search for the nearest K nearest neighbors of a given coordinates.
  [[nodiscard]] auto query(const Eigen::Ref<const Vector<double>> &lon,
                           const Eigen::Ref<const Vector<double>> &lat,
                           const uint32_t k, bool within,
                           size_t num_threads) const -> pybind11::tuple;

  [[nodiscard]] auto inverse_distance_weighting(
      const Eigen::Ref<const Vector<double>> &lon,
      const Eigen::Ref<const Vector<double>> &lat,
      const std::optional<double> &radius, uint32_t k, uint32_t p, bool within,
      size_t num_threads) const -> pybind11::tuple;

  [[nodiscard]] auto radial_basis_function(
      const Eigen::Ref<const Vector<double>> &lon,
      const Eigen::Ref<const Vector<double>> &lat,
      const std::optional<double> &radius, uint32_t k,
      detail::math::RadialBasisFunction rbf,
      const std::optional<double> &epsilon, double smooth, bool within,
      size_t num_threads) const -> pybind11::tuple;

  [[nodiscard]] auto window_function(
      const Eigen::Ref<const Vector<double>> &lon,
      const Eigen::Ref<const Vector<double>> &lat, double radius, uint32_t k,
      detail::math::window::Function wf, const std::optional<double> &arg,
      bool within, size_t num_threads) const -> pybind11::tuple;

  /// Get a tuple that fully encodes the state of this instance
  [[nodiscard]] auto getstate() const -> pybind11::tuple;

  /// Create a new instance from a registered state of an instance of this
  /// object.
  static auto setstate(const pybind11::tuple &state) -> RTree;

 private:
  /// Strategy used to compute the distance between two points.
  strategy_t strategy_{};
};

}  // namespace pyinterp::geodetic

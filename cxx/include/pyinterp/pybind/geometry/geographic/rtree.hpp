// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

#include "pyinterp/config/rtree.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/geographic/point.hpp"
#include "pyinterp/geometry/rtree.hpp"
#include "pyinterp/math/interpolate/rbf.hpp"
#include "pyinterp/math/interpolate/window_function.hpp"
#include "pyinterp/parallel_for.hpp"
#include "pyinterp/pybind/ndarray_serialization.hpp"
#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::geometry::geographic::pybind {

using pyinterp::pybind::NanobindArray1DUInt8;

/// RTree spatial index
///
/// This class provides spatial indexing for geographic coordinates using an
/// R-tree data structure. It expects longitude/latitude directly in degrees
/// and performs all calculations on the spheroid.
class RTree : public pyinterp::geometry::RTree<Point, double> {
 public:
  /// Scalar type for coordinates and values
  using value_type = double;

  /// Point type
  using point_t = Point;

  /// Base class type
  using base_t = pyinterp::geometry::RTree<point_t, value_type>;

  /// Coordinate type
  using coordinate_t = typename base_t::coordinate_t;

  /// Distance type between points
  using distance_t = typename base_t::distance_t;

  /// Query result type (distance, value, point)
  using result_t = typename base_t::result_t;

  /// Promoted type for arithmetic
  using promotion_t = typename base_t::promotion_t;

  /// Coordinate matrix type (n, 2)
  using CoordinateMatrix =
      Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;

  /// Value vector type
  using ValueVector = Vector<double>;

  /// Default constructor (WGS84 spheroid)
  RTree() = default;

  /// Bulk-load points using STR packing algorithm
  ///
  /// @param[in] coordinates Matrix of shape (n, 2) containing
  /// coordinates (lon, lat).
  /// @param[in] values Vector of size n containing values at each point
  void packing(const Eigen::Ref<const CoordinateMatrix>& coordinates,
               const Eigen::Ref<const ValueVector>& values);

  /// Insert points into the tree
  ///
  /// @param[in] coordinates Matrix of shape (n, 2) containing
  /// coordinates (lon, lat).
  /// @param[in] values Vector of size n containing values at each point
  void insert(const Eigen::Ref<const CoordinateMatrix>& coordinates,
              const Eigen::Ref<const ValueVector>& values);

  /// Query k-nearest neighbors for multiple points
  ///
  /// @param[in] coordinates Query coordinates, shape (n, 3) or (n, 2)
  /// @param[in] k Number of neighbors to find
  /// @param[in] check Type of boundary verification to apply
  /// @param[in] num_threads Number of threads (0 = auto)
  /// @return Tuple of (distances, values) matrices [n_points x k]
  [[nodiscard]] auto query(
      const Eigen::Ref<const CoordinateMatrix>& coordinates,
      const config::rtree::Query& config) const
      -> std::tuple<Matrix<distance_t>, Matrix<promotion_t>>;

  /// Inverse distance weighting interpolation
  ///
  /// @param[in] coordinates Query coordinates, shape (n, 3) or (n, 2)
  /// @param[in] config Configuration for IDW interpolation
  /// @return Tuple of (interpolated values, neighbor counts)
  [[nodiscard]] auto inverse_distance_weighting(
      const Eigen::Ref<const CoordinateMatrix>& coordinates,
      const config::rtree::InverseDistanceWeighting& config) const
      -> std::tuple<ValueVector, Vector<uint32_t>>;

  /// Kriging interpolation
  ///
  /// @param[in] coordinates Query coordinates, shape (n, 3) or (n, 2)
  /// @param[in] config Configuration for Kriging interpolation
  /// @return Tuple of (interpolated values, neighbor counts)
  [[nodiscard]] auto kriging(
      const Eigen::Ref<const CoordinateMatrix>& coordinates,
      const config::rtree::Kriging& config) const
      -> std::tuple<ValueVector, Vector<uint32_t>>;

  /// Radial basis function interpolation
  ///
  /// @param[in] coordinates Query coordinates, shape (n, 3) or (n, 2)
  /// @param[in] config Configuration for RBF interpolation
  /// @return Tuple of (interpolated values, neighbor counts)
  [[nodiscard]] auto radial_basis_function(
      const Eigen::Ref<const CoordinateMatrix>& coordinates,
      const config::rtree::RadialBasisFunction& config) const
      -> std::tuple<ValueVector, Vector<uint32_t>>;

  /// Window function based interpolation
  ///
  /// @param coordinates Query coordinates, shape (n, 3) or (n, 2)
  /// @param config Configuration for window function interpolation
  /// @return Tuple of (interpolated values, neighbor counts)
  [[nodiscard]] auto window_function(
      const Eigen::Ref<const CoordinateMatrix>& coordinates,
      const config::rtree::InterpolationWindow& config) const
      -> std::tuple<ValueVector, Vector<uint32_t>>;

  /// Serialize the RTree state
  /// @return Serialized byte buffer
  [[nodiscard]] auto getstate() const -> nanobind::tuple {
    serialization::Writer state;
    {
      nanobind::gil_scoped_release release;
      state = base_t::pack();
    }
    return nanobind::make_tuple(
        pyinterp::pybind::writer_to_ndarray(std::move(state)));
  }

  /// Deserialize an RTree3D from a byte buffer
  /// @param buffer Serialized data
  /// @return Deserialized RTree3D instance
  [[nodiscard]] static auto setstate(const nanobind::tuple& state) -> RTree {
    if (state.size() != 1) {
      throw std::invalid_argument("Invalid state");
    }
    auto array = nanobind::cast<NanobindArray1DUInt8>(state[0]);
    auto reader = pyinterp::pybind::reader_from_ndarray(array);
    {
      nanobind::gil_scoped_release release;
      RTree rtree;
      static_cast<base_t&>(rtree) = base_t::unpack(reader);
      return rtree;
    }
  }

 private:
  /// Validate coordinate matrix dimensions
  static void validate_coordinates(
      const Eigen::Ref<const CoordinateMatrix>& coordinates,
      const Eigen::Ref<const ValueVector>& values);

  /// Convert input coordinates to a geographic point.
  /// @param coordinates Input matrix (n, 2)
  /// @param row Row index
  /// @return Point
  [[nodiscard]] auto to_point(
      const Eigen::Ref<const CoordinateMatrix>& coordinates,
      Eigen::Index row) const -> point_t;

  /// Helper: perform batch query operation
  template <typename QueryFunc>
  [[nodiscard]] auto batch_query(
      const Eigen::Ref<const CoordinateMatrix>& coordinates,
      const config::rtree::Query& config, QueryFunc&& query_func) const
      -> std::tuple<Matrix<distance_t>, Matrix<promotion_t>>;

  /// Helper: perform batch interpolation operation
  template <typename InterpolateFunc>
  [[nodiscard]] auto batch_interpolate(
      const Eigen::Ref<const CoordinateMatrix>& coordinates, size_t num_threads,
      InterpolateFunc&& interpolate_func) const
      -> std::tuple<ValueVector, Vector<uint32_t>>;
};

// ==========================================================================
// Implementation details
// =========================================================================

void RTree::validate_coordinates(
    const Eigen::Ref<const CoordinateMatrix>& coordinates,
    const Eigen::Ref<const ValueVector>& values) {
  if (coordinates.rows() != values.size()) {
    throw std::invalid_argument(
        "Number of coordinates must match number of values");
  }
}

// ==========================================================================

auto RTree::to_point(const Eigen::Ref<const CoordinateMatrix>& coordinates,
                     Eigen::Index row) const -> point_t {
  const auto lon = coordinates(row, 0);
  const auto lat = coordinates(row, 1);

  return {lon, lat};
}

// ==========================================================================

void RTree::packing(const Eigen::Ref<const CoordinateMatrix>& coordinates,
                    const Eigen::Ref<const ValueVector>& values) {
  RTree::validate_coordinates(coordinates, values);

  std::vector<std::pair<point_t, double>> items;
  items.reserve(static_cast<size_t>(coordinates.rows()));

  for (int64_t ix = 0; ix < coordinates.rows(); ++ix) {
    items.emplace_back(RTree::to_point(coordinates, ix), values[ix]);
  }

  base_t::packing(items);
}

// ==========================================================================

void RTree::insert(const Eigen::Ref<const CoordinateMatrix>& coordinates,
                   const Eigen::Ref<const ValueVector>& values) {
  validate_coordinates(coordinates, values);

  for (int64_t ix = 0; ix < coordinates.rows(); ++ix) {
    base_t::insert({RTree::to_point(coordinates, ix), values[ix]});
  }
}

// ==========================================================================

template <typename QueryFunc>
auto RTree::batch_query(const Eigen::Ref<const CoordinateMatrix>& coordinates,
                        const config::rtree::Query& config,
                        QueryFunc&& query_func) const
    -> std::tuple<Matrix<distance_t>, Matrix<promotion_t>> {
  const auto n = coordinates.rows();
  Matrix<distance_t> distances(n, config.k());
  Matrix<promotion_t> values(n, config.k());

  // Initialize with sentinel values
  distances.setConstant(std::numeric_limits<distance_t>::quiet_NaN());
  values.setConstant(std::numeric_limits<promotion_t>::quiet_NaN());

  parallel_for(
      static_cast<size_t>(n),
      [&](size_t start, size_t end) {
        for (size_t idx = start; idx < end; ++idx) {
          const auto ix = static_cast<int64_t>(idx);
          auto results = query_func(to_point(coordinates, ix), config.k());

          for (size_t jx = 0; jx < results.size() && jx < config.k(); ++jx) {
            distances(ix, static_cast<int64_t>(jx)) = std::get<0>(results[jx]);
            values(ix, static_cast<int64_t>(jx)) = std::get<1>(results[jx]);
          }
        }
      },
      config.num_threads());

  return {std::move(distances), std::move(values)};
}

// ==========================================================================

template <typename InterpolateFunc>
auto RTree::batch_interpolate(
    const Eigen::Ref<const CoordinateMatrix>& coordinates, size_t num_threads,
    InterpolateFunc&& interpolate_func) const
    -> std::tuple<ValueVector, Vector<uint32_t>> {
  const auto n = coordinates.rows();
  ValueVector values(n);
  Vector<uint32_t> counts(n);

  values.setConstant(std::numeric_limits<promotion_t>::quiet_NaN());
  counts.setZero();

  parallel_for(
      static_cast<size_t>(n),
      [&](size_t start, size_t end) {
        for (size_t idx = start; idx < end; ++idx) {
          const auto ix = static_cast<int64_t>(idx);
          std::tie(values[ix], counts[ix]) =
              interpolate_func(to_point(coordinates, ix));
        }
      },
      num_threads);

  return {std::move(values), std::move(counts)};
}

// ==========================================================================

auto RTree::query(const Eigen::Ref<const CoordinateMatrix>& coordinates,
                  const config::rtree::Query& config) const
    -> std::tuple<Matrix<distance_t>, Matrix<promotion_t>> {
  return batch_query(coordinates, config,
                     [this, config](const point_t& pt, uint32_t neighbors) {
                       return base_t::query(pt, neighbors, config.radius(),
                                            config.boundary_check());
                     });
}

// =========================================================================

auto RTree::inverse_distance_weighting(
    const Eigen::Ref<const CoordinateMatrix>& coordinates,
    const config::rtree::InverseDistanceWeighting& config) const
    -> std::tuple<ValueVector, Vector<uint32_t>> {
  return batch_interpolate(
      coordinates, config.num_threads(),
      [this, config](const point_t& pt) -> std::pair<promotion_t, uint32_t> {
        return base_t::inverse_distance_weighting(pt, config.radius(),
                                                  config.k(), config.p(),
                                                  config.boundary_check());
      });
}

// =========================================================================

auto RTree::kriging(const Eigen::Ref<const CoordinateMatrix>& coordinates,
                    const config::rtree::Kriging& config) const
    -> std::tuple<ValueVector, Vector<uint32_t>> {
  auto model = math::interpolate::Kriging<promotion_t>(
      config.sigma(), config.lambda(), config.nugget(),
      config.covariance_model(), config.drift_function());
  return batch_interpolate(
      coordinates, config.num_threads(),
      [this, &model,
       &config](const point_t& pt) -> std::pair<promotion_t, uint32_t> {
        return base_t::kriging(model, pt, config.radius(), config.k(),
                               config.boundary_check());
      });
}

// =========================================================================

auto RTree::radial_basis_function(
    const Eigen::Ref<const CoordinateMatrix>& coordinates,
    const config::rtree::RadialBasisFunction& config) const
    -> std::tuple<ValueVector, Vector<uint32_t>> {
  auto model = math::interpolate::RBF<promotion_t>(
      config.epsilon(), config.smooth(), config.rbf());
  return batch_interpolate(
      coordinates, config.num_threads(),
      [this, &model,
       &config](const point_t& pt) -> std::pair<promotion_t, uint32_t> {
        return base_t::radial_basis_function(
            model, pt, config.radius(), config.k(), config.boundary_check());
      });
}

// =========================================================================

auto RTree::window_function(
    const Eigen::Ref<const CoordinateMatrix>& coordinates,
    const config::rtree::InterpolationWindow& config) const
    -> std::tuple<ValueVector, Vector<uint32_t>> {
  auto model = math::interpolate::InterpolationWindow<coordinate_t>(
      config.wf(), config.arg());
  return batch_interpolate(
      coordinates, config.num_threads(),
      [this, &model,
       &config](const point_t& pt) -> std::pair<promotion_t, uint32_t> {
        return base_t::window_function(model, pt, config.radius(), config.k(),
                                       config.boundary_check());
      });
}

// ==========================================================================

/// @brief Register RTree class and its methods to a Python module.
/// @param[in,out] m Python module
auto init_rtree(nanobind::module_& m) -> void;

}  // namespace pyinterp::geometry::geographic::pybind

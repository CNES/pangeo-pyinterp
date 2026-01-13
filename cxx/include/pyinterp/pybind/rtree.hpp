// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <concepts>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

#include "pyinterp/config/rtree.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/geographic/coordinates.hpp"
#include "pyinterp/geometry/geographic/spheroid.hpp"
#include "pyinterp/geometry/point.hpp"
#include "pyinterp/geometry/rtree.hpp"
#include "pyinterp/math/interpolate/rbf.hpp"
#include "pyinterp/math/interpolate/window_function.hpp"
#include "pyinterp/parallel_for.hpp"
#include "pyinterp/pybind/ndarray_serialization.hpp"
#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::pybind {

/// RTree spatial index for 3D points
///
/// This class provides spatial indexing capabilities for 3D coordinates
/// using an R-tree data structure. It supports two coordinate systems:
/// - ECEF (Earth-Centered, Earth-Fixed): Cartesian coordinates in meters
/// - Geodetic: (longitude, latitude, altitude) in degrees/degrees/meters
///
/// The coordinate system is selected at construction time via the `spheroid`
/// parameter. If a spheroid is provided, input coordinates are assumed to be
/// geodetic and will be converted to ECEF internally. If no spheroid is given,
/// input coordinates are assumed to be in ECEF.
template <std::floating_point T>
class RTree3D : public geometry::RTree<geometry::ECEF<T>, T> {
 public:
  /// Scalar type for coordinates and values
  using value_type = T;

  /// Point type (3D Cartesian internally)
  using point_t = geometry::ECEF<T>;

  /// Base class type
  using base_t = geometry::RTree<point_t, T>;

  /// Coordinate type
  using coordinate_t = typename base_t::coordinate_t;

  /// Distance type between points
  using distance_t = typename base_t::distance_t;

  /// Query result type (distance, value, point)
  using result_t = typename base_t::result_t;

  /// Promoted type for arithmetic
  using promotion_t = typename base_t::promotion_t;

  /// Coordinate matrix type: (n, 3) or (n, 2)
  using CoordinateMatrix = RowMajorMatrix<T>;

  /// Value vector type
  using ValueVector = Vector<T>;

  /// Default constructor (WGS84 spheroid)
  RTree3D() : spheroid_(geometry::geographic::Spheroid()) {}

  /// Construct with specified coordinate system
  /// @param[in] spheroid Optional spheroid used to convert geodetic inputs to
  /// ECEF; if std::nullopt, inputs are assumed already ECEF.
  explicit RTree3D(
      const std::optional<geometry::geographic::Spheroid>& spheroid)
      : spheroid_(spheroid) {}

  /// Get the spheroid
  [[nodiscard]] constexpr auto spheroid() const noexcept
      -> const std::optional<geometry::geographic::Spheroid>& {
    return spheroid_;
  }

  /// Bulk-load points using STR packing algorithm
  ///
  /// @param[in] coordinates Matrix of shape (n, 3) or (n, 2) containing
  /// coordinates.
  /// For ECEF: (x, y, z) in meters
  /// For geodetic: (lon, lat, alt) in degrees/degrees/meters
  /// If shape is (n, 2), the third coordinate is assumed to be zero.
  /// @param[in] values Vector of size n containing values at each point
  void packing(const Eigen::Ref<const CoordinateMatrix>& coordinates,
               const Eigen::Ref<const ValueVector>& values);

  /// Insert points into the tree
  ///
  /// @param[in] coordinates Matrix of shape (n, 3) or (n, 2) containing
  /// coordinates
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

  /// Serialize the RTree3D state
  /// @return Serialized byte buffer
  [[nodiscard]] auto getstate() const -> nanobind::tuple {
    serialization::Writer state;
    {
      nanobind::gil_scoped_release release;
      state = base_t::pack();
    }
    return nanobind::make_tuple(writer_to_ndarray(std::move(state)));
  }

  /// Deserialize an RTree3D from a byte buffer
  /// @param buffer Serialized data
  /// @return Deserialized RTree3D instance
  [[nodiscard]] static auto setstate(const nanobind::tuple& state) -> RTree3D {
    if (state.size() != 1) {
      throw std::invalid_argument("Invalid state");
    }
    auto array = nanobind::cast<NanobindArray1DUInt8>(state[0]);
    auto reader = reader_from_ndarray(array);
    {
      nanobind::gil_scoped_release release;
      RTree3D<T> rtree;
      static_cast<base_t&>(rtree) = base_t::unpack(reader);
      return rtree;
    }
  }

 private:
  /// Spheroid for geodetic calculations
  std::optional<geometry::geographic::Spheroid> spheroid_;

  /// Convert input coordinates to internal ECEF representation
  /// @param coordinates Input matrix (n, 3) or (n, 2)
  /// @param row Row index
  /// @return 3D point in internal coordinate system
  [[nodiscard]] auto to_internal_point(
      const Eigen::Ref<const CoordinateMatrix>& coordinates,
      Eigen::Index row) const -> point_t;

  /// Convert geodetic (lon, lat, alt) to ECEF (x, y, z)
  [[nodiscard]] auto geodetic_to_ecef(T lon, T lat, T alt) const -> point_t;

  /// Validate coordinate matrix dimensions
  static void validate_coordinates(
      const Eigen::Ref<const CoordinateMatrix>& coordinates,
      const Eigen::Ref<const ValueVector>& values);

  static void validate_coordinates(
      const Eigen::Ref<const CoordinateMatrix>& coordinates);

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
// ==========================================================================

template <std::floating_point T>
void RTree3D<T>::validate_coordinates(
    const Eigen::Ref<const CoordinateMatrix>& coordinates,
    const Eigen::Ref<const ValueVector>& values) {
  validate_coordinates(coordinates);
  if (coordinates.rows() != values.size()) {
    throw std::invalid_argument(
        "Number of coordinates must match number of values");
  }
}

// ==========================================================================

template <std::floating_point T>
void RTree3D<T>::validate_coordinates(
    const Eigen::Ref<const CoordinateMatrix>& coordinates) {
  if (coordinates.cols() != 2 && coordinates.cols() != 3) {
    throw std::invalid_argument("Coordinates must have shape (n, 2) or (n, 3)");
  }
}

// ==========================================================================

template <std::floating_point T>
auto RTree3D<T>::geodetic_to_ecef(T lon, T lat, T alt) const -> point_t {
  auto transformer = geometry::geographic::Coordinates(*spheroid_);
  return transformer.lla_to_ecef<T>(geometry::LLA<T>{lon, lat, alt});
}

// ==========================================================================

template <std::floating_point T>
auto RTree3D<T>::to_internal_point(
    const Eigen::Ref<const CoordinateMatrix>& coordinates,
    Eigen::Index row) const -> point_t {
  const auto c0 = coordinates(row, 0);
  const auto c1 = coordinates(row, 1);
  const auto c2 = coordinates.cols() == 3 ? coordinates(row, 2) : T{0};

  if (spheroid_ == std::nullopt) {
    // Already in ECEF, use directly
    return point_t{c0, c1, c2};
  }
  // Convert from geodetic (lon, lat, alt) to ECEF
  return geodetic_to_ecef(c0, c1, c2);
}

// ==========================================================================

template <std::floating_point T>
void RTree3D<T>::packing(const Eigen::Ref<const CoordinateMatrix>& coordinates,
                         const Eigen::Ref<const ValueVector>& values) {
  validate_coordinates(coordinates, values);

  std::vector<std::pair<point_t, T>> items;
  items.reserve(static_cast<size_t>(coordinates.rows()));

  for (int64_t ix = 0; ix < coordinates.rows(); ++ix) {
    items.emplace_back(to_internal_point(coordinates, ix), values[ix]);
  }

  base_t::packing(items);
}

// ==========================================================================

template <std::floating_point T>
void RTree3D<T>::insert(const Eigen::Ref<const CoordinateMatrix>& coordinates,
                        const Eigen::Ref<const ValueVector>& values) {
  validate_coordinates(coordinates, values);

  for (int64_t ix = 0; ix < coordinates.rows(); ++ix) {
    base_t::insert({to_internal_point(coordinates, ix), values[ix]});
  }
}

// ==========================================================================

template <std::floating_point T>
template <typename QueryFunc>
auto RTree3D<T>::batch_query(
    const Eigen::Ref<const CoordinateMatrix>& coordinates,
    const config::rtree::Query& config, QueryFunc&& query_func) const
    -> std::tuple<Matrix<distance_t>, Matrix<promotion_t>> {
  validate_coordinates(coordinates);

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
          auto point = to_internal_point(coordinates, ix);
          auto results = query_func(point, config.k());

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

template <std::floating_point T>
template <typename InterpolateFunc>
auto RTree3D<T>::batch_interpolate(
    const Eigen::Ref<const CoordinateMatrix>& coordinates, size_t num_threads,
    InterpolateFunc&& interpolate_func) const
    -> std::tuple<ValueVector, Vector<uint32_t>> {
  validate_coordinates(coordinates);

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
              interpolate_func(to_internal_point(coordinates, ix));
        }
      },
      num_threads);

  return {std::move(values), std::move(counts)};
}

// ==========================================================================

template <std::floating_point T>
auto RTree3D<T>::query(const Eigen::Ref<const CoordinateMatrix>& coordinates,
                       const config::rtree::Query& config) const
    -> std::tuple<Matrix<distance_t>, Matrix<promotion_t>> {
  return batch_query(coordinates, config,
                     [this, config](const point_t& pt, uint32_t neighbors) {
                       return base_t::query(pt, neighbors,
                                            static_cast<T>(config.radius()),
                                            config.boundary_check());
                     });
}

// ==========================================================================

template <std::floating_point T>
auto RTree3D<T>::inverse_distance_weighting(
    const Eigen::Ref<const CoordinateMatrix>& coordinates,
    const config::rtree::InverseDistanceWeighting& config) const
    -> std::tuple<ValueVector, Vector<uint32_t>> {
  return batch_interpolate(
      coordinates, config.num_threads(),
      [this, config](const point_t& pt) -> std::pair<promotion_t, uint32_t> {
        return base_t::inverse_distance_weighting(
            pt, static_cast<T>(config.radius()), config.k(), config.p(),
            config.boundary_check());
      });
}

// ==========================================================================

template <std::floating_point T>
auto RTree3D<T>::kriging(const Eigen::Ref<const CoordinateMatrix>& coordinates,
                         const config::rtree::Kriging& config) const
    -> std::tuple<ValueVector, Vector<uint32_t>> {
  auto model = math::interpolate::Kriging<promotion_t>(
      static_cast<promotion_t>(config.sigma()),
      static_cast<promotion_t>(config.lambda()),
      static_cast<promotion_t>(config.nugget()), config.covariance_model(),
      config.drift_function());
  return batch_interpolate(
      coordinates, config.num_threads(),
      [this, &model,
       &config](const point_t& pt) -> std::pair<promotion_t, uint32_t> {
        return base_t::kriging(model, pt, static_cast<T>(config.radius()),
                               config.k(), config.boundary_check());
      });
}

// ==========================================================================

template <std::floating_point T>
auto RTree3D<T>::radial_basis_function(
    const Eigen::Ref<const CoordinateMatrix>& coordinates,
    const config::rtree::RadialBasisFunction& config) const
    -> std::tuple<ValueVector, Vector<uint32_t>> {
  auto model = math::interpolate::RBF<promotion_t>(
      static_cast<promotion_t>(config.epsilon()),
      static_cast<promotion_t>(config.smooth()), config.rbf());
  return batch_interpolate(
      coordinates, config.num_threads(),
      [this, &model,
       &config](const point_t& pt) -> std::pair<promotion_t, uint32_t> {
        return base_t::radial_basis_function(
            model, pt, static_cast<T>(config.radius()), config.k(),
            config.boundary_check());
      });
}

// ==========================================================================

template <std::floating_point T>
auto RTree3D<T>::window_function(
    const Eigen::Ref<const CoordinateMatrix>& coordinates,
    const config::rtree::InterpolationWindow& config) const
    -> std::tuple<ValueVector, Vector<uint32_t>> {
  auto model = math::interpolate::InterpolationWindow<coordinate_t>(
      config.wf(), static_cast<coordinate_t>(config.arg()));
  return batch_interpolate(
      coordinates, config.num_threads(),
      [this, &model,
       &config](const point_t& pt) -> std::pair<promotion_t, uint32_t> {
        return base_t::window_function(model, pt,
                                       static_cast<T>(config.radius()),
                                       config.k(), config.boundary_check());
      });
}

// ==========================================================================

/// @brief Register RTree3D class and its methods to a Python module.
/// @param[in,out] m Python module
auto init_rtree_3d(nanobind::module_& m) -> void;

}  // namespace pyinterp::pybind

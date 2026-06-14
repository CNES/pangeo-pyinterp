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
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "pyinterp/config/rtree.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/geometry/point.hpp"
#include "pyinterp/geometry/rtree.hpp"
#include "pyinterp/math/interpolate/observation.hpp"
#include "pyinterp/math/interpolate/optimal_interpolation.hpp"
#include "pyinterp/parallel_for.hpp"
#include "pyinterp/pybind/ndarray_serialization.hpp"
#include "pyinterp/serialization_buffer.hpp"

namespace pyinterp::pybind {

/// @brief Spatial index for 4D point data with per-observation error variance.
///
/// Designed to back the Optimal Interpolation (OI / BLUE) estimator: the
/// container indexes scattered observations in a 4D Cartesian space
/// (typically @c (x, y, z, t) or @c (x, y, t, u)) and stores, alongside each
/// value, the measurement-error variance contributed to the diagonal of
/// @f$R@f$ in @f$(C_{oo} + R) w = c_{og}@f$.
///
/// The wrapper is intentionally narrow: it exposes indexing, k-NN queries,
/// the Optimal-Interpolation (BLUE) estimator (@ref optimal_interpolation)
/// and serialization, but no IDW / kriging / RBF / window-function methods
/// — those are not meaningful in 4D and remain on RTree3D.
///
/// @tparam T Floating-point scalar type for both coordinates and values
///   (matches @ref pyinterp::math::interpolate::Observation::value /
///   @ref ...::sigma2).
template <std::floating_point T>
class RTree4D
    : public geometry::RTree<geometry::Cartesian4D<T>,
                             math::interpolate::Observation<T>> {
 public:
  /// Scalar type
  using value_type = T;

  /// Underlying 4D point type
  using point_t = geometry::Cartesian4D<T>;

  /// Observation value type
  using observation_t = math::interpolate::Observation<T>;

  /// Base RTree
  using base_t = geometry::RTree<point_t, observation_t>;

  /// Coordinate type
  using coordinate_t = typename base_t::coordinate_t;

  /// Distance type
  using distance_t = typename base_t::distance_t;

  /// Coordinate matrix type (N, 4)
  using CoordinateMatrix = RowMajorMatrix<T>;

  /// Vector of T (used for values and sigma2 arrays)
  using ValueVector = Vector<T>;

  /// Default constructor — pure Cartesian, no spheroid argument.
  RTree4D() = default;

  /// @brief Bulk-load (point, observation) pairs using STR packing.
  ///
  /// @param[in] coordinates Matrix of shape @c (n, 4) with the four
  ///   coordinates of each observation.
  /// @param[in] values Vector of length @c n with the observed values.
  /// @param[in] sigma2 Vector of length @c n with the measurement-error
  ///   variance of each observation. Must be strictly positive.
  void packing(const Eigen::Ref<const CoordinateMatrix>& coordinates,
               const Eigen::Ref<const ValueVector>& values,
               const Eigen::Ref<const ValueVector>& sigma2);

  /// @brief Insert (point, observation) pairs into the existing tree.
  void insert(const Eigen::Ref<const CoordinateMatrix>& coordinates,
              const Eigen::Ref<const ValueVector>& values,
              const Eigen::Ref<const ValueVector>& sigma2);

  /// @brief k-nearest neighbour query for many points.
  ///
  /// @param[in] coordinates Query coordinates, shape @c (n, 4).
  /// @param[in] config k-NN search configuration (k, radius, num_threads).
  /// @return Tuple @c (distances, values, sigma2) of shape @c (n, k). Cells
  ///   beyond the actual neighbour count are filled with NaN.
  [[nodiscard]] auto query(
      const Eigen::Ref<const CoordinateMatrix>& coordinates,
      const config::rtree::Query& config) const
      -> std::tuple<Matrix<distance_t>, Matrix<T>, Matrix<T>>;

  /// @brief Optimal Interpolation (BLUE) at many query points.
  ///
  /// For each query point, this method (a) retrieves up to @c config.k()
  /// neighbours from the 4D tree (filtered by the optional @c config.radius()
  /// in the packed Euclidean metric), (b) builds the anisotropic
  /// covariance system with per-query length scales and field standard
  /// deviation, (c) solves it via Cholesky (LDLT fallback) and (d) returns
  /// the analysed value, the formal error standard deviation, and the
  /// neighbour count.
  ///
  /// All five per-query arrays (@c lx, @c ly, @c lz, @c lt, @c sigma) are
  /// expected to be already evaluated at every query point: scalar /
  /// geographic variation is handled by the Python wrapper (sampling a
  /// @c Grid2D when needed).
  ///
  /// @param[in] coordinates Query points, shape @c (m, 4).
  /// @param[in] lx Decorrelation length along the first axis, shape @c (m,).
  /// @param[in] ly Decorrelation length along the second axis, shape @c (m,).
  /// @param[in] lz Decorrelation length along the third axis, shape @c (m,).
  /// @param[in] lt Decorrelation length along the fourth axis, shape @c (m,).
  /// @param[in] sigma Field standard deviation, shape @c (m,).
  /// @param[in] config OI configuration (covariance, k, radius, num_threads).
  /// @return Tuple @c (values, errors, neighbors) — three vectors of length
  ///   @c m: analysed value, formal error standard deviation, and number of
  ///   neighbours actually used. Cells with no neighbour return NaN/0.
  [[nodiscard]] auto optimal_interpolation(
      const Eigen::Ref<const CoordinateMatrix>& coordinates,
      const Eigen::Ref<const ValueVector>& lx,
      const Eigen::Ref<const ValueVector>& ly,
      const Eigen::Ref<const ValueVector>& lz,
      const Eigen::Ref<const ValueVector>& lt,
      const Eigen::Ref<const ValueVector>& sigma,
      const config::rtree::OptimalInterpolation& config) const
      -> std::tuple<ValueVector, ValueVector, Vector<uint32_t>>;

  /// Serialize the tree state.
  [[nodiscard]] auto getstate() const -> nanobind::tuple {
    serialization::Writer state;
    {
      nanobind::gil_scoped_release release;
      state = base_t::pack();
    }
    return nanobind::make_tuple(writer_to_ndarray(std::move(state)));
  }

  /// Deserialize a tree state into a new RTree4D.
  [[nodiscard]] static auto setstate(const nanobind::tuple& state) -> RTree4D {
    if (state.size() != 1) {
      throw std::invalid_argument("Invalid state");
    }
    auto array = nanobind::cast<NanobindArray1DUInt8>(state[0]);
    auto reader = reader_from_ndarray(array);
    {
      nanobind::gil_scoped_release release;
      RTree4D<T> rtree;
      static_cast<base_t&>(rtree) = base_t::unpack(reader);
      return rtree;
    }
  }

 private:
  /// Build a 4D point from a row of the coordinate matrix.
  ///
  /// Boost.Geometry's `model::point` only ships constructors for up to 3
  /// arguments, so we default-construct and set each component
  /// individually.
  [[nodiscard]] static auto to_point(
      const Eigen::Ref<const CoordinateMatrix>& coordinates,
      Eigen::Index row) noexcept -> point_t {
    point_t p;
    boost::geometry::set<0>(p, coordinates(row, 0));
    boost::geometry::set<1>(p, coordinates(row, 1));
    boost::geometry::set<2>(p, coordinates(row, 2));
    boost::geometry::set<3>(p, coordinates(row, 3));
    return p;
  }

  /// Validate the (coordinates, values, sigma2) shape triple.
  static void validate_inputs(
      const Eigen::Ref<const CoordinateMatrix>& coordinates,
      const Eigen::Ref<const ValueVector>& values,
      const Eigen::Ref<const ValueVector>& sigma2);
};

// ============================================================================
// Implementation
// ============================================================================

template <std::floating_point T>
void RTree4D<T>::validate_inputs(
    const Eigen::Ref<const CoordinateMatrix>& coordinates,
    const Eigen::Ref<const ValueVector>& values,
    const Eigen::Ref<const ValueVector>& sigma2) {
  if (coordinates.cols() != 4) {
    throw std::invalid_argument("Coordinates must have shape (n, 4)");
  }
  if (coordinates.rows() != values.size() ||
      coordinates.rows() != sigma2.size()) {
    throw std::invalid_argument(
        "coordinates, values and sigma2 must all have the same length");
  }
  for (Eigen::Index i = 0; i < sigma2.size(); ++i) {
    if (!(sigma2[i] > T{0})) {
      throw std::invalid_argument(
          "sigma2 must be strictly positive everywhere");
    }
  }
}

template <std::floating_point T>
void RTree4D<T>::packing(
    const Eigen::Ref<const CoordinateMatrix>& coordinates,
    const Eigen::Ref<const ValueVector>& values,
    const Eigen::Ref<const ValueVector>& sigma2) {
  validate_inputs(coordinates, values, sigma2);

  std::vector<typename base_t::value_t> items;
  items.reserve(static_cast<size_t>(coordinates.rows()));
  for (Eigen::Index i = 0; i < coordinates.rows(); ++i) {
    items.emplace_back(to_point(coordinates, i),
                       observation_t{.value = values[i], .sigma2 = sigma2[i]});
  }
  base_t::packing(items);
}

template <std::floating_point T>
void RTree4D<T>::insert(
    const Eigen::Ref<const CoordinateMatrix>& coordinates,
    const Eigen::Ref<const ValueVector>& values,
    const Eigen::Ref<const ValueVector>& sigma2) {
  validate_inputs(coordinates, values, sigma2);

  for (Eigen::Index i = 0; i < coordinates.rows(); ++i) {
    base_t::insert({to_point(coordinates, i),
                    observation_t{.value = values[i], .sigma2 = sigma2[i]}});
  }
}

template <std::floating_point T>
auto RTree4D<T>::query(const Eigen::Ref<const CoordinateMatrix>& coordinates,
                       const config::rtree::Query& config) const
    -> std::tuple<Matrix<distance_t>, Matrix<T>, Matrix<T>> {
  if (coordinates.cols() != 4) {
    throw std::invalid_argument("Query coordinates must have shape (n, 4)");
  }

  const auto n = coordinates.rows();
  Matrix<distance_t> distances(n, config.k());
  Matrix<T> values(n, config.k());
  Matrix<T> sigma2(n, config.k());

  distances.setConstant(std::numeric_limits<distance_t>::quiet_NaN());
  values.setConstant(std::numeric_limits<T>::quiet_NaN());
  sigma2.setConstant(std::numeric_limits<T>::quiet_NaN());

  parallel_for(
      static_cast<size_t>(n),
      [&](size_t start, size_t end) {
        for (size_t idx = start; idx < end; ++idx) {
          const auto ix = static_cast<Eigen::Index>(idx);
          const auto pt = to_point(coordinates, ix);
          const auto results =
              base_t::query(pt, config.k(), static_cast<T>(config.radius()),
                            config.boundary_check());
          for (size_t jx = 0; jx < results.size() && jx < config.k(); ++jx) {
            distances(ix, static_cast<Eigen::Index>(jx)) = results[jx].first;
            values(ix, static_cast<Eigen::Index>(jx)) =
                results[jx].second.value;
            sigma2(ix, static_cast<Eigen::Index>(jx)) =
                results[jx].second.sigma2;
          }
        }
      },
      config.num_threads());

  return {std::move(distances), std::move(values), std::move(sigma2)};
}

template <std::floating_point T>
auto RTree4D<T>::optimal_interpolation(
    const Eigen::Ref<const CoordinateMatrix>& coordinates,
    const Eigen::Ref<const ValueVector>& lx,
    const Eigen::Ref<const ValueVector>& ly,
    const Eigen::Ref<const ValueVector>& lz,
    const Eigen::Ref<const ValueVector>& lt,
    const Eigen::Ref<const ValueVector>& sigma,
    const config::rtree::OptimalInterpolation& config) const
    -> std::tuple<ValueVector, ValueVector, Vector<uint32_t>> {
  if (coordinates.cols() != 4) {
    throw std::invalid_argument("Query coordinates must have shape (m, 4)");
  }
  const Eigen::Index m = coordinates.rows();
  if (lx.size() != m || ly.size() != m || lz.size() != m || lt.size() != m ||
      sigma.size() != m) {
    throw std::invalid_argument(
        "lx, ly, lz, lt and sigma must all have the same length as the "
        "number of query points");
  }

  // Build a single estimator — its only state is the kernel pointer, safe to
  // share across worker threads.
  const math::interpolate::OptimalInterpolation<T> oi(config.covariance_model());

  ValueVector values(m);
  ValueVector errors(m);
  Vector<uint32_t> neighbors(m);
  values.setConstant(std::numeric_limits<T>::quiet_NaN());
  errors.setConstant(std::numeric_limits<T>::quiet_NaN());
  neighbors.setZero();

  const T radius = static_cast<T>(config.radius());
  const uint32_t k = config.k();

  parallel_for(
      static_cast<size_t>(m),
      [&](size_t start, size_t end) {
        // Per-thread scratch buffers — sized to k, reused across queries.
        Eigen::Matrix<T, -1, 4, Eigen::RowMajor> obs_coords(k, 4);
        Vector<T> obs_values(k);
        Vector<T> obs_sigma2(k);

        for (size_t idx = start; idx < end; ++idx) {
          const auto ix = static_cast<Eigen::Index>(idx);
          const auto pt = to_point(coordinates, ix);

          // `value()` returns (point, observation) pairs filtered by radius
          // — exactly what the OI math needs (coordinates + value + σ²).
          const auto pairs = base_t::value(pt, radius, k,
                                           config.boundary_check());
          const auto np = static_cast<Eigen::Index>(pairs.size());
          if (np == 0) {
            continue;  // values/errors stay NaN, neighbors stays 0.
          }
          if (obs_coords.rows() != np) {
            obs_coords.resize(np, 4);
            obs_values.resize(np);
            obs_sigma2.resize(np);
          }
          for (Eigen::Index i = 0; i < np; ++i) {
            const auto& pair = pairs[static_cast<size_t>(i)];
            obs_coords(i, 0) = boost::geometry::get<0>(pair.first);
            obs_coords(i, 1) = boost::geometry::get<1>(pair.first);
            obs_coords(i, 2) = boost::geometry::get<2>(pair.first);
            obs_coords(i, 3) = boost::geometry::get<3>(pair.first);
            obs_values[i] = pair.second.value;
            obs_sigma2[i] = pair.second.sigma2;
          }
          neighbors[ix] = static_cast<uint32_t>(np);

          const Eigen::Vector<T, 4> query{
              coordinates(ix, 0), coordinates(ix, 1), coordinates(ix, 2),
              coordinates(ix, 3)};
          const Eigen::Vector<T, 4> inv_L{T{1} / lx[ix], T{1} / ly[ix],
                                          T{1} / lz[ix], T{1} / lt[ix]};
          const auto result =
              oi.solve(obs_coords.topRows(np), obs_values.head(np),
                       obs_sigma2.head(np), query, inv_L, sigma[ix]);
          values[ix] = result.value;
          errors[ix] = std::sqrt(result.error2);
        }
      },
      config.num_threads());

  return {std::move(values), std::move(errors), std::move(neighbors)};
}

/// @brief Register RTree4D and its factory with a Python module.
auto init_rtree_4d(nanobind::module_& m) -> void;

}  // namespace pyinterp::pybind

// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <cstdint>
#include <limits>

#include "pyinterp/broadcast.hpp"
#include "pyinterp/config/windowed.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/math/interpolate/cache.hpp"
#include "pyinterp/math/interpolate/cache_loader.hpp"
#include "pyinterp/math/interpolate/interpolation_result.hpp"
#include "pyinterp/math/interpolate/univariate.hpp"
#include "pyinterp/parallel_for.hpp"
#include "pyinterp/pybind/grid.hpp"

namespace pyinterp::windowed::pybind {

/// @brief Alias for the one-dimensional grid type
/// @tparam DataType Data type stored in the grid
template <typename DataType>
using Grid1D = pyinterp::pybind::Grid1D<DataType>;

/// @brief Alias for the interpolation result type
/// @tparam T Value type
template <typename T>
using InterpolationResult = pyinterp::math::interpolate::InterpolationResult<T>;

/// @brief Alias for the 1D interpolation cache type
using UnivariateCache =
    pyinterp::math::interpolate::InterpolationCache<double, double>;

/// @brief Perform univariate interpolation for a single point
/// @tparam DataType Data type of the grid
/// @tparam ResultType Result type of the interpolation
/// @param grid 1D grid containing data to interpolate
/// @param[in] x X-coordinate for interpolation
/// @param[in] cfg Configuration parameters for interpolation
/// @param[in,out] interpolator Interpolator object
/// @param[in,out] cache Interpolation cache
/// @return Interpolated value
template <typename DataType, typename ResultType>
[[nodiscard]] auto univariate_single(
    const Grid1D<DataType>& grid, const double x,
    const config::windowed::Univariate& cfg,
    math::interpolate::Univariate<double>* interpolator, UnivariateCache& cache)
    -> InterpolationResult<ResultType> {
  auto cache_load_result = math::interpolate::update_cache_if_needed(
      cache, grid, std::make_tuple(x), cfg.univariate().boundary_mode(),
      cfg.common().bounds_error());
  if (!cache_load_result.success) {
    // The cache could not be loaded. If an error message is provided and
    // bounds_error is true, that means the requested coordinate is outside
    // the grid domain.
    if (cache_load_result.error_message.has_value()) {
      throw std::out_of_range(cache_load_result.error_message.value());
    }
    // Otherwise, the interpolation window cannot be constructed because part
    // of it is out of bounds.
    return {};
  }
  if (!cache.is_valid()) {
    // Cache contains only NaN values, interpolation cannot proceed
    return {};
  }
  auto result =
      (*interpolator)(cache.template coords_as_eigen<0>(), cache.vector(), x);
  return InterpolationResult<ResultType>{static_cast<ResultType>(result)};
}

/// @brief Calculate derivative for a single point
/// @tparam DataType Data type of the grid
/// @tparam ResultType Result type of the derivative
/// @param grid 1D grid containing data to interpolate
/// @param[in] x X-coordinate for derivative calculation
/// @param[in] cfg Configuration parameters for interpolation
/// @param[in,out] interpolator Interpolator object
/// @param[in,out] cache Interpolation cache
/// @return Derivative value
template <typename DataType, typename ResultType>
[[nodiscard]] auto univariate_derivative_single(
    const Grid1D<DataType>& grid, const double x,
    const config::windowed::Univariate& cfg,
    math::interpolate::Univariate<double>* interpolator, UnivariateCache& cache)
    -> InterpolationResult<ResultType> {
  auto cache_load_result = math::interpolate::update_cache_if_needed(
      cache, grid, std::make_tuple(x), cfg.univariate().boundary_mode(),
      cfg.common().bounds_error());
  if (!cache_load_result.success) {
    // The cache could not be loaded. If an error message is provided and
    // bounds_error is true, that means the requested coordinate is outside
    // the grid domain.
    if (cache_load_result.error_message.has_value()) {
      throw std::out_of_range(cache_load_result.error_message.value());
    }
    // Otherwise, the interpolation window cannot be constructed because part
    // of it is out of bounds.
    return {};
  }
  if (!cache.is_valid()) {
    // Cache contains only NaN values, derivative calculation cannot proceed
    return {};
  }
  auto result = interpolator->derivative(cache.template coords_as_eigen<0>(),
                                         cache.vector(), x);
  return InterpolationResult<ResultType>{static_cast<ResultType>(result)};
}

/// @brief Vectorized univariate interpolation
/// @tparam DataType Data type of the grid
/// @tparam ResultType Result type of the interpolation
/// @param grid 1D grid containing data to interpolate
/// @param[in] x X-coordinates for interpolation
/// @param[in] cfg Configuration parameters for interpolation
/// @return Vector of interpolated values
template <typename DataType, typename ResultType>
[[nodiscard]] auto univariate(const Grid1D<DataType>& grid,
                              const Eigen::Ref<const Eigen::VectorXd>& x,
                              const config::windowed::Univariate& cfg) {
  Vector<ResultType> result(x.size());
  result.setConstant(std::numeric_limits<ResultType>::quiet_NaN());

  parallel_for(
      x.size(),
      [&](const int64_t start, const int64_t end) {
        // Create cache and interpolator for this thread
        // For 1D, we only need window_size for the X dimension (Y dimension is
        // unused)
        auto cache = UnivariateCache(cfg.univariate().half_window_size(),
                                     1);  // Y dimension unused for 1D
        auto interpolator = cfg.univariate().factory<double>();

        for (int64_t ix = start; ix < end; ++ix) {
          auto interpolated_value = univariate_single<DataType, ResultType>(
              grid, x[ix], cfg, interpolator.get(), cache);
          if (interpolated_value.has_value()) {
            result[ix] = *interpolated_value.value;
          }
        }
      },
      cfg.common().num_threads());

  return result;
}

/// @brief Vectorized univariate derivative calculation
/// @tparam DataType Data type of the grid
/// @tparam ResultType Result type of the derivative
/// @param grid 1D grid containing data to interpolate
/// @param[in] x X-coordinates for derivative calculation
/// @param[in] cfg Configuration parameters for interpolation
/// @return Vector of derivative values
template <typename DataType, typename ResultType>
[[nodiscard]] auto univariate_derivative(
    const Grid1D<DataType>& grid, const Eigen::Ref<const Eigen::VectorXd>& x,
    const config::windowed::Univariate& cfg) {
  Vector<ResultType> result(x.size());
  result.setConstant(std::numeric_limits<ResultType>::quiet_NaN());

  parallel_for(
      x.size(),
      [&](const int64_t start, const int64_t end) {
        // Create cache and interpolator for this thread
        // For 1D, we only need window_size for the X dimension (Y dimension is
        // unused)
        auto cache = UnivariateCache(cfg.univariate().half_window_size(),
                                     1);  // Y dimension unused for 1D
        auto interpolator = cfg.univariate().factory<double>();

        for (int64_t ix = start; ix < end; ++ix) {
          auto derivative_value =
              univariate_derivative_single<DataType, ResultType>(
                  grid, x[ix], cfg, interpolator.get(), cache);
          if (derivative_value.has_value()) {
            result[ix] = *derivative_value.value;
          }
        }
      },
      cfg.common().num_threads());

  return result;
}

}  // namespace pyinterp::windowed::pybind

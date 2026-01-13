// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include "pyinterp/broadcast.hpp"
#include "pyinterp/config/windowed.hpp"
#include "pyinterp/math/interpolate/bivariate.hpp"
#include "pyinterp/math/interpolate/cache.hpp"
#include "pyinterp/math/interpolate/cache_loader.hpp"
#include "pyinterp/math/interpolate/interpolation_result.hpp"
#include "pyinterp/math/interpolate/linear.hpp"
#include "pyinterp/math/interpolate/nearest.hpp"
#include "pyinterp/parallel_for.hpp"

namespace pyinterp::windowed::pybind {

/// @brief Alias for the interpolation result type
/// @tparam T Value type
template <typename T>
using InterpolationResult = pyinterp::math::interpolate::InterpolationResult<T>;

/// @brief Alias for the 3D interpolation cache type
/// @tparam ZType Type of the third axis
template <typename ZType>
using InterpolationCache3D =
    pyinterp::math::interpolate::InterpolationCache<double, double, double,
                                                    ZType>;

/// @brief Perform trivariate interpolation for a single point
/// @tparam GridType Type of the grid
/// @tparam ResultType Result type of the interpolation
/// @tparam ZType Type of the third axis
/// @param[in] grid 3D grid containing data to interpolate
/// @param[in] x X-coordinate for interpolation
/// @param[in] y Y-coordinate for interpolation
/// @param[in] z Z-coordinate for interpolation
/// @param cfg Configuration parameters for interpolation
/// @param interpolator Interpolator object
/// @param cache Interpolation cache
/// @return Interpolated value
template <typename GridType, typename ResultType, typename ZType>
[[nodiscard]] auto trivariate_single(
    const GridType& grid, const double x, const double y, const ZType z,
    const config::windowed::Trivariate& cfg,
    math::interpolate::BivariateBase<double>* interpolator,
    InterpolationCache3D<ZType>& cache) -> InterpolationResult<ResultType> {
  auto cache_load_result = math::interpolate::update_cache_if_needed(
      cache, grid, std::make_tuple(x, y, z), cfg.spatial().boundary_mode(),
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

  const auto z0 = cache.template coord<2>(0);
  const auto z1 = cache.template coord<2>(1);
  const auto f0 = (*interpolator)(cache.template coords_as_eigen<0>(),
                                  cache.template coords_as_eigen<1>(),
                                  cache.matrix(0), x, y);
  const auto f1 = (*interpolator)(cache.template coords_as_eigen<0>(),
                                  cache.template coords_as_eigen<1>(),
                                  cache.matrix(1), x, y);

  if (cfg.third_axis().method() == config::AxisMethod::kLinear) {
    // Linear interpolation along Z axis
    return {
        static_cast<ResultType>(math::interpolate::linear(z, z0, z1, f0, f1))};
  }
  // Nearest neighbor interpolation along Z axis
  return {
      static_cast<ResultType>(math::interpolate::nearest(z, z0, z1, f0, f1))};
}

/// @brief Vectorized trivariate interpolation
/// @tparam GridType Type of the grid
/// @tparam ResultType Result type of the interpolation
/// @tparam ZType Type of the third axis
/// @param[in] grid 3D grid containing data to interpolate
/// @param[in] x X-coordinates for interpolation
/// @param[in] y Y-coordinates for interpolation
/// @param[in] z Z-coordinates for interpolation
/// @param cfg Configuration parameters for interpolation
/// @return Vector of interpolated values
template <typename GridType, typename ResultType, typename ZType>
[[nodiscard]] auto trivariate(const GridType& grid,
                              const Eigen::Ref<const Eigen::VectorXd>& x,
                              const Eigen::Ref<const Eigen::VectorXd>& y,
                              const Eigen::Ref<const Vector<ZType>>& z,
                              const config::windowed::Trivariate& cfg) {
  broadcast::check_eigen_shape("x", x, "y", y, "z", z);
  Vector<ResultType> result(x.size());
  result.setConstant(std::numeric_limits<ResultType>::quiet_NaN());

  parallel_for(
      x.size(),
      [&](const int64_t start, const int64_t end) {
        // Create cache and interpolator for this thread
        auto cache =
            InterpolationCache3D<ZType>(cfg.spatial().half_window_size_x(),
                                        cfg.spatial().half_window_size_y());
        auto interpolator = cfg.spatial().factory<double>();

        for (int64_t ix = start; ix < end; ++ix) {
          auto interpolated_value =
              trivariate_single<GridType, ResultType, ZType>(
                  grid, x[ix], y[ix], z[ix], cfg, interpolator.get(), cache);
          if (interpolated_value.has_value()) {
            result[ix] = *interpolated_value.value;
          }
        }
      },
      cfg.common().num_threads());

  return result;
}

/// Vectorized trivariate interpolation with Z coordinate as a Python object
/// @tparam GridType Type of the grid
/// @tparam ResultType Result type of the interpolation
/// @param[in] grid 3D grid containing data to interpolate
/// @param[in] x X-coordinates for interpolation
/// @param[in] y Y-coordinates for interpolation
/// @param[in] z Z-coordinates for interpolation as a Python object
/// @param cfg Configuration parameters for interpolation
/// @return Vector of interpolated values
template <typename GridType, typename ResultType>
[[nodiscard]] auto trivariate(const GridType& grid,
                              const Eigen::Ref<const Eigen::VectorXd>& x,
                              const Eigen::Ref<const Eigen::VectorXd>& y,
                              const nanobind::object& z,
                              const config::windowed::Trivariate& cfg)
    -> Vector<ResultType> {
  if constexpr (GridType::kHasTemporalAxis) {
    // Z is temporal axis, cast to int64_t
    auto z_as_int64 = grid.template pybind_axis<2>().cast_to_int64(z);
    {
      nanobind::gil_scoped_release release;
      return trivariate<GridType, ResultType, int64_t>(grid, x, y, z_as_int64,
                                                       cfg);
    }
  } else {
    // Z is spatial axis, cast to its native type
    using ZType = typename GridType::template math_axis_value_t<2>;
    auto z_as_type = nanobind::cast<Eigen::Ref<const Vector<ZType>>>(z);
    {
      nanobind::gil_scoped_release release;
      return trivariate<GridType, ResultType, ZType>(grid, x, y, z_as_type,
                                                     cfg);
    }
  }
}

}  // namespace pyinterp::windowed::pybind

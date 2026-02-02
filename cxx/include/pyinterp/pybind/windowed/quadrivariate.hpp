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

/// @brief Alias for the 4D interpolation cache type
/// @tparam T Value type
/// @tparam ZType Type of the third axis coordinate
template <typename ZType>
using InterpolationCache4D =
    pyinterp::math::interpolate::InterpolationCache<double, double, double,
                                                    ZType, double>;

/// @brief Container for spatial interpolators needed for quadrivariate
/// interpolation
/// @tparam T Value type
struct QuadrvariateSpatialInterpolators {
  /// @brief Interpolator for the (z, u) plane at the spatial corner (x0, y0)
  /// This interpolator provides values on the (z, u) axes for the lower-left
  /// corner of the (x, y) plane used in the 4D interpolation.
  std::unique_ptr<math::interpolate::BivariateBase<double>> interpolator_x0y0{
      nullptr};
  /// @brief Interpolator for the (z, u) plane at the spatial corner (x0, y1)
  /// This interpolator provides values on the (z, u) axes for the upper-left
  /// corner of the (x, y) plane used in the 4D interpolation.
  std::unique_ptr<math::interpolate::BivariateBase<double>> interpolator_x0y1{
      nullptr};
  /// @brief Interpolator for the (z, u) plane at the spatial corner (x1, y0)
  /// This interpolator provides values on the (z, u) axes for the lower-right
  /// corner of the (x, y) plane used in the 4D interpolation.
  std::unique_ptr<math::interpolate::BivariateBase<double>> interpolator_x1y0{
      nullptr};
  /// @brief Interpolator for the (z, u) plane at the spatial corner (x1, y1)
  /// This interpolator provides values on the (z, u) axes for the upper-right
  /// corner of the (x, y) plane used in the 4D interpolation.
  std::unique_ptr<math::interpolate::BivariateBase<double>> interpolator_x1y1{
      nullptr};
  /// @brief Constructor
  explicit QuadrvariateSpatialInterpolators(
      const config::windowed::Quadrivariate& cfg)
      : interpolator_x0y0(cfg.spatial().factory<double>()),
        interpolator_x0y1(cfg.spatial().factory<double>()),
        interpolator_x1y0(cfg.spatial().factory<double>()),
        interpolator_x1y1(cfg.spatial().factory<double>()) {}
};

/// @brief Perform quadrivariate interpolation for a single point
/// @tparam GridType Type of the grid
/// @tparam ResultType Result type of the interpolation
/// @tparam ZType Type of the third axis coordinate
/// @param[in] grid 4D grid containing data to interpolate
/// @param[in] x X-coordinate for interpolation
/// @param[in] y Y-coordinate for interpolation
/// @param[in] z Z-coordinate for interpolation
/// @param[in] u U-coordinate for interpolation
/// @param cfg Configuration parameters for interpolation
/// @param interpolator Interpolator object
/// @param cache Interpolation cache
/// @return Interpolated value
template <typename GridType, typename ResultType, typename ZType>
[[nodiscard]] auto quadrivariate_single(
    const GridType& grid, const double x, const double y, const ZType z,
    const double u, const config::windowed::Quadrivariate& cfg,
    QuadrvariateSpatialInterpolators& interpolators,
    InterpolationCache4D<ZType>& cache) -> InterpolationResult<ResultType> {
  auto cache_load_result = math::interpolate::update_cache_if_needed(
      cache, grid, std::make_tuple(x, y, z, u), cfg.spatial().boundary_mode(),
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
  if (cache_load_result.was_updated) {
    // Cache was updated, prepare the interpolators with new data
    const auto x_coords = cache.template coords_as_eigen<0>();
    const auto y_coords = cache.template coords_as_eigen<1>();

    interpolators.interpolator_x0y0->prepare(x_coords, y_coords,
                                             cache.matrix(0, 0));
    interpolators.interpolator_x0y1->prepare(x_coords, y_coords,
                                             cache.matrix(0, 1));
    interpolators.interpolator_x1y0->prepare(x_coords, y_coords,
                                             cache.matrix(1, 0));
    interpolators.interpolator_x1y1->prepare(x_coords, y_coords,
                                             cache.matrix(1, 1));
  }
  const auto z0 = cache.template coord<2>(0);
  const auto z1 = cache.template coord<2>(1);
  const auto u0 = cache.template coord<3>(0);
  const auto u1 = cache.template coord<3>(1);

  const auto f00 = (*interpolators.interpolator_x0y0)(x, y);
  const auto f01 = (*interpolators.interpolator_x0y1)(x, y);
  const auto f10 = (*interpolators.interpolator_x1y0)(x, y);
  const auto f11 = (*interpolators.interpolator_x1y1)(x, y);

  ResultType f0;
  ResultType f1;

  if (cfg.third_axis().method() == config::AxisMethod::kLinear) {
    // Linear interpolation along Z axis
    f0 =
        static_cast<ResultType>(math::interpolate::linear(z, z0, z1, f00, f10));
    f1 =
        static_cast<ResultType>(math::interpolate::linear(z, z0, z1, f01, f11));
  } else {
    // Nearest neighbor interpolation along Z axis
    f0 = static_cast<ResultType>(
        math::interpolate::nearest(z, z0, z1, f00, f10));
    f1 = static_cast<ResultType>(
        math::interpolate::nearest(z, z0, z1, f01, f11));
  }

  if (cfg.fourth_axis().method() == config::AxisMethod::kLinear) {
    // Linear interpolation along U axis
    return {
        static_cast<ResultType>(math::interpolate::linear(u, u0, u1, f0, f1))};
  }
  return {
      static_cast<ResultType>(math::interpolate::nearest(u, u0, u1, f0, f1))};
}

/// @brief Vectorized quadrivariate interpolation
/// @tparam GridType Type of the grid
/// @tparam ResultType Result type of the interpolation
/// @tparam ZType Type of the third axis coordinate
/// @param[in] grid 4D grid containing data to interpolate
/// @param[in] x X-coordinates for interpolation
/// @param[in] y Y-coordinates for interpolation
/// @param[in] z Z-coordinates for interpolation
/// @param[in] u U-coordinates for interpolation
/// @param cfg Configuration parameters for interpolation
/// @return Vector of interpolated values
template <typename GridType, typename ResultType, typename ZType>
[[nodiscard]] auto quadrivariate(const GridType& grid,
                                 const Eigen::Ref<const Vector<double>>& x,
                                 const Eigen::Ref<const Vector<double>>& y,
                                 const Eigen::Ref<const Vector<ZType>>& z,
                                 const Eigen::Ref<const Vector<double>>& u,
                                 const config::windowed::Quadrivariate& cfg) {
  broadcast::check_eigen_shape("x", x, "y", y, "z", z, "u", u);
  Vector<ResultType> result(x.size());
  result.setConstant(std::numeric_limits<ResultType>::quiet_NaN());

  parallel_for(
      x.size(),
      [&](const int64_t start, const int64_t end) {
        // Create cache and interpolator for this thread
        auto cache =
            InterpolationCache4D<ZType>(cfg.spatial().half_window_size_x(),
                                        cfg.spatial().half_window_size_y());
        auto interpolators = QuadrvariateSpatialInterpolators(cfg);

        for (int64_t ix = start; ix < end; ++ix) {
          auto interpolated_value =
              quadrivariate_single<GridType, ResultType, ZType>(
                  grid, x[ix], y[ix], z[ix], u[ix], cfg, interpolators, cache);
          if (interpolated_value.has_value()) {
            result[ix] = *interpolated_value.value;
          }
        }
      },
      cfg.common().num_threads());

  return result;
}

/// @brief Vectorized quadrivariate interpolation with Z coordinate as a Python
/// object.
/// @tparam GridType Type of the grid
/// @tparam ResultType Result type of the interpolation
/// @param[in] grid 4D grid containing data to interpolate
/// @param[in] x X-coordinates for interpolation
/// @param[in] y Y-coordinates for interpolation
/// @param[in] z Z-coordinates for interpolation as a Python object
/// @param[in] u U-coordinates for interpolation
/// @param cfg Configuration parameters for interpolation
/// @return Vector of interpolated values
template <typename GridType, typename ResultType>
[[nodiscard]] auto quadrivariate(const GridType& grid,
                                 const Eigen::Ref<const Vector<double>>& x,
                                 const Eigen::Ref<const Vector<double>>& y,
                                 const nanobind::object& z,
                                 const Eigen::Ref<const Vector<double>>& u,
                                 const config::windowed::Quadrivariate& cfg)
    -> Vector<ResultType> {
  if constexpr (GridType::kHasTemporalAxis) {
    // Z is temporal axis, cast to int64_t
    auto z_as_int64 = grid.template pybind_axis<2>().cast_to_int64(z);
    {
      nanobind::gil_scoped_release release;

      return quadrivariate<GridType, ResultType, int64_t>(grid, x, y,
                                                          z_as_int64, u, cfg);
    }
  } else {
    // Z is spatial axis, cast to its native type
    using ZType = typename GridType::template math_axis_value_t<2>;
    auto z_as_type = nanobind::cast<Eigen::Ref<const Vector<ZType>>>(z);
    {
      nanobind::gil_scoped_release release;

      return quadrivariate<GridType, ResultType, ZType>(grid, x, y, z_as_type,
                                                        u, cfg);
    }
  }
}

}  // namespace pyinterp::windowed::pybind

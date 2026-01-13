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
#include <optional>

#include "pyinterp/broadcast.hpp"
#include "pyinterp/config/geometric.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/math/interpolate/geometric/bivariate.hpp"
#include "pyinterp/math/interpolate/geometric/multivariate.hpp"
#include "pyinterp/math/interpolate/interpolation_result.hpp"
#include "pyinterp/parallel_for.hpp"

namespace pyinterp::geometric::pybind {
namespace detail {

/// @brief Result type for single-point trivariate interpolation.
template <typename T>
using TrivariateInterpolationResult =
    pyinterp::math::interpolate::InterpolationResult<T>;

/// @brief Single-point trivariate interpolation.
/// @tparam Point Point template class.
/// @tparam GridType Type of the grid.
/// @tparam ResultType Type of the interpolation result.
/// @tparam ZType Type of the third axis coordinate.
/// @param[in] grid The trivariate grid.
/// @param[in] x X coordinate of the query point.
/// @param[in] y Y coordinate of the query point.
/// @param[in] z Z coordinate of the query point.
/// @param[in] spatial_interpolator Spatial interpolator for the (X,Y) plane.
/// @param[in] z_axis_interpolator Interpolator for the Z axis.
/// @param[in] bounds_error Whether to raise an error if the point is out of
/// bounds.
/// @return The interpolation result.
template <template <class> class Point, typename GridType, typename ResultType,
          typename ZType>
[[nodiscard]] auto trivariate_single(
    const GridType& grid, const double x, const double y, const ZType z,
    const math::interpolate::geometric::Bivariate<Point, ResultType>*
        spatial_interpolator,
    const math::interpolate::geometric::AxisInterpolator<ZType, ResultType>&
        z_axis_interpolator,
    const bool bounds_error) -> TrivariateInterpolationResult<ResultType> {
  // Early exit if out of bounds
  auto x_indexes = grid.template find_indexes<0>(x, bounds_error);
  if (!x_indexes.has_value()) {
    return {std::nullopt};
  }

  auto y_indexes = grid.template find_indexes<1>(y, bounds_error);
  if (!y_indexes.has_value()) {
    return {std::nullopt};
  }

  auto z_indexes = grid.template find_indexes<2>(z, bounds_error);
  if (!z_indexes.has_value()) {
    return {std::nullopt};
  }

  auto [ix0, ix1] = *x_indexes;
  auto [iy0, iy1] = *y_indexes;
  auto [iz0, iz1] = *z_indexes;

  // Cache axis references
  const auto& x_axis = grid.template axis<0>();
  const auto& y_axis = grid.template axis<1>();
  const auto& z_axis = grid.template axis<2>();

  // Fetch grid values for the 8 corners of the 3D cell
  const auto v000 = static_cast<ResultType>(grid.value(ix0, iy0, iz0));
  const auto v010 = static_cast<ResultType>(grid.value(ix0, iy1, iz0));
  const auto v100 = static_cast<ResultType>(grid.value(ix1, iy0, iz0));
  const auto v110 = static_cast<ResultType>(grid.value(ix1, iy1, iz0));
  const auto v001 = static_cast<ResultType>(grid.value(ix0, iy0, iz1));
  const auto v011 = static_cast<ResultType>(grid.value(ix0, iy1, iz1));
  const auto v101 = static_cast<ResultType>(grid.value(ix1, iy0, iz1));
  const auto v111 = static_cast<ResultType>(grid.value(ix1, iy1, iz1));

  // Construct spatial points and bounds
  const auto x0 = static_cast<ResultType>(x_axis.coordinate_value(ix0));
  const auto x1 = static_cast<ResultType>(x_axis.coordinate_value(ix1));
  const auto y0 = static_cast<ResultType>(y_axis.coordinate_value(iy0));
  const auto y1 = static_cast<ResultType>(y_axis.coordinate_value(iy1));
  const auto z0 = static_cast<ZType>(z_axis.coordinate_value(iz0));
  const auto z1 = static_cast<ZType>(z_axis.coordinate_value(iz1));

  // Create query point and bounding box
  using SpatialPoint3D =
      math::interpolate::geometric::SpatialPoint3D<Point, ResultType, ZType>;

  const auto query = SpatialPoint3D(
      Point<ResultType>(
          static_cast<ResultType>(x_axis.normalize_coordinate(x, x0)),
          static_cast<ResultType>(y)),
      z);
  const auto bounds_lower = SpatialPoint3D(Point<ResultType>(x0, y0), z0);
  const auto bounds_upper = SpatialPoint3D(Point<ResultType>(x1, y1), z1);

  // Create data cube
  const auto data = math::interpolate::geometric::DataCube<ResultType>(
      v000, v010, v100, v110, v001, v011, v101, v111);

  // Perform trivariate interpolation
  const auto result = math::interpolate::geometric::trivariate(
      query, bounds_lower, bounds_upper, data, spatial_interpolator,
      z_axis_interpolator);

  return {static_cast<ResultType>(result)};
}

/// @brief Vectorized trivariate interpolation
/// @tparam Point Point template class.
/// @tparam GridType Type of the grid.
/// @tparam ResultType Type of the interpolation result.
/// @tparam ZType Type of the third axis coordinate.
/// @param[in] grid The trivariate grid.
/// @param[in] x X coordinates of the query points.
/// @param[in] y Y coordinates of the query points.
/// @param[in] z Z coordinates of the query points.
/// @param[in] config Configuration for trivariate interpolation.
/// @return Vector of interpolated values.
template <template <class> class Point, typename GridType, typename ResultType,
          typename ZType>
[[nodiscard]] auto trivariate(const GridType& grid,
                              const Eigen::Ref<const Eigen::VectorXd>& x,
                              const Eigen::Ref<const Eigen::VectorXd>& y,
                              const Eigen::Ref<const Vector<ZType>>& z,
                              const config::geometric::Trivariate& config)
    -> Vector<ResultType> {
  broadcast::check_eigen_shape("x", x, "y", y, "z", z);

  // Create spatial interpolator once (outside parallel region)
  auto spatial_interpolator =
      math::interpolate::geometric::make_interpolator<Point, ResultType>(
          config.spatial().method(), config.spatial().exponent());
  const auto* spatial_interpolator_ptr = spatial_interpolator.get();

  // Create z-axis interpolator
  const auto z_axis_method =
      config.third_axis().method() == config::AxisMethod::kLinear
          ? math::interpolate::geometric::AxisMethod::kLinear
          : math::interpolate::geometric::AxisMethod::kNearest;
  const auto z_axis_interpolator =
      math::interpolate::geometric::get_axis_interpolator<ZType, ResultType>(
          z_axis_method);

  Vector<ResultType> result(x.size());
  result.setConstant(std::numeric_limits<ResultType>::quiet_NaN());

  parallel_for(
      x.size(),
      [&](const int64_t start, const int64_t end) {
        for (int64_t ix = start; ix < end; ++ix) {
          auto interpolated_value =
              detail::trivariate_single<Point, GridType, ResultType, ZType>(
                  grid, x[ix], y[ix], z[ix], spatial_interpolator_ptr,
                  z_axis_interpolator, config.common().bounds_error());

          if (interpolated_value.has_value()) {
            result[ix] = *interpolated_value.value;
          }
        }
      },
      config.common().num_threads());

  return result;
}

/// @brief Vectorized trivariate interpolation with Z coordinate as a Python
/// object.
/// @tparam Point Point template class.
/// @tparam GridType Type of the grid.
/// @tparam ResultType Type of the interpolation result.
/// @param[in] grid The trivariate grid.
/// @param[in] x X coordinates of the query points.
/// @param[in] y Y coordinates of the query points.
/// @param[in] z Z coordinates of the query points as a Python object.
/// @param[in] config Configuration for trivariate interpolation.
/// @return Vector of interpolated values.
template <template <class> class Point, typename GridType, typename ResultType>
auto trivariate(const GridType& grid,
                const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& y,
                const nanobind::object& z,
                const config::geometric::Trivariate& config)
    -> Vector<ResultType> {
  if constexpr (GridType::kHasTemporalAxis) {
    // Z is temporal axis, cast to int64_t
    auto z_as_int64 = grid.template pybind_axis<2>().cast_to_int64(z);
    {
      nanobind::gil_scoped_release release;

      return trivariate<Point, GridType, ResultType, int64_t>(
          grid, x, y, z_as_int64, config);
    }
  } else {
    // Z is spatial axis, cast to its native type
    using ZType = typename GridType::template math_axis_value_t<2>;
    auto z_as_type = nanobind::cast<Eigen::Ref<const Vector<ZType>>>(z);
    {
      nanobind::gil_scoped_release release;

      return trivariate<Point, GridType, ResultType, ZType>(grid, x, y,
                                                            z_as_type, config);
    }
  }
}

constexpr const char* const kTrivariateDocstring = R"(
Perform trivariate interpolation on a 3D grid.

Args:
    grid: 3D grid containing data to interpolate.
    x: X coordinates of the points to interpolate.
    y: Y coordinates of the points to interpolate.
    z: Z coordinates (third axis) of the points to interpolate.
    config: Configuration for trivariate interpolation.

Returns:
    Interpolated values at the specified points.

Raises:
    IndexError: If a point is out of the grid bounds
      and `config.common.bounds_error` is set to `True`.
)";

}  // namespace detail

template <template <class> class Point, typename GridType, typename ResultType>
auto bind_trivariate(nanobind::module_& m) -> void {
  m.def(
      "trivariate",
      [](const GridType& grid, const Eigen::Ref<const Eigen::VectorXd>& x,
         const Eigen::Ref<const Eigen::VectorXd>& y, const nanobind::object& z,
         const config::geometric::Trivariate& config) -> Vector<ResultType> {
        return detail::trivariate<Point, GridType, ResultType>(grid, x, y, z,
                                                               config);
      },
      nanobind::arg("grid"), nanobind::arg("x"), nanobind::arg("y"),
      nanobind::arg("z"), nanobind::arg("config"),
      detail::kTrivariateDocstring);
}

}  // namespace pyinterp::geometric::pybind

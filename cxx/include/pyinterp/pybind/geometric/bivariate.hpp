// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include "pyinterp/broadcast.hpp"
#include "pyinterp/config/geometric.hpp"
#include "pyinterp/eigen.hpp"
#include "pyinterp/math/interpolate/geometric/bivariate.hpp"
#include "pyinterp/math/interpolate/interpolation_result.hpp"
#include "pyinterp/parallel_for.hpp"
#include "pyinterp/pybind/grid.hpp"

namespace pyinterp::geometric::pybind {

/// @brief Alias for the two-dimensional grid type
/// @tparam DataType Data type stored in the grid
template <typename DataType>
using Grid2D = pyinterp::pybind::Grid2D<DataType>;

namespace detail {

/// @brief Alias for the interpolation result type
template <typename T>
using InterpolationResult = math::interpolate::InterpolationResult<T>;

/// @brief Single-point bivariate interpolation
/// @tparam Point Point type (e.g., Point2D)
/// @tparam DataType Data type stored in the grid
/// @tparam ResultType Result type of the interpolation
/// @param[in] grid 2D grid containing data to interpolate
/// @param[in] x X coordinate of the point to interpolate
/// @param[in] y Y coordinate of the point to interpolate
/// @param[in] interpolator Bivariate interpolator
/// @param[in] bounds_error Whether to raise an error for out-of-bounds points
/// @return Interpolated value at the specified point
template <template <class> class Point, typename DataType, typename ResultType>
[[nodiscard]] auto bivariate_single(
    const Grid2D<DataType>& grid, const double x, const double y,
    const math::interpolate::geometric::Bivariate<Point, ResultType>*
        interpolator,
    const bool bounds_error) -> InterpolationResult<ResultType> {
  // Early exit if out of bounds
  auto x_indexes = grid.template find_indexes<0>(x, bounds_error);
  if (!x_indexes.has_value()) {
    return {std::nullopt};
  }

  auto y_indexes = grid.template find_indexes<1>(y, bounds_error);
  if (!y_indexes.has_value()) {
    return {std::nullopt};
  }

  auto [ix0, ix1] = *x_indexes;
  auto [iy0, iy1] = *y_indexes;

  // Cache axis references (micro-optimization)
  const auto& x_axis = grid.template axis<0>();
  const auto& y_axis = grid.template axis<1>();

  // Fetch grid values once (avoid repeated calls)
  const auto v00 = static_cast<ResultType>(grid.value(ix0, iy0));
  const auto v01 = static_cast<ResultType>(grid.value(ix0, iy1));
  const auto v10 = static_cast<ResultType>(grid.value(ix1, iy0));
  const auto v11 = static_cast<ResultType>(grid.value(ix1, iy1));

  // Construct points
  const auto x0 = static_cast<ResultType>(x_axis.coordinate_value(ix0));
  const auto x1 = static_cast<ResultType>(x_axis.coordinate_value(ix1));
  const auto y0 = static_cast<ResultType>(y_axis.coordinate_value(iy0));
  const auto y1 = static_cast<ResultType>(y_axis.coordinate_value(iy1));

  const auto p = Point<ResultType>(
      static_cast<ResultType>(x_axis.normalize_coordinate(x, x0)),
      static_cast<ResultType>(y));
  const auto p0 = Point<ResultType>(x0, y0);
  const auto p1 = Point<ResultType>(x1, y1);

  // Interpolate
  const auto result = interpolator->evaluate(p, p0, p1, v00, v01, v10, v11);

  return {static_cast<ResultType>(result)};
}

constexpr const char* const kBivariateDocstring = R"(
Perform bivariate interpolation on a 2D grid.

Args:
    grid: 2D grid containing data to interpolate.
    x: X coordinates of the points to interpolate.
    y: Y coordinates of the points to interpolate.
    config: Configuration for bivariate interpolation.

Returns:
    Interpolated values at the specified points.

Raises:
    IndexError: If a point is out of the grid bounds
      and `config.common.bounds_error` is set to `True`.
)";

}  // namespace detail

/// @brief Vectorized bivariate interpolation
/// @tparam Point Point type (e.g., Point2D)
/// @tparam DataType Data type stored in the grid
/// @tparam ResultType Result type of the interpolation
/// @param[in] grid 2D grid containing data to interpolate
/// @param[in] x X coordinates of the points to interpolate
/// @param[in] y Y coordinates of the points to interpolate
/// @param[in] config Configuration for bivariate interpolation
/// @return Interpolated values at the specified points
template <template <class> class Point, typename DataType, typename ResultType>
[[nodiscard]] auto bivariate(const Grid2D<DataType>& grid,
                             const Eigen::Ref<const Eigen::VectorXd>& x,
                             const Eigen::Ref<const Eigen::VectorXd>& y,
                             const config::geometric::Bivariate& config)
    -> Vector<ResultType> {
  broadcast::check_eigen_shape("x", x, "y", y);

  // Create interpolator once (outside parallel region)
  auto interpolator =
      math::interpolate::geometric::make_interpolator<Point, ResultType>(
          config.spatial().method(), config.spatial().exponent());
  const auto* interpolator_ptr = interpolator.get();

  Vector<ResultType> result(x.size());
  result.setConstant(std::numeric_limits<ResultType>::quiet_NaN());

  parallel_for(
      x.size(),
      [&](const int64_t start, const int64_t end) {
        for (int64_t ix = start; ix < end; ++ix) {
          auto interpolated_value =
              detail::bivariate_single<Point, DataType, ResultType>(
                  grid, x[ix], y[ix], interpolator_ptr,
                  config.common().bounds_error());

          if (interpolated_value.has_value()) {
            result[ix] = *interpolated_value.value;
          }
        }
      },
      config.common().num_threads());

  return result;
}

/// @brief Bind bivariate interpolation function to Python module
/// @tparam Point Point type (e.g., Point2D)
/// @tparam DataType Data type stored in the grid
/// @tparam ResultType Result type of the interpolation
/// @param[in,out] m Python module to bind the function to
template <template <class> class Point, typename DataType, typename ResultType>
auto bind_bivariate(nanobind::module_& m) -> void {
  m.def(
      "bivariate",
      [](const Grid2D<DataType>& grid,
         const Eigen::Ref<const Eigen::VectorXd>& x,
         const Eigen::Ref<const Eigen::VectorXd>& y,
         const config::geometric::Bivariate& config) -> Vector<ResultType> {
        return bivariate<Point, DataType, ResultType>(grid, x, y, config);
      },
      nanobind::arg("grid"), nanobind::arg("x"), nanobind::arg("y"),
      nanobind::arg("config"), detail::kBivariateDocstring,
      nanobind::call_guard<nanobind::gil_scoped_release>());
}

}  // namespace pyinterp::geometric::pybind

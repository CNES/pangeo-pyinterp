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
#include "pyinterp/pybind/geometric/outer.hpp"
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
    const math::interpolate::geometric::Bivariate<Point, ResultType>&
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
  // Normalize x1 relative to x0 so that, on a periodic axis, the upper bracket
  // wrapped past the period boundary (e.g. 0 read as 360) keeps the cell
  // monotonic; otherwise the interpolation weight collapses across the seam.
  const auto x1 = static_cast<ResultType>(
      x_axis.normalize_coordinate(x_axis.coordinate_value(ix1), x0));
  const auto y0 = static_cast<ResultType>(y_axis.coordinate_value(iy0));
  const auto y1 = static_cast<ResultType>(y_axis.coordinate_value(iy1));

  const auto p = Point<ResultType>(
      static_cast<ResultType>(x_axis.normalize_coordinate(x, x0)),
      static_cast<ResultType>(y));
  const auto p0 = Point<ResultType>(x0, y0);
  const auto p1 = Point<ResultType>(x1, y1);

  // Interpolate
  const auto result = interpolator.evaluate(p, p0, p1, v00, v01, v10, v11);

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

constexpr const char* const kBivariateOuterDocstring = R"(
Perform bivariate interpolation on the grid defined by 1D coordinates.

The points to interpolate at are all the combinations of the coordinates
``x`` and ``y`` (outer product). Each coordinate is located once on its axis,
which is much faster than interpolating the flattened points with
:py:func:`bivariate`, and gives the same results.

Args:
    grid: 2D grid containing data to interpolate.
    x: X coordinates of the grid to interpolate at.
    y: Y coordinates of the grid to interpolate at.
    config: Configuration for bivariate interpolation.

Returns:
    Interpolated values, flattened with ``x`` varying fastest: reshape them
    to ``(x.size, y.size)`` in Fortran order.

Raises:
    IndexError: If a coordinate is out of the grid bounds
      and `config.common.bounds_error` is set to `True`.
)";

}  // namespace detail

/// @brief Bivariate interpolation on the grid defined by 1D coordinates
/// @tparam Point Point type (e.g., Point2D)
/// @tparam DataType Data type stored in the grid
/// @tparam ResultType Result type of the interpolation
/// @param[in] grid 2D grid containing data to interpolate
/// @param[in] x X coordinates of the grid to interpolate at
/// @param[in] y Y coordinates of the grid to interpolate at
/// @param[in] config Configuration for bivariate interpolation
/// @return Interpolated values at the points (x[i], y[j]), stored at the
/// index i + j * x.size()
template <template <class> class Point, typename DataType, typename ResultType>
[[nodiscard]] auto bivariate_outer(const Grid2D<DataType>& grid,
                                   const Eigen::Ref<const Eigen::VectorXd>& x,
                                   const Eigen::Ref<const Eigen::VectorXd>& y,
                                   const config::geometric::Bivariate& config)
    -> Vector<ResultType> {
  const auto bounds_error = config.common().bounds_error();
  const auto num_threads = config.common().num_threads();
  if (x.size() == 0 || y.size() == 0) {
    return Vector<ResultType>(0);
  }
  const auto x_cells =
      detail::locate_axis<0, ResultType>(grid, x, bounds_error);
  const auto y_cells =
      detail::locate_axis<1, ResultType>(grid, y, bounds_error);

  if (config.spatial().method() ==
      math::interpolate::geometric::InterpolationMethod::kBilinear) {
    using Bilinear = math::interpolate::geometric::Bilinear<Point, ResultType>;
    // The relative positions in the cells only depend on the axes
    const auto weights = [](const auto& cells) {
      std::vector<ResultType> result(cells.size());
      for (size_t ix = 0; ix < cells.size(); ++ix) {
        const auto& cell = cells[ix];
        if (cell.is_valid()) {
          result[ix] = Bilinear::weight(cell.value, cell.lower, cell.upper);
        }
      }
      return result;
    };
    const auto t = weights(x_cells);
    const auto u = weights(y_cells);

    return detail::evaluate_outer<ResultType>(
        x.size(), y.size(),
        [&](const int64_t row, const int64_t begin, const int64_t end,
            ResultType* output) -> void {
          const auto& y_cell = y_cells[row];
          if (!y_cell.is_valid()) {
            return;
          }
          const auto u_row = u[row];
          for (auto ix = begin; ix < end; ++ix) {
            const auto& x_cell = x_cells[ix];
            if (!x_cell.is_valid()) {
              continue;
            }
            output[ix] = Bilinear::combine(
                t[ix], u_row,
                static_cast<ResultType>(grid.value(x_cell.i0, y_cell.i0)),
                static_cast<ResultType>(grid.value(x_cell.i0, y_cell.i1)),
                static_cast<ResultType>(grid.value(x_cell.i1, y_cell.i0)),
                static_cast<ResultType>(grid.value(x_cell.i1, y_cell.i1)));
          }
        },
        num_threads);
  }

  auto interpolator_ptr =
      math::interpolate::geometric::make_interpolator<Point, ResultType>(
          config.spatial().method(), config.spatial().exponent());
  const auto& interpolator = *interpolator_ptr;

  return detail::evaluate_outer<ResultType>(
      x.size(), y.size(),
      [&](const int64_t row, const int64_t begin, const int64_t end,
          ResultType* output) -> void {
        const auto& y_cell = y_cells[row];
        if (!y_cell.is_valid()) {
          return;
        }
        for (auto ix = begin; ix < end; ++ix) {
          const auto& x_cell = x_cells[ix];
          if (!x_cell.is_valid()) {
            continue;
          }
          output[ix] = interpolator.evaluate(
              Point<ResultType>(x_cell.value, y_cell.value),
              Point<ResultType>(x_cell.lower, y_cell.lower),
              Point<ResultType>(x_cell.upper, y_cell.upper),
              static_cast<ResultType>(grid.value(x_cell.i0, y_cell.i0)),
              static_cast<ResultType>(grid.value(x_cell.i0, y_cell.i1)),
              static_cast<ResultType>(grid.value(x_cell.i1, y_cell.i0)),
              static_cast<ResultType>(grid.value(x_cell.i1, y_cell.i1)));
        }
      },
      num_threads);
}

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
  auto interpolator_ptr =
      math::interpolate::geometric::make_interpolator<Point, ResultType>(
          config.spatial().method(), config.spatial().exponent());
  const auto& interpolator = *interpolator_ptr;

  Vector<ResultType> result(x.size());
  result.setConstant(std::numeric_limits<ResultType>::quiet_NaN());

  parallel_for(
      x.size(),
      [&](const int64_t start, const int64_t end) {
        for (int64_t ix = start; ix < end; ++ix) {
          auto interpolated_value =
              detail::bivariate_single<Point, DataType, ResultType>(
                  grid, x[ix], y[ix], interpolator,
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

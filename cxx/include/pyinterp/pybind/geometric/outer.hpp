// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "pyinterp/eigen.hpp"
#include "pyinterp/parallel_for.hpp"

namespace pyinterp::geometric::pybind::detail {

/// @brief Cell of an axis that frames a coordinate to interpolate at
///
/// The interpolation on the grid defined by the outer product of 1D
/// coordinates locates each coordinate once on its axis, instead of once per
/// point. The bounds and the coordinate are converted as the point-by-point
/// interpolation does, so that both give the same results.
/// @tparam T Type of the coordinates
template <typename T>
struct AxisCell {
  int64_t i0{-1};  ///< Index of the lower bound, -1 if out of the axis
  int64_t i1{-1};  ///< Index of the upper bound
  T lower{};       ///< Lower bound of the cell
  T upper{};       ///< Upper bound of the cell
  T value{};       ///< Coordinate to interpolate at

  /// @brief Check if the coordinate is framed by the axis
  [[nodiscard]] constexpr auto is_valid() const noexcept -> bool {
    return i0 != -1;
  }
};

/// @brief Locate the coordinates to interpolate at on an axis of a grid
///
/// On the first axis, which can be periodic, the upper bound and the
/// coordinate are normalized relative to the lower bound, as in the
/// point-by-point interpolation.
/// @tparam I Index of the axis
/// @tparam T Type of the coordinates of the cells
/// @tparam GridType Type of the grid
/// @tparam Coordinate Type of the coordinates to locate
/// @param[in] grid Grid to interpolate
/// @param[in] coordinates Coordinates to locate
/// @param[in] bounds_error Whether to raise an error for a coordinate out of
/// the axis
/// @return The cell of each coordinate
template <size_t I, typename T, typename GridType, typename Coordinate>
[[nodiscard]] auto locate_axis(
    const GridType& grid,
    const Eigen::Ref<const Vector<Coordinate>>& coordinates,
    const bool bounds_error) -> std::vector<AxisCell<T>> {
  const auto& axis = grid.template axis<I>();
  std::vector<AxisCell<T>> result(static_cast<size_t>(coordinates.size()));
  for (int64_t ix = 0; ix < coordinates.size(); ++ix) {
    const auto coordinate = coordinates[ix];
    const auto indexes =
        grid.template find_indexes<I>(coordinate, bounds_error);
    if (!indexes.has_value()) {
      continue;
    }
    auto& cell = result[static_cast<size_t>(ix)];
    const auto [i0, i1] = *indexes;
    cell.i0 = i0;
    cell.i1 = i1;
    cell.lower = static_cast<T>(axis.coordinate_value(i0));
    if constexpr (I == 0) {
      cell.upper = static_cast<T>(
          axis.normalize_coordinate(axis.coordinate_value(i1), cell.lower));
      cell.value =
          static_cast<T>(axis.normalize_coordinate(coordinate, cell.lower));
    } else {
      cell.upper = static_cast<T>(axis.coordinate_value(i1));
      cell.value = static_cast<T>(coordinate);
    }
  }
  return result;
}

/// @brief Number of rows of the output processed together
constexpr int64_t kOuterRowBlock = 16;

/// @brief Number of columns of the output processed together
constexpr int64_t kOuterColumnBlock = 256;

/// @brief Evaluate the rows of a grid of points
///
/// The output is stored with the first axis varying fastest: a row holds the
/// values along the first axis for one combination of the coordinates of the
/// other axes. The grid stores its last axis contiguously, so the rows are
/// processed by tiles to reuse the cache lines read from the grid.
/// @tparam ResultType Type of the interpolated values
/// @tparam RowEvaluator Callable as evaluate(row, begin, end, output) that
/// writes the values of the columns [begin, end) of a row to output, which
/// points to the first value of the row
/// @param[in] columns Number of points along the first axis
/// @param[in] rows Number of combinations of the coordinates of the other
/// axes
/// @param[in] evaluate Row evaluator
/// @param[in] num_threads Number of threads to use, 0 for all CPUs
/// @return The interpolated values
template <typename ResultType, typename RowEvaluator>
[[nodiscard]] auto evaluate_outer(const int64_t columns, const int64_t rows,
                                  const RowEvaluator& evaluate,
                                  const int64_t num_threads)
    -> Vector<ResultType> {
  Vector<ResultType> result(columns * rows);
  result.setConstant(std::numeric_limits<ResultType>::quiet_NaN());
  if (result.size() == 0) {
    return result;
  }
  auto* data = result.data();

  // Each thread processes a range of blocks of rows
  const auto row_blocks = (rows + kOuterRowBlock - 1) / kOuterRowBlock;
  parallel_for(
      row_blocks,
      [&](const int64_t start, const int64_t end) {
        for (auto block = start; block < end; ++block) {
          const auto first_row = block * kOuterRowBlock;
          const auto last_row = std::min(first_row + kOuterRowBlock, rows);
          for (int64_t begin = 0; begin < columns; begin += kOuterColumnBlock) {
            const auto end_column =
                std::min(begin + kOuterColumnBlock, columns);
            for (auto row = first_row; row < last_row; ++row) {
              evaluate(row, begin, end_column, data + row * columns);
            }
          }
        }
      },
      num_threads);
  return result;
}

}  // namespace pyinterp::geometric::pybind::detail
